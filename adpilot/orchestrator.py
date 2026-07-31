"""LangGraph 编排（ADR-1: 状态机而非自由 Agent）。

广告投放是强结构 DAG，用状态机保证确定性、可断点、可审计。图结构:

    START -> setup -> simulate -> diagnose -> route
                                      route --(还有天数)--> bid -> simulate -> ...
                                      route --(跑满)-------> END

每模拟日: bid（日终决策，D0 的初始出价在 setup 完成）-> simulate（AdSim 跑一天）
-> diagnose（结构化诊断）。出价 Agent 的调整改 PlatformPlan，simulate 前同步进 DB。

诊断/决策全部留痕在 state，跑完可导出决策日志时间线（W5 看板的数据源）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from adpilot.bidding import BiddingAgent, BidConfig, PlatformPlan
from adpilot.decision import Decision, DecisionLog
from adpilot.diagnosis import DiagnosisReport, diagnose_day
from adpilot.metrics import DayMetrics
from adsim.auction import make_platform_markets
from adsim.models import Ad, AdGroup, Base, Campaign, Platform
from adsim.simulate import AdSimEngine, usd_to_micros

MICROS = 1_000_000


@dataclass
class PilotConfig:
    total_days: int = 7
    start_date: date = date(2026, 7, 1)
    platforms: tuple[Platform, ...] = (
        Platform.META_SIM, Platform.GOOGLE_SIM, Platform.TIKTOK_SIM)
    daily_budget_usd: float = 100.0
    # 每平台创意质量分（来自 CTR 预排序；给定则据此建多条 Ad）
    creative_qualities: dict[Platform, list[float]] = field(default_factory=dict)
    bid: BidConfig = field(default_factory=BidConfig)
    auto_approve: bool = True     # 演示: 自动放权跑全自动闭环
    diagnosis_llm: object = None  # LLMClient | None: 给定则诊断附 LLM 自然语言复述
    lang: str = "zh"              # 诊断复述语言（zh/en）
    seed: int = 42
    base_fit_mu: float = float(np.log(4.0))
    base_fit_sigma: float = 0.7


class PilotState(TypedDict):
    day_index: int
    total_days: int
    plans: dict
    ids: dict
    history: dict
    diagnoses: list


class AdPilotOrchestrator:
    """持有 DB 引擎与 AdSim 引擎；LangGraph 节点是本类的绑定方法。"""

    def __init__(self, config: PilotConfig | None = None):
        self.cfg = config or PilotConfig()
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)
        markets = {Platform(k): v for k, v in make_platform_markets(
            self.cfg.base_fit_mu, self.cfg.base_fit_sigma).items()}
        self.sim = AdSimEngine(markets, seed=self.cfg.seed)
        self.cfg.bid.lang = self.cfg.lang     # 决策文本跟随 UI 语言
        self.bidder = BiddingAgent(self.cfg.bid)
        self.log = DecisionLog()
        self.graph = self._build_graph()

    # ---------- 图定义 ----------
    def _build_graph(self):
        g = StateGraph(PilotState)
        g.add_node("setup", self._setup)
        g.add_node("bid", self._bid)
        g.add_node("simulate", self._simulate)
        g.add_node("diagnose", self._diagnose)
        g.add_edge(START, "setup")
        g.add_edge("setup", "simulate")
        g.add_edge("bid", "simulate")
        g.add_edge("simulate", "diagnose")
        g.add_conditional_edges("diagnose", self._route,
                                {"continue": "bid", "done": END})
        return g.compile()

    def run(self) -> PilotState:
        init: PilotState = dict(
            day_index=0, total_days=self.cfg.total_days,
            plans={}, ids={}, history={}, diagnoses=[])
        # recursion_limit 要够大: 每天 3 节点 × total_days + 余量
        return self.graph.invoke(
            init, config={"recursion_limit": self.cfg.total_days * 6 + 20})

    # ---------- 节点 ----------
    def _setup(self, state: PilotState) -> dict:
        cfg = self.cfg
        plans: dict[Platform, PlatformPlan] = {}
        ids: dict[Platform, dict] = {}
        with Session(self.engine) as s:
            for p in cfg.platforms:
                c = Campaign(platform=p, name=f"pilot-{p.value}",
                             daily_budget_micros=usd_to_micros(cfg.daily_budget_usd),
                             external_ref=f"ext-{p.value}")
                s.add(c); s.flush()
                g = AdGroup(campaign_id=c.id, name="ag",
                            bid_micros=usd_to_micros(5.0))  # 占位，D0 初始出价覆盖
                s.add(g); s.flush()
                quals = cfg.creative_qualities.get(p, [0.6, 0.5, 0.4])
                ads = [Ad(ad_group_id=g.id, creative_id=f"{p.value}-cr{i}",
                          creative_quality=q) for i, q in enumerate(quals)]
                s.add_all(ads); s.flush()
                plans[p] = PlatformPlan(
                    platform=p, bid_micros=g.bid_micros,
                    budget_micros=c.daily_budget_micros)
                ids[p] = dict(campaign_id=c.id, ad_group_id=g.id,
                              ad_ids=[a.id for a in ads])
            s.commit()

        # D0 初始出价
        decisions = self.bidder.initial_bids(plans)
        self.bidder.apply(plans, decisions, auto_approve=cfg.auto_approve)
        for d in decisions:
            self.log.add(d)
        return dict(plans=plans, ids=ids)

    def _bid(self, state: PilotState) -> dict:
        # 进入新的一天（router 不能改 state，日递增放在这里）
        day = state["day_index"] + 1
        plans = state["plans"]
        decisions = self.bidder.daily_decisions(day, plans, state["history"])
        self.bidder.apply(plans, decisions, auto_approve=self.cfg.auto_approve)
        for d in decisions:
            self.log.add(d)
        return dict(day_index=day, plans=plans)

    def _simulate(self, state: PilotState) -> dict:
        cfg = self.cfg
        day = state["day_index"]
        sim_date = cfg.start_date + timedelta(days=day)
        plans = state["plans"]
        ids = state["ids"]
        history = state["history"]
        with Session(self.engine) as s:
            for p in cfg.platforms:
                plan = plans[p]
                if plan.status != "active":
                    continue
                # 同步 plan -> DB 行（出价 Agent 的调整在此生效）
                camp = s.get(Campaign, ids[p]["campaign_id"])
                ag = s.get(AdGroup, ids[p]["ad_group_id"])
                camp.daily_budget_micros = plan.budget_micros
                ag.bid_micros = plan.bid_micros
                ads = [s.get(Ad, aid) for aid in ids[p]["ad_ids"]]
                s.flush()
                st = self.sim.run_day(s, sim_date, camp, ag, ads)
                history.setdefault(p, []).append(DayMetrics(
                    platform=p, day_index=day, impressions=st.wins,
                    clicks=st.clicks, conversions=st.conversions,
                    spend_micros=st.spend_micros,
                    revenue_micros=st.revenue_micros))
        return dict(history=history)

    def _diagnose(self, state: PilotState) -> dict:
        day = state["day_index"]
        today = {p: h[-1] for p, h in state["history"].items()
                 if h and h[-1].day_index == day}
        rpt = diagnose_day(day, today, state["history"],
                           self.log.for_day(day), self.cfg.bid.target_cpa_usd,
                           llm=self.cfg.diagnosis_llm, lang=self.cfg.lang)
        diagnoses = state["diagnoses"] + [rpt]
        return dict(diagnoses=diagnoses)

    def _route(self, state: PilotState) -> str:
        # 刚诊断完 day_index 这一天；还有下一天则继续（递增在 _bid 里做）
        return "continue" if state["day_index"] + 1 < state["total_days"] else "done"
