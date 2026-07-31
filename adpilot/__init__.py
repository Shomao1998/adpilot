"""adpilot: 编排层（W4）—— LangGraph 状态机串联全流程 + 出价 Agent + 诊断 + 决策日志。

分层:
    metrics.py    归一化指标 (DayMetrics): CTR/CVR/CPA/ROAS，全程 micros USD。
    decision.py   决策日志 (Decision/DecisionLog): 每个动作的理由/预期/置信度/采纳态。
    bidding.py    出价 Agent: 可解释规则策略（初始出价/边际ROAS再分配/疲劳/超成本）。
    ctr_ranking.py CTR 预排序: LLM 对创意变体打分 -> creative_quality + 初始预算权重。
    diagnosis.py  评估诊断 Agent: 每模拟日的结构化诊断（规则 + 可选 LLM 复述）。
    orchestrator.py LangGraph DAG: setup -> [bid -> simulate -> diagnose] * N 日。

ADR-1: 用状态机而非自由 ReAct Agent —— 广告流程是强结构 DAG，确定性/可断点/
可审计优先。ADR-5: 全程 USD micros。
"""
from adpilot.metrics import DayMetrics
from adpilot.decision import Decision, DecisionLog
from adpilot.bidding import BiddingAgent, BidConfig
from adpilot.ctr_ranking import CtrScore, RankedCreative, rank_creatives
from adpilot.diagnosis import DiagnosisReport, diagnose_day
from adpilot.orchestrator import AdPilotOrchestrator, PilotConfig

__all__ = [
    "DayMetrics",
    "Decision", "DecisionLog",
    "BiddingAgent", "BidConfig",
    "CtrScore", "RankedCreative", "rank_creatives",
    "DiagnosisReport", "diagnose_day",
    "AdPilotOrchestrator", "PilotConfig",
]
