"""看板后端测试: payload 序列化契约 + FastAPI 端点。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from adpilot import AdPilotOrchestrator, PilotConfig
from webapp.payload import build_dashboard_payload
from webapp.server import app


def _run(days=3):
    orch = AdPilotOrchestrator(PilotConfig(total_days=days))
    final = orch.run()
    return build_dashboard_payload(final, orch.log, 25.0), final, orch


class TestPayload:
    def test_shape_and_totals(self):
        payload, final, orch = _run(3)
        assert set(payload) == {"summary", "platforms", "diagnoses", "decisions"}
        assert payload["summary"]["days"] == 3
        assert len(payload["diagnoses"]) == 3
        assert len(payload["platforms"]) == 3
        # 每平台逐日时序完整
        for pl in payload["platforms"]:
            assert len(pl["series"]) == 3
            assert {"spend_usd", "roas", "ctr", "cpa_usd"} <= set(pl["series"][0])
        # 汇总消耗 = 各平台消耗之和
        s = payload["summary"]["total_spend_usd"]
        assert abs(s - sum(pl["totals"]["spend_usd"]
                           for pl in payload["platforms"])) < 0.05

    def test_cpa_null_when_no_conversion(self):
        """无转化的日/平台 CPA 应为 None（前端渲染 '—'），不是 inf。"""
        payload, *_ = _run(3)
        for pl in payload["platforms"]:
            for row in pl["series"]:
                if row["conversions"] == 0:
                    assert row["cpa_usd"] is None

    def test_decisions_include_initial_bid(self):
        payload, *_ = _run(3)
        actions = {d["action"] for d in payload["decisions"]}
        assert "set_initial_bid" in actions

    def test_json_serializable(self):
        import json
        payload, *_ = _run(3)
        json.dumps(payload)  # 不应抛错（无 inf/enum 逃逸）


class TestEndpoints:
    def test_run_endpoint(self):
        client = TestClient(app)
        r = client.get("/api/run?days=3")
        assert r.status_code == 200
        body = r.json()
        assert body["summary"]["days"] == 3

    def test_results_pending_or_ready(self):
        """后台预热: 未就绪返回 202 {pending}，就绪返回 200 payload。"""
        client = TestClient(app)
        r = client.get("/api/results")
        assert r.status_code in (200, 202)
        if r.status_code == 200:
            assert "platforms" in r.json()
        else:
            assert r.json() == {"pending": True}

    def test_index_served(self):
        client = TestClient(app)
        r = client.get("/")
        assert r.status_code == 200
        assert "AdPilot" in r.text

    def test_days_validation(self):
        client = TestClient(app)
        assert client.get("/api/run?days=0").status_code == 422
        assert client.get("/api/run?days=99").status_code == 422


class TestNarratorLang:
    def test_stub_bilingual(self):
        from adpilot.diagnosis import DiagnosisNarrative
        from webapp.server import TemplateNarrator
        user = "领先 meta_sim、垫底 tiktok_sim；综合 ROAS 5.16。"
        zh = TemplateNarrator("zh").complete("", user, DiagnosisNarrative)
        en = TemplateNarrator("en").complete("", user, DiagnosisNarrative)
        assert "效率领先" in zh.narrative
        assert "leads on efficiency" in en.narrative and "meta_sim" in en.narrative


class TestScenarioEndpoints:
    def test_brief(self):
        r = TestClient(app).get("/api/brief")
        assert r.status_code == 200
        assert r.json()["budget"]["currency"] == "USD"

    def test_creatives(self):
        body = TestClient(app).get("/api/creatives").json()
        assert body["n_pass"] >= 1 and body["n_total"] == len(body["items"])
        # 过审项应有缩略图路径
        for it in body["items"]:
            if it["verdict"] == "pass":
                assert it["thumb"] and it["thumb"].startswith("/static/gen/")

    def test_creatives_sorted_pass_first_ctr_desc(self):
        items = TestClient(app).get("/api/creatives").json()["items"]
        passed = [i for i in items if i["verdict"] == "pass"]
        others = [i for i in items if i["verdict"] != "pass"]
        # 过审全部排在未过审前面
        n_pass = len(passed)
        assert all(i["verdict"] == "pass" for i in items[:n_pass])
        assert all(i["verdict"] != "pass" for i in items[n_pass:])
        # 过审内部按 CTR 降序
        scores = [i["ctr_score"] for i in items[:n_pass]]
        assert scores == sorted(scores, reverse=True)

    def test_campaigns(self):
        body = TestClient(app).get("/api/campaigns").json()
        assert len(body["tree"]) == 3
        assert len(body["api_log"]) >= 3
        # API 日志请求/响应是结构化 JSON
        c = body["api_log"][0]
        assert isinstance(c["request"], dict) and isinstance(c["response"], dict)


class TestWizard:
    def test_empty_input_400(self):
        r = TestClient(app).post("/api/wizard", json={"input": "  "})
        assert r.status_code == 400

    def test_offline_returns_503_with_hint(self, monkeypatch):
        """离线（未设 ADPILOT_LIVE）: 503 + 清晰提示，不 500。"""
        monkeypatch.delenv("ADPILOT_LIVE", raising=False)
        r = TestClient(app).post("/api/wizard", json={"input": "便携咖啡机"})
        assert r.status_code == 503
        assert "ADPILOT_LIVE" in r.json()["error"]

    def test_latest_result_links_gallery(self):
        """输入向导生成后，创意画廊页优先返回该结果（联动）；示例 Brief 恒固定。"""
        from webapp.infra import STORE
        STORE.set("latest:gallery", None)
        client = TestClient(app)
        try:
            STORE.set("latest:gallery", {"items": [], "n_total": 0,
                                         "n_pass": 0, "_probe": True})
            assert client.get("/api/creatives").json().get("_probe") is True
            # 示例 Brief 不联动，仍是固定示例
            assert "_probe" not in client.get("/api/brief").json()
        finally:
            STORE.set("latest:gallery", None)
        # 清空后回落到离线示例
        assert "_probe" not in client.get("/api/creatives").json()
