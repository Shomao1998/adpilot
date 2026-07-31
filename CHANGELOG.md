<a id="en"></a>
# Changelog

**English** · [中文](#zh)

Iteration is driven by "does it actually run" — every change was verified in the local live environment on port 8001 (DeepSeek + fal).

## 2026-07 · Going live + end-to-end polish (this round)

### Added

- **DeepSeek provider**: `ADPILOT_PROVIDER=deepseek`, OpenAI-compatible, far cheaper than Claude, now the default live provider. `make_live_llm()` picks `claude` / `deepseek` by env var.
- **Text-to-image (fal.ai)**: three-tier imagery fallback `A scrape hero image → B text-to-image → C gradient template`. Both `FalImageSource` (FLUX.1-schnell) and a generic `OpenAICompatImageSource` (SiliconFlow, etc.) are implemented, selected by `ADPILOT_IMAGE_PROVIDER`. Verified to produce real product images even from a bare keyword.
- **Data lineage**: creative CTR scores from the wizard feed the delivery simulation via `creative_qualities` — Decision Log / cross-platform dashboard / Campaign all become "delivering *these* creatives" instead of input-independent fixed data. Pilot results carry a `source: live|sample` tag.
- **Funnel navigation**: the five flat tabs became a numbered stepper `① input → ② creatives → ③ launch → ④ simulate & tune → ⑤ compare`, with a subtitle describing the current step and a whole-platform "Your campaign / Sample data" label.
- **Dashboard trend annotations**: a "how to read" box (auto-surfaces the biggest mover + number of agent adjustments), dashed adjustment-day markers on the line charts, per-platform start→end trend bars (green up / red down), and ROAS delta arrows in the summary table.
- **Image-aware CTR scoring + flip cards**: `visual_appeal` now comes from local CV metrics (contrast / colorfulness / sharpness) analyzing the real creative image, overriding the LLM's text-based guess; the scorer is a pluggable `VisualScorer` protocol (swap in a VLM later without touching the layers above). Creative cards flip in 3D on hover to reveal the four-dim breakdown + the source of the visual score.
- **Whole-platform bilingual**: 中文 / English dropdown; UI + LLM output (brief / copy / review reasons / diagnosis) unify to the chosen language.
- **Cloud architecture (locally simulated)**: `docker-compose.cloud.yml` reproduces a cloud-native topology on one machine — stateless **web** + **worker** containers, **Redis** (state + RQ queue), **Postgres** (run persistence), **MinIO** (S3-compatible image store). A single `ADPILOT_BACKEND=local|cloud` switch keeps local mode byte-for-byte identical; state/queue/blob are swappable protocols (`webapp/infra.py`). Cloud-agnostic, with AWS / GCP / **Azure** mappings and an Azure Container Apps + Service Bus + KEDA target documented in the README. API providers (DeepSeek, fal.ai) chosen mainly for cost, each behind a swappable protocol.

### Fixed

- **CJK / emoji tofu (▯) in rendered creatives**: Pillow's default font can't even render Chinese. Switched to a CJK font fallback chain (Noto / Hiragino / STHeiti / Arial Unicode), stripped emoji before layout, and made line-wrapping full-width-aware to avoid overflow.
- **Chinese mode producing English copy**: `lang_note` is now a symmetric strong directive for both languages; removed the conflicting "English-first" line from the copy prompt; added CTA language fallback ("Shop Now" → "立即选购" in Chinese creatives).
- **English mode diagnosis leaking Chinese**: the diagnosis system prompt is now language-native (no hard-coded "write in Chinese"), plus a CJK guard — if the LLM still leaks Chinese in English mode, it falls back to a deterministic English sentence.
- **Campaign page not linking**: the wizard failed to clear the frontend `campaigns` cache, so step 3 still showed the old sample; now the whole scenario cache is cleared.
- **Creative gallery not following the URL**: wizard results are written to `_latest`, and gallery / Campaign / decision log are built live from the current input.

### Tests

- Case count **115 → 153**, all passing (`pytest`, in-memory SQLite, `FakeLLM` at zero API cost).

## 2026-07-13 · W1–W6 baseline

- Six-week milestones (event-sourced simulator / three-platform adapters / creative pipeline / LangGraph orchestration / bidding agent / dashboard / Docker) fully built.
- Five real LLM paths (intent / copy / L2 review / CTR ranking / diagnosis) verified on `claude-opus-4-8`; see `LIVE-CHECKLIST.md`.
- USD-micros pricing throughout, no FX / currency conversion (ADR-5).

---

<a id="zh"></a>
# 更新记录

[English](#en) · **中文**

本项目按"能不能真跑通"驱动迭代——每条改动都在本地 8001 实时环境（DeepSeek + fal）验证过。

## 2026-07 · 实时化 + 端到端体验打磨（本轮）

### 新增

- **DeepSeek provider**：`ADPILOT_PROVIDER=deepseek`，OpenAI 兼容、成本远低于 Claude，成为默认实时 provider。`make_live_llm()` 按环境变量选 `claude` / `deepseek`。
- **文生图接入（fal.ai）**：创意配图三级兜底 `A 抓商品主图 → B 文生图 → C 渐变模板`。`FalImageSource`（FLUX.1-schnell）与通用 `OpenAICompatImageSource`（硅基流动等）均实现，`ADPILOT_IMAGE_PROVIDER` 选择。实测纯关键词也能出真实产品图。
- **数据血缘打通**：输入向导生成的创意 CTR 分经 `creative_qualities` 喂进投放模拟——决策日志 / 跨平台看板 / Campaign 全部变成"投放你这批创意"的结果，而非与输入无关的固定数据。pilot 结果带 `source: live|sample` 来源标记。
- **漏斗式导航**：五个平铺 tab 改成 `① 输入 → ② 创意 → ③ 上线 → ④ 每日模拟+调优 → ⑤ 跨平台对比` 的带序号步骤条，顶部副标题描述当前步 + 全平台"本次投放 / 示例数据"标注。
- **看板趋势注解**：跨平台看板加读图提示框（自动点出最大变化 + Agent 调整次数）、折线上 Agent 出价/预算调整日虚线竖标、每平台起止趋势条（涨绿跌红）、汇总表 ROAS 涨跌箭头。
- **图像感知 CTR 打分 + 翻转卡**：`visual_appeal` 维改由本地 CV 指标（对比度 / 色彩丰富度 / 清晰度）分析真实创意图得出，覆盖 LLM 的文本猜测；打分器做成 `VisualScorer` 协议可插拔（将来换 VLM 不动上层）。创意卡片悬浮 3D 翻转，背面展示四维打分明细 + 视觉分来源。
- **全平台语言切换**：中 / English 下拉，界面 + LLM 生成内容（brief / 文案 / 审核理由 / 诊断复述）随选择的语言统一输出。
- **云架构（本地模拟）**：`docker-compose.cloud.yml` 在单机重现云原生拓扑——无状态 **web** + **worker** 容器、**Redis**（状态 + RQ 队列）、**Postgres**（结果持久化）、**MinIO**（S3 兼容图片存储）。一个 `ADPILOT_BACKEND=local|cloud` 开关让本地模式逐字节不变；状态/队列/对象存储都是可换协议（`webapp/infra.py`）。云无关，README 附 AWS / GCP / **Azure** 映射及 Azure Container Apps + Service Bus + KEDA 落地方案。API 供应商（DeepSeek、fal.ai）主要按成本选，各藏在可换协议后。

### 修复

- **创意图渲染中文/emoji 乱码**：Pillow 默认字体连中文都渲不了（豆腐块 ▯）。换成 CJK 字体回退链（Noto / Hiragino / STHeiti / Arial Unicode），排版前剔除 emoji，折行按中文全角自适应防溢出。
- **中文模式文案出英文**：`lang_note` 改为中英对称强约束；删掉文案提示词里"英文为主"的冲突句；CTA 加语言兜底（中文创意里的 "Shop Now" → "立即选购"）。
- **英文模式诊断复述漏中文**：诊断系统提示词改为语言原生（不再硬写"用中文"），加 CJK 检测兜底——英文模式若 LLM 仍漏中文，退回确定性英文句。
- **Campaign 页不联动**：向导跑完漏清 `campaigns` 前端缓存，导致第 3 步仍显示旧示例；改为清整个场景缓存。
- **创意画廊不随 URL 联动**：向导结果写入 `_latest`，画廊 / Campaign / 决策日志按本次输入实时构建。

### 测试

- 用例数 **115 → 153**，全部通过（`pytest`，内存 SQLite，`FakeLLM` 零 API 成本）。

## 2026-07-13 · W1–W6 基线

- 六周里程碑（事件溯源模拟器 / 三平台适配器 / 创意管线 / LangGraph 编排 / 出价 Agent / 看板 / Docker）机器全部完成。
- 五个真实 LLM 路径（意图 / 文案 / L2 审核 / CTR 预排序 / 诊断）用 `claude-opus-4-8` 实测通过，清单见 `LIVE-CHECKLIST.md`。
- 全程 USD micros 计价，无汇率 / 币种转换（ADR-5）。
