# --live 真实 LLM 验证清单

拿到 `ANTHROPIC_API_KEY` 后,按本清单一次性验证所有真实 LLM 调用路径。
在此之前全部走 FakeLLM / stub 离线跑通(104 测试通过),真实 LLM 只需验证
**提示词质量 + 结构化输出契约**,机器逻辑已验证过。

## 0. 前置设置(一次)

```bash
cd adsim
pip install -e '.[llm]'            # 装 anthropic SDK(其余依赖 -e . 已装)
export ANTHROPIC_API_KEY=sk-ant-...  # 或 ant auth login 后可留空
python -c "import anthropic; print(anthropic.__version__)"   # 确认 SDK 就位
```

模型: 全部走 `claude-opus-4-8`(见 `adcreative/llm.py:ClaudeLLM`)。
预计总成本: **< $1**(5 处调用,每处一到数次结构化输出,输入输出都短)。

## 1. 五个真实 LLM 调用点(全部经 `LLMClient` 协议 + Pydantic 结构化输出)

| # | 调用点 | 系统提示词 | 输出 schema | 被哪个 demo 触发 |
|---|--------|-----------|------------|-----------------|
| 1 | `adcreative/intent.py` `parse_brief` | `INTENT_SYSTEM` | `Brief` | demo_w3 `--live` |
| 2 | `adcreative/copywriter.py` `generate_copy` | `COPY_SYSTEM` | `CopyBatch` | demo_w3 `--live` |
| 3 | `adcreative/review.py` `review_variant`(L2) | `REVIEW_SYSTEM` | `ReviewVerdict` | demo_w3 `--live` |
| 4 | `adpilot/ctr_ranking.py` `rank_creatives` | `CTR_SYSTEM` | `CtrScoreBatch` | demo_w4 `--live` |
| 5 | `adpilot/diagnosis.py` `_narrate` | `DIAG_SYSTEM` | `DiagnosisNarrative` | demo_w4 `--live` |

两个 demo 的 `--live` 合起来覆盖全部五个。

## 2. 验证命令

### A. 创意管线(调用 1/2/3)—— W3 验收

```bash
# 用一个真实商品 URL(Shopify/独立站商品页均可)
python scripts/demo_w3.py --live "https://<真实商品页>"
```

**期望 / 通过判据**:
- `[1/3] 意图解析`: 打印出的产品名/卖点/价格与页面**实际内容一致**(不是编造),
  缺预算时有一条追问且用了默认值兜底
- `[2/3]`: 三平台各产出文案,`meta`/`tiktok` 有 headline、`google` 是 RSA 资产结构
  (schema 已强制 30/90 字符和资产数量,若 LLM 违规会自动重试——观察是否最终产出)
- reject/revise 回流可用(若真实文案干净可能不触发,可故意用敏感品类 URL 试)
- **W3 达标**: 末行 `过审创意 N 组 (要求 ≥6)`,N ≥ 6
- 无 `ValidationError` 逃逸(结构化输出契约成立)

> 注: 图片走本地 Pillow 排版(降级预案),不消耗 API,`demo_w3_out/` 下应有 PNG。

### B. CTR 预排序 + 诊断复述(调用 4/5)—— W4 验收

```bash
python scripts/demo_w4.py --live --days 7
```

**期望 / 通过判据**:
- `[1/3] CTR 预排序`: 每平台三个真实文案(见 `_CTR_VARIANT_TEXT`)被真打分,
  Top 变体应是三个里**明显更完整/更具体**的那条(v0),弱文案(v2,如 "coffee maker for sale")
  排最后——验证相对排序合理
- `[2/3]`: 每天诊断的 `诊断:` 一行是**真实 Claude 生成的自然语言**(不再是 stub 模板),
  内容应与该天的结构化事实(领先/垫底平台、ROAS)一致,不编造数字
- `[3/3]`: 决策日志时间线含初始出价 + 至少一次预算再分配/疲劳触发
- 末尾 `W4 验收通过: ... N 次生效的自动调整`,N ≥ 1

## 3. 逐项打勾

- [ ] `pip install -e '.[llm]'` 成功,`import anthropic` 通过
- [ ] **调用1 意图解析**: 抽取内容与页面一致,无编造
- [ ] **调用2 文案生成**: 三平台分化正确,RSA 资产合规,无 ValidationError 逃逸
- [ ] **调用3 L2 审核**: 干净文案 pass / 问题文案给出 revise 理由(可用敏感品类验证)
- [ ] **W3**: 过审 ≥6 组,`demo_w3_out/` 有 PNG
- [ ] **调用4 CTR 预排序**: 强文案排前、弱文案排后,权重 Top3 约 70%
- [ ] **调用5 诊断复述**: narrative 是真实生成、与结构化事实一致、不编数字
- [ ] **W4**: 7 日闭环跑通,≥1 次生效自动调整
- [ ] (可选)故意用敏感品类/夸大文案,确认 reject/revise 回流真的触发

## 4. 排障

- **401 / 认证失败**: `ANTHROPIC_API_KEY` 未设或失效;或已 `ant auth login` 但 key 变量
  为空字符串覆盖了 profile——`unset ANTHROPIC_API_KEY` 再试
- **ValidationError 逃逸**(重试后仍失败): 提示词约束不够,看 `complete_with_retry`
  已把错误回喂重试一次仍不过;调对应模块的系统提示词(如 RSA 字符限制说得更死)
- **结构化输出 400**: 确认模型是 `claude-opus-4-8`(支持 structured outputs);
  `messages.parse(..., output_format=Schema)` 见 `adcreative/llm.py:34`
- **诊断 narrative 为空**: `_narrate` 对 LLM 失败做了 try/except 吞掉(设计如此,
  不阻断闭环)——若期望有内容却空了,单独跑该调用看真实报错

## 5. 验证后

全部打勾即表示 W1–W4 的真实 LLM 路径验证完毕。此前离线已验证的机器逻辑
(拍卖/事件流/报表/审核规则/出价规则/编排状态机)不受影响。下一步进 W5 Dashboard。
