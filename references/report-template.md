# 可转债日报模板 + LLM Prompt

## Markdown 日报模板

```
# 可转债每日分析报告 — YYYY-MM-DD

> 扫描时间: YYYY-MM-DD HH:MM:SS | 全市场转债: N 只 | 入选: M 只

## 1. 市场概览
## 2. 双低策略精选
## 3. 高YTM防御组合
## 4. 条款事件监控
   4.1 强赎预警 (含连续日计数)
   4.2 下修候选
   4.3 回售触发 (含连续日计数)
   4.4 临近到期
## 5. 正股联动精选
## 6. 行业分布
## 7. 信用风险告警
## 8. 综合评分排名 (Top N)
## 9. AI逐券研判
## 10. 数据溯源

## 免责声明
```

## LLM 逐券分析 Prompt (默认)

可通过 `config["llm"]["prompt_template"]` 自定义。

```
你是一位资深可转债分析师，专精A股转债的估值、条款博弈和正股联动分析。
请基于以下数据，对该可转债做一次专业的逐券分析。

## 转债基本信息
- 转债名称：{bond_name}（{bond_code}）
- 正股：{stock_name}（{stock_code}）
- 转债价格：{cb_price:.2f} 元
- 转股溢价率：{premium_rate:.1f}%
- 转股价值：{conversion_value:.2f}
- 双低值：{double_low:.1f}
- 到期收益率：{ytm:.2f}%

## 信号检测结果
{signal_summary}

## 评分
- 估值维度：{val_score:.0f}/100
- 条款维度：{clause_score:.0f}/100
- 正股联动：{link_score:.0f}/100
- 市场结构：{struct_score:.0f}/100
- 综合评分：{composite:.0f}/100（{grade}）

## 风险提示
{risk_summary}

## 要求
1. **一句话定性**：这只转债当前处于什么状态？（进攻/防守/博弈/回避）
2. **估值分析**：当前价格和溢价率是否合理？双低值有吸引力吗？
3. **条款研判**：是否有强赎风险、下修可能、或回售保护？
4. **正股联动**：正股走势对转债的影响方向。
5. **综合建议**：对这种转债，应该关注什么？用什么策略？
  （注意用「值得关注」「可跟踪」等措辞，禁止推荐买入/卖出/目标价）
6. 控制在 200-350 字。
```

### 占位符说明

| 占位符 | 来源 | 说明 |
|--------|------|------|
| `{bond_name}` | ScoreResult.bond_name | 转债名称 |
| `{bond_code}` | ScoreResult.bond_code | 转债代码 |
| `{stock_name}` | ScoreResult.stock_name | 正股名称 |
| `{stock_code}` | ScoreResult.stock_code | 正股代码 |
| `{cb_price}` | ScoreResult.cb_price | 转债价格 (元) |
| `{premium_rate}` | ScoreResult.premium_rate | 转股溢价率 (%) |
| `{conversion_value}` | ScoreResult.conversion_value | 转股价值 |
| `{double_low}` | ScoreResult.double_low | 双低值 |
| `{ytm}` | ScoreResult.ytm × 100 | 到期收益率 (%)，LLM 模板中直接显示为百分比 |
| `{signal_summary}` | ScoreResult.to_summary_dict() | Top 3 触发信号，用「、」连接 |
| `{risk_summary}` | ScoreResult.to_summary_dict() | 风险标记，用「；」连接 |
| `{val_score}` | ScoreResult.valuation_score | 估值维度分 |
| `{clause_score}` | ScoreResult.clause_score | 条款维度分 |
| `{link_score}` | ScoreResult.linkage_score | 正股联动分 |
| `{struct_score}` | ScoreResult.structure_score | 市场结构分 |
| `{composite}` | ScoreResult.composite_score | 综合评分 |
| `{grade}` | ScoreResult.grade | 等级 (A+/A/B+/B/C/D/E) |

## 规则备选分析 (Fallback)

当 LLM 不可用时（无 API key 或 API 调用全部失败），`CBAnalyst._fallback_analysis()` 基于规则生成分析：

- **等级 A+/A**：综合信号偏积极
- **等级 B+/B**：信号温和偏多，值得跟踪
- **等级 C**：多空信号均衡，观望为主
- **等级 D/E**：信号偏弱，注意风险
- **折价 (premium < 0)**：提示转股套利空间
- **溢价率偏高**：提示进攻性受限
- **双低值 < 130**：提示具备性价比
- **YTM > 3%**：提示债底保护
- **价格 < 100**：提示回售和下修博弈机会

## LLM 后端架构

```
CBAnalyst
  ├── LLMBackend (Protocol)
  │     └── generate(prompt, model, max_tokens) -> str
  ├── AnthropicBackend (默认)
  │     └── Anthropic/DeepSeek API (兼容端点)
  └── 备选: _fallback_analysis() 规则引擎
```

配置 `config["llm"]["provider"]` 可切换后端（当前支持 `"anthropic"`）。通过 `CBAnalyst(backend=...)` 可注入自定义后端实现。
