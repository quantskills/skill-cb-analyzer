---
name: skill-cb-analyzer
description: A-share convertible bond daily comprehensive analyzer: double-low strategy,
  clause-driven events, stock linkage, risk monitoring, Black-Scholes options pricing,
  historical/implied volatility, IC + stratified backtesting. 21 signal detectors,
  4-dimension weighted scoring, LLM per-bond analysis, Markdown + JSON dual-format
  daily report.
version: 1.3.0
category: quant-skills
triggers:
  - 可转债
  - 转债
  - 双低策略
  - 强赎
  - 下修
  - 可转债分析
  - convertible bond
  - cb analysis
  - cb-analyzer
  - mcp
  - 可转债日报
  - 波动率
  - 期权定价
  - 回测
data_sources:
  - akshare bond_cb_jsl (集思录 — CB quotes, clause data, premiums)
  - akshare bond_cb_stock_map (CB ↔ stock mapping)
  - akshare bond_zh_cov_info_ths (同花顺 — maturity dates, issue scale, coupon rates)
  - akshare bond_zh_hs_cov_spot (同花顺 — volume, amount)
  - pandadata get_stock_daily (underlying stock K-line, configurable lookback)
  - pandadata get_stock_detail (stock info: industry, list status)
  - scipy.stats (Black-Scholes norm.cdf/pdf, Spearman rank correlation)
  - deepseek/claude API (LLM per-bond analysis, pluggable backend)
mcp_tools:
  - run_cb_analyzer: Full daily CB analysis with optional LLM per-bond analysis
  - get_latest_report: Read full content/metadata of the most recent CB report
  - check_trading_day: Verify if a date is an A-share trading day
  - search_bonds: Search ranked bonds by name/code in latest report
output_formats:
  - markdown日报 (11节)
  - json结构化数据
schedule: 每日收盘后
---

# 可转债每日分析 (skill-cb-analyzer)

A 股可转债每日综合分析：双低策略 + 条款驱动 + 正股联动 + 波动率期权 + 风险监控。通过 **21 个信号检测器**（估值4 + 条款4 + 联动4 + 波动率4 + 结构风险5）和 **四维加权评分**，配合 **Black-Scholes 期权定价**（Delta/Gamma/Vega、HV/IV）、**IC + 分层回测** 以及 **LLM 逐券分析**（DeepSeek/Claude，后端可插拔），生成 Markdown 日报 + JSON 结构化数据。

## 快速开始

```bash
pip install -r requirements.txt
pip install -e .                    # 可选：安装 cb-mcp / cb-analyzer 命令
cp .env.example .env  # 编辑填入凭证

python run.py                       # 最新交易日（默认启用LLM分析）
python run.py --date 20260701       # 指定日期
python run.py --no-llm              # 跳过LLM，仅规则引擎
python run.py --top-n 30 --verbose  # 自定义参数
python run.py --output-dir output   # 自定义输出目录
python run.py --cleanup-cache 30    # 清理30天前的缓存
python run.py --backtest            # 运行回测分析（IC + 分层收益）
```

## MCP 服务

本 Skill 支持 MCP (Model Context Protocol)，可被 Claude Desktop 或其他 LLM 客户端直接调用。

### 启动

```bash
python mcp_server.py
# 或安装后:
pip install -e .
cb-mcp
```

### 可用工具

| 工具 | 说明 | 关键参数 |
|------|------|----------|
| `run_cb_analyzer` | 运行完整可转债日分析（默认启用LLM，含波动率/BS信号） | `date`, `top_n`, `use_llm`(默认true), `llm_top_n` |
| `get_latest_report` | 获取最新报告（含回测章节） | `full_content` (true=全文, false=预览) |
| `check_trading_day` | 检查 A 股交易日 | `date` (YYYYMMDD) |
| `search_bonds` | 在最新报告中搜索债券 | `query` (名称/代码), `top_n` |

### Claude Desktop 配置

将以下内容添加到 Claude Desktop 的 `claude_desktop_config.json`（参考 `mcp_config.example.json`）：

```json
{
  "mcpServers": {
    "cb-analyzer": {
      "command": "python",
      "args": ["D:/python/PandaAI/Pandaai_skill_03/mcp_server.py"],
      "env": {
        "ANTHROPIC_AUTH_TOKEN": "your-api-key",
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
        "ANTHROPIC_MODEL": "claude-sonnet-4-20250514"
      }
    }
  }
}
```

配置后重启 Claude Desktop，即可通过自然语言调用可转债分析功能。

## 信号检测器 (21个)

### 估值信号 (A组，权重 40%)

| 检测器 | 权重 | 触发条件 | 说明 |
|--------|:----:|----------|------|
| 双低信号 | 4 | 价格 < 120 且 溢价率 < 20% | 经典双低策略 |
| YTM防御 | 3 | YTM > 国债 + 2% | 债底保护强，国债收益率可配置 |
| 纯债溢价率 | 3 | 转债价/纯债价值 < 1.05 | 接近债底，下行有限 |
| 溢价率分位 | 2 | 当前溢价率处于历史低分位 | 相对低估，基于 HistoryStore 历史数据 |

### 条款事件 (B组，权重 30%)

| 检测器 | 权重 | 触发条件 | 说明 |
|--------|:----:|----------|------|
| 强赎进度 | 4 | 正股价/转股价 > 1.20(预警)/> 1.28(高危) | 含连续日计数追踪 |
| 下修概率 | 3 | 综合评分 > 0.4 | 转股价下修博弈（评级分层票息模型） |
| 回售进度 | 2 | 正股价 < 转股价 × 70% | 含连续日计数追踪 |
| 临近到期 | 1 | 剩余期限 < 1年 | 到期折价/溢价风险 |

### 正股联动 (C组，权重 20%)

| 检测器 | 权重 | 触发条件 | 说明 |
|--------|:----:|----------|------|
| 正股动量 | 3 | 20日涨幅 ± 均线排列 | 多头/空头排列双向信号 |
| 转债偏离 | 2 | 转债涨幅 − 正股涨幅 < −3% | 补涨信号 |
| Delta弹性 | 2 | BS Delta > 0.7（高股性）或 < 0.3（偏债性） | 真实 Black-Scholes Delta = N(d1)，HV 不可用时回落至 cv/cb_price |
| 正股形态 | 1 | MA金叉/死叉，放量突破/破位 | 技术面双向信号 |

### 波动率与期权 (C组扩展，含于联动维度)

| 检测器 | 权重 | 触发条件 | 说明 |
|--------|:----:|----------|------|
| IV分位 | 2 | 隐含波动率处于历史低/高分位 | IV 低分位=期权便宜(看涨)，IV 高分位=偏贵(看跌) |
| 波动率背离 | 2 | IV − HV > 10% 或 HV − IV > 10% | IV >> HV 期权偏贵(看跌)，HV >> IV 期权便宜(看涨) |
| 波动率扩张 | 2 | 当前 HV 较 20 日前 > +20% 或 < −20% | 波动率扩张=更多机会(看涨)，收缩=机会减少(看跌) |
| Delta质量 | 2 | BS Delta > 0.70（高股性）或 < 0.30（偏债性） | BS Delta 始终 ∈ (0,1)，含 Gamma 辅助信息 |

### 市场结构与风险 (D组，权重 10%)

| 检测器 | 权重 | 触发条件 | 说明 |
|--------|:----:|----------|------|
| 成交量活跃 | 2 | 日成交额 > 5000万 | 流动性充足（amount 在数据获取层归一化为万元） |
| 余额趋势 | 1 | 余额减少 > 5% | 转股推进（基于 HistoryStore 历史对比） |
| 信用风险 | — | ST正股(含 *ST/PT/退市)/评级 < A/低价 | 惩罚项(−20)；检测 list_status 字段 |
| 强赎公告排除 | — | 已公告强赎 | 直接排除 |
| 流动性风险 | — | 成交额 < 100万 | 惩罚项(−10) |

## 期权定价模型

本 Skill 引入 Black-Scholes 欧式看涨期权定价模型（`core/options_pricing.py`），使用 `scipy.stats.norm.cdf/pdf`：

| 函数 | 公式 | 用途 |
|------|------|------|
| `bs_call_price(S, K, T, r, σ)` | S·N(d₁) − K·e^(−rT)·N(d₂) | 期权理论价格 |
| `bs_delta(S, K, T, r, σ)` | N(d₁) | 替换旧 cv/cb_price 近似，始终 ∈ (0,1) |
| `bs_gamma(S, K, T, r, σ)` | N'(d₁) / (S·σ·√T) | Delta 变化率 |
| `bs_vega(S, K, T, r, σ)` | S·N'(d₁)·√T / 100 | 每 1% 波动率变化的价格影响 |
| `historical_volatility(close, window)` | std(log_returns) × √252 | 历史波动率（年化），窗口可配置 |
| `implied_volatility(price, S, K, T, r)` | 二分法求解 | 反推隐含波动率 |

**关键假设：** 可转债期权价值 ≈ 转债价格 − 纯债价值。IV 通过将 BS 价格与期权价值匹配得到。

## 回测框架

通过 `core/backtester.py` 验证评分模型的历史有效性（`--backtest` 标志）：

| 指标 | 说明 |
|------|------|
| Rank IC | Spearman 秩相关系数（每日评分 vs N日远期收益） |
| IC IR | 平均 IC / IC 标准差 |
| IC 胜率 | IC > 0 的交易日占比 |
| 分层回测 | 按综合评分分 5 组，等权计算远期收益，累计收益曲线 |

回测基于 `output/` 目录中的历史 JSON 报告 + `cache/` 中的历史价格数据。数据不足时自动降级显示提示信息。随着每日运行自动积累历史。

## 评分模型

```
综合分 = 0.40×估值 + 0.30×条款 + 0.20×联动(含波动率) + 0.10×结构
最终分 = clamp(综合分 × 100 + 风险惩罚, 0, 100)
```

每个维度下限为 0.30（可配置 `scoring.dimension_floor`），信号缺失视为中性而非零分。

等级阈值（可配置 `scoring.grades`）：

| 分数 | 等级 | 含义 |
|------|------|------|
| 55-100 | A+ | 强烈关注 |
| 45-54 | A | 值得跟踪 |
| 40-44 | B+ | 偏积极 |
| 37-39 | B | 温和 |
| 34-36 | C | 中性 |
| 30-33 | D | 偏弱 |
| 0-29 | E | 回避 |

## 报告结构 (11节)

1. 市场概览 — 转债总数、均价、平均溢价率、平均YTM
2. 双低策略精选 — Top 15 双低转债（价格<120 且 溢价率<20%）
3. 高YTM防御组合 — YTM > 3% 债性转债
4. 条款事件监控 — 强赎预警(含连续日)/下修候选/回售触发(含连续日)/到期
5. 正股联动精选 — 折价套利机会，含 IV/HV 列
6. 行业分布 — 行业集中度
7. 信用风险告警 — ST/评级下调/面值退市风险
8. 综合评分排名 — Top N 详细评分（含 BS Delta、波动率信号）
9. AI逐券研判 — LLM深度分析（后端可插拔，提示词可配置）
10. 数据溯源 — 每类数据来源
11. 策略回测 — IC分析 + 分层回测（`--backtest` 启用）

## 数据流架构

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│  data_fetcher │───▶│ bond_calculator │───▶│ 21 detectors  │
│  (AKShare +   │    │  (metrics)     │    │  (A/B/C/D/Vol)│
│   Pandadata)  │    └──────────────┘    └───────┬───────┘
└──────┬───────┘                                  │
       │                                          ▼
       │                                  ┌───────────────┐
       ├─────────────────────────────────▶│ HistoryStore   │◀── 历史数据
       │                                  │ (parquet)     │    (A4/B1/B3/D2
       │                                  └───────┬───────┘     + scores)
       │                                          │
       │                           ┌──────────────┤
       │                           ▼              ▼
       │                  ┌──────────────┐ ┌──────────────┐
       │                  │ options_pricing│ │  backtester  │
       │                  │ (BS/HV/IV)    │ │ (IC + 分层)  │
       │                  └───────┬──────┘ └──────┬───────┘
       │                          │               │
       ▼                          ▼               ▼
┌──────────┐    ┌──────────┐    ┌─────────┴─────┐
│  scorer  │◀───│ pipeline │───▶│   reporter    │
│ (4-dim)  │    │ (orchestrator) │ (MD + JSON)  │
└──────────┘    └──────┬───────┘ └───────────────┘
                       │
                       ▼
                ┌──────────┐
                │ CBAnalyst │  (--llm)
                │ (pluggable backend)
                └──────────┘
```

## 数据源

| 数据类别 | 主源 | 说明 |
|----------|------|------|
| 可转债行情 | AKShare bond_cb_jsl (集思录) | 价格/溢价率/转股价/强赎触发价/回售触发价 |
| 转债期限+规模+票息 | AKShare bond_zh_cov_info_ths (同花顺) | 到期日、发行规模、票面利率（如API提供） |
| 转债成交量 | AKShare bond_zh_hs_cov_spot (同花顺) | amount 元→万元归一化在 data_fetcher 层完成 |
| 正股K线 | Pandadata get_stock_daily | ~500只正股，可通过 config 配置 lookback 天数 |
| 正股信息 | Pandadata get_stock_detail | 行业分类、list_status（ST检测） |
| 期权定价 | scipy.stats | Black-Scholes norm.cdf/pdf，Spearman 秩相关 |
| LLM分析 | DeepSeek / Claude API | 后端可插拔（LLMBackend Protocol），provider 可配置 |
| 交易所映射 | exchange_utils | 600/601/688/689→SH, 000/002/300→SZ, 8xx→BJ |

## 配置参考 (config.json)

```json
{
  "scan": { "lookback_days": 120 },
  "valuation": { "treasury_1y": 1.5, "treasury_3y": 2.2, "double_low_price_max": 120, ... },
  "clause": { "redemption_consecutive_days": 15, "redemption_total_days": 30, ... },
  "options": {
    "risk_free_rate": 0.025,
    "hv_window": 60,
    "iv_low_percentile": 25,
    "iv_high_percentile": 75,
    "hv_iv_divergence_threshold": 0.10,
    "vol_expansion_lookback": 20,
    "bs_delta_high": 0.70,
    "bs_delta_low": 0.30
  },
  "risk": { "min_daily_turnover": 100, "credit_exclude_st": true, ... },
  "scoring": {
    "valuation_weight": 0.40, "clause_weight": 0.30,
    "linkage_weight": 0.20, "structure_weight": 0.10,
    "dimension_floor": 0.30,
    "grades": [
      {"threshold": 55, "grade": "A+", "label": "强烈关注"},
      {"threshold": 45, "grade": "A", "label": "值得跟踪"}, ...
    ]
  },
  "backtest": {
    "forward_days": 5,
    "n_quintiles": 5,
    "min_periods": 2
  },
  "llm": {
    "provider": "anthropic",
    "model": "deepseek-v4-pro",
    "top_n": 5,
    "prompt_template": "自定义提示词模板...",
    "max_tokens": 2048, "timeout": 120.0, "max_retries": 2, "llm_retries": 3
  },
  "output": { "dir": "output" }
}
```

## 核心约束

| 约束 | 说明 |
|------|------|
| 强赎已公告排除 | 已发强赎公告的转债直接剔除 |
| 信用风险惩罚 | ST/−15，评级过低/−20，流动性不足/−10；检测 list_status 字段 |
| 交易所后缀映射 | 600/601/603/605/688/689→SH, 000/001/002/003/300/301→SZ, 4xx/8xx→BJ |
| BS Delta 替换 | Delta弹性使用真实 N(d1)，HV 不可用时回落 cv/cb_price |
| 票息优先实际值 | API 提供票面利率时优先使用，否则回退评级分层估算 |
| 不荐券 | 措辞为「值得关注」「可跟踪」，禁止「买入」「目标价」 |
| 交易日智能跳过 | 节假日不跑空；is_trading_day() API故障默认False |
| 数据容错 | 单只转债数据缺失跳过，不阻塞全流程 |
| 评分可审计 | JSON输出包含评分子项 |
| 列名规范化 | 中文列名在缓存前转为英文（`CB_COLUMN_MAP` 在 `_types.py` 中共享） |
| 双重 pipeline 消除 | --llm 模式下 pipeline 仅运行一次，LLM 分析后轻量 regenerate_report() |
| 后端可插拔 | LLMBackend Protocol，支持 AnthropicBackend 及未来 MCP 后端 |
| NaN 安全 | `safe_float()` 替代 `float() or sentinel`，NaN 输入正确归位哨兵值 |
| 空头信号一致 | 四个 `composite_score` 统一保留负向 strength，clamp 到 [0, 1] |
| API 容错 | `init_api()` / `get_last_trade_date()` 异常捕获，pipeline 优雅降级 |
| HistoryStore 缓存 | session 级缓存避免每只转债重复读取 parquet 文件 |
| HistoryStore 完整性 | SHA-256 校验 + .bak 自动备份 + 原子写入（临时文件重命名） |
| 回测数据不足降级 | 数据不足时显示提示信息，不阻塞报告生成 |

## 目录结构

```
skill-cb-analyzer/
├── SKILL.md                    # Agent 工作流入口
├── config.json                 # 运行配置
├── pyproject.toml              # Python 项目
├── requirements.txt            # 依赖
├── .env.example                # 凭证模板
├── run.py                      # CLI 入口
├── mcp_server.py               # MCP 服务 (4 tools)
├── mcp_config.example.json     # Claude Desktop MCP 配置示例
├── core/
│   ├── _types.py               # 共享类型、CB_COLUMN_MAP、safe_float()
│   ├── bond_calculator.py      # 可转债计算引擎 (含 rating-based coupon schedule)
│   ├── data_fetcher.py         # 数据获取 (含列名规范化 CB_COLUMN_MAP、票息保留)
│   ├── exchange_utils.py       # A股交易所后缀映射 (600→SH, 688→SH, 300→SZ, 8xx→BJ)
│   ├── cache.py                # 日期分区 Parquet 缓存 (含 is_stale 过期检测)
│   ├── history_store.py        # 逐券时序存储 (A4分位/B1B3连续日/D2余额趋势 + 评分列)
│   ├── valuation.py            # A组：估值信号 (4)
│   ├── clause_monitor.py       # B组：条款事件 (4，含连续日追踪)
│   ├── stock_linkage.py        # C组：正股联动 (4，BS Delta 替代简化和近似)
│   ├── options_pricing.py      # C组扩展：波动率/期权 (4) + Black-Scholes/HV/IV
│   ├── risk_filter.py          # D组：结构+风险 (5，含 list_status ST检测)
│   ├── scorer.py               # 四维加权评分 (可配置等级+维度下限)
│   ├── pipeline.py             # 端到端流水线 (含 regenerate_report、run_backtest)
│   ├── backtester.py           # 回测框架 (IC分析 + 分层回测 + 前向收益)
│   └── reporter.py             # Markdown + JSON 报告 (11节)
├── llm/
│   └── analyst.py              # LLM分析 (LLMBackend Protocol + AnthropicBackend + 规则备选)
├── tests/                      # 测试用例 (161个)
├── data/                       # HistoryStore 持久化数据
├── references/                 # 参考文档
├── cache/                      # 数据缓存
└── output/                     # 报告输出
```

## 免责声明

本 Skill 输出仅供研究参考，**不构成任何投资建议**。可转债交易存在市场风险、信用风险和条款风险，投资者应独立判断并承担交易风险。过往表现不代表未来收益。

## License

GPLv3
