# 数据源路由文档

> **访问方式：** 除 CLI (`python run.py`) 外，本 Skill 还支持 MCP 协议调用。
> 启动 `python mcp_server.py` 后，可通过 `run_cb_analyzer`、`get_latest_report`、
> `check_trading_day`、`search_bonds` 四个工具远程调用分析功能。
> 详见 `SKILL.md` MCP 服务章节和 `mcp_config.example.json`。

## 数据获取优先级

### 可转债行情
```
AKShare bond_cb_jsl (集思录) ──主源──▶ 价格/溢价率/转股价/到期日/评级
AKShare bond_zh_cov_info_ths (同花顺) ──补充──▶ 到期日、发行规模
AKShare bond_zh_hs_cov_spot (同花顺) ──补充──▶ 成交量、成交额
AKShare bond_cb_stock_map ──映射──▶ 转债↔正股代码对应
```

### 正股K线
```
Pandadata get_stock_daily (主) ──fail──▶ 跳过（无法继续）
```

### 正股信息
```
Pandadata get_stock_detail (主) ──fail──▶ 使用空信息（降级继续）
```

## AKShare 接口说明

### bond_cb_jsl (集思录)

集思录可转债实时行情，包含：
- 转债代码、转债名称、转债最新价
- 转股价、转股价值、转股溢价率
- 正股代码、正股名称、正股价格
- 债券评级、到期日、剩余年限
- 回售触发价、强赎触发价、到期赎回价
- 成交额

### bond_zh_cov_info_ths (同花顺)

同花顺可转债基本信息，补充字段：
- 到期日（maturity_date）—— 部分集思录缺失的到期日由此补全
- 发行规模（issue_scale）

### bond_zh_hs_cov_spot (同花顺)

同花顺可转债现货行情，补充字段：
- 成交量（volume）
- 成交额（amount）

**重要：amount 字段单位处理**
- 成交量/额（amount）单位始终为**元**
- `data_fetcher.py` 中 `_fetch_cb_turnover_data()` 在返回前除以 10000，统一转为**万元**
- 下游所有模块（valuation、risk_filter、reporter）接收的 turnover 均为万元
- **成交量活跃信号 (D1)**：硬编码阈值 5000 万元（`detect_volume()` 中 `turnover >= 5000`）
- **流动性风险信号 (D5)**：可配置阈值 `risk.min_daily_turnover`（默认 100 万元）

### bond_cb_stock_map

可转债与正股代码映射表。

## Pandadata 接口说明

### get_stock_daily

获取正股日K线数据：
- 分块拉取（200只/块）
- 线程池并发（最大4线程）
- 含重试+指数退避（最多3次）
- K线回看天数通过 `config["scan"]["lookback_days"]` 配置（默认 120 天）

### get_stock_detail

获取正股基本信息：
- 名称、行业分类
- **list_status**：上市状态（用于 ST/*ST/PT/退市 检测）

## 数据缓存

### 行情缓存 (cache/)

```
cache/
  YYYYMMDD/
    cb_quote.parquet    # 转债行情（列名已英文化）
    stock_kline.parquet # 正股K线
    stock_info.parquet  # 正股信息
    .cache_meta.json    # 缓存元信息
```

### 历史时序 (data/)

```
data/
  cb_history.parquet    # 逐券时序数据
                        # 列: trade_date, bond_code, premium_rate,
                        #     outstanding_balance, redemption_ratio, putback_ratio
```

HistoryStore 为以下检测器提供历史数据：
- **A4 溢价率分位**：`get_premium_history(bond_code)` → 计算当前分位
- **B1 强赎连续日**：`get_consecutive_days(field="redemption_ratio", direction="above")`
- **B3 回售连续日**：`get_consecutive_days(field="putback_ratio", direction="below")`
- **D2 余额趋势**：`get_previous_balance(bond_code, trade_date)` → 计算环比变化

`load_history()` 具有 session 级缓存：同一 pipeline 运行中多次调用仅读取一次 parquet 文件，
`save_snapshot()` 写入后自动失效缓存。`get_premium_history()` 等方法支持可选 `df` 参数接收预加载的 DataFrame。

## 列名规范化

`core/_types.py` 中的 `CB_COLUMN_MAP`（原分散在 `data_fetcher.py` 和 `bond_calculator.py` 中的各自副本，v1.2 合并为单一共享定义）将 AKShare 返回的中文列名统一转换为英文，防止 Windows 环境下 parquet 写入中文列名导致的编码损坏。关键映射：

| 原始列名 | 规范化列名 |
|----------|-----------|
| 代码/转债代码/债券代码 | bond_code |
| 转债名称/债券简称 | bond_name |
| 转债最新价/现价/债现价 | cb_price |
| 转股溢价率 | premium_raw |
| 到期日/到期时间 | maturity_date |
| 债券评级/信用评级 | credit_rating |
| 成交额/amount | turnover |
| 发行规模 | issue_scale |

## 容灾策略

| 数据 | 主源 | 备源 | 降级行为 |
|------|------|------|----------|
| 转债行情 | AKShare 集思录 | bond_cb_daily | 报告标注数据缺失 |
| 到期日/规模 | 集思录 | 同花顺 bond_zh_cov_info_ths | 使用默认值 |
| 成交量/额 | 集思录 | 同花顺 bond_zh_hs_cov_spot | 成交量/额显示为 0 |
| 正股K线 | Pandadata | — | 跳过正股联动信号 |
| 正股信息 | Pandadata | — | 行业显示"未知"，跳过ST检测 |
| LLM分析 | AnthropicBackend | 规则引擎备选 | 输出基于规则的分析 |
| 历史数据 | cb_history.parquet | — | 首次运行时历史为空，分位/连续日不可用 |
