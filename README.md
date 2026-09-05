# QMT Quant Research

面向沪深 A 股的 **PIT（point-in-time）量化研究、严格样本外验证、数据审计与受控 MiniQMT 执行工程**。

本仓库当前不是“V3→V7 已投产”的线性版本链。权威主线是 **V5-C pre-2026 stock-selection research**：2017-2025 数据已经冻结并建立不可变 lineage；2026 是隔离 holdout，只有 pre-2026 Basic Alpha Gate 通过且存在精确冻结 candidate SHA256 后才允许一次性评估。

## 当前状态

- **2017-2025 历史数据**：冻结，不重新抓取成功 shard。
- **数据拓扑**：20 个确定性 shard，`max-parallel: 5`。
- **BaoStock**：固定 `0.9.3`；历史恢复链保留重连、自愈和 bounded socket timeout rebinding。
- **严格参考约束**：PIT universe、ST、停牌、涨跌停、raw-reference 均 fail-closed。
- **覆盖率门禁**：symbol `>= 0.98`，session `>= 0.97`。
- **V5-C**：stock-selection-only，nested/purged，`risk_on` 固定 always-on；因子/variant/weight 选择禁止使用 2026。
- **Basic Alpha Gate**：当前 C1 与 C7 均未通过，因此 **2026 绩效仍保持盲化**。
- **实盘**：默认 dry-run；V5 研究 candidate 当前没有生产 scoring adapter，系统会 fail-closed，禁止用旧 V3 scorer 冒充 V5。

权威历史 lineage 位于：

```text
research_lineage/v5_c_pre2026.json
```

永久冻结副本发布于 GitHub Release：

```text
v5-c-pre2026-frozen-lineage-v1
```

该 Release 只包含 frozen 2017-2025 bars/reference、recovered shard13、20 个 authoritative PIT exposure shard、正确 pinned 的 industry artifact 以及 pre-2026 nested research 报告；不包含 2026 holdout。

## 数据来源与复现

当前 V5 历史主线以 **冻结的 BaoStock 0.9.3 artifacts** 为核心，而不是每次运行重新下载市场数据。仓库仍保留 Tushare/AkShare 等旧工具或参考数据路径，但它们不应被描述为 V5-C 权威 frozen lineage 的市场行情主源。

研究 CI 应优先消费 frozen artifacts / immutable Release，而不是重建已经成功的 2017-2025 数据。这样可以避免 BaoStock 服务波动、复权基线变化和 GitHub Actions artifact 到期导致结论不可复现。

## V5-C 研究原则

V5-C 的研究边界：

1. 使用严格 PIT 股票池，不用今天的股票池回看历史。
2. ST、停牌、涨跌停、raw limit reference 不得降级。
3. signal 与 execution 保持因果时序，nested inner/outer evidence 做 purge。
4. stock selection 层研究时 `risk_on` 始终为 true，禁止把 timing 收益混入 Alpha Gate。
5. 2026 不参与 factor、neutralization variant、weight、portfolio 参数选择。
6. Basic Alpha Gate 不通过时，只能继续 pre-2026 Alpha research，不能查看 2026 策略收益。
7. Gate 通过后也只能用精确 candidate SHA 和精确 run lineage 创建隔离的一次性 no-refit holdout evaluator。

### 当前 C1 core factors

```text
liquidity_stability
low_volatility
low_downside_risk
short_reversal
```

已明确排除的旧 momentum core 不应被静默重新引入。

## 数据审计

```bat
python run_data_audit.py ^
  --start 20170101 ^
  --end 20251231 ^
  --reference-dir data\reference ^
  --bar-cache-dir data\qmt_bars
```

当前审计不仅检查覆盖率，还检查：

- adjusted OHLC 为正且 high/low 包络关系合理；
- volume/amount 非负；
- raw `open/close/preClose` 合法；
- `down_limit < up_limit` 且 `down_limit <= pre_close <= up_limit`；
- 已知整日停牌从应交易 session 分母中排除；
- legacy `__NONE__` 只作为“provider 成功返回空结果”的 provenance，不作为真实股票代码。

输出包含：

```text
symbol_session_coverage.csv
market_data_quality.csv
data_audit.json
```

## 回测执行模型

默认 signal 到 execution 至少延迟一个交易 session。当前引擎显式执行：

- A 股 T+1 卖出约束；
- 100 股整数手；
- 佣金与卖出印花税；
- strict PIT ST / suspension / limit guards；
- raw unadjusted price-limit reference；
- deterministic probabilistic-fill simulation。

日线数据不能重建盘中“先触板还是先成交”的路径，因此当前结果会明确报告：

```text
intraday_limit_touch_modelled = false
```

不能把日线开盘/一字板模型表述为完整的 intraday limit-touch 仿真。

## 实盘目标生成：显式来源、默认拒绝

`generate_live_targets.py` 不再默认读取 `output/v3_research/best_config.json`，也不会在配置缺失时静默回退到 `StrategyConfig()`。

旧 StrategyConfig 研究如果确实要生成 dry-run 目标，必须显式给出配置：

```bat
python generate_live_targets.py ^
  --strategy-config path\to\strategy.json ^
  --as-of YYYYMMDD ^
  --reference-dir data\reference
```

V5 candidate 必须经过 schema、Basic Alpha Gate、holdout unlock/pass 和 exact SHA256 校验；但在 V5 production scorer 尚未实现前，传入 V5 candidate 会 **fail-closed**，不会偷偷调用 V3 scorer。

目标文件带有：

```text
signal_date
strategy_source
strategy_sha256
code
target_weight
```

## MiniQMT 执行安全

`run_qmt_executor.py` 默认 dry-run。即使显式打开 live，也必须满足：

- `--enable-live`；
- `--confirm-live LIVE`；
- target `signal_date` 等于当前 Asia/Shanghai 市场日期；
- target 带合法 strategy SHA256；
- acceptance report 绑定同一个 strategy SHA256；
- pre-trade risk gate 通过。

执行器还会：

- MiniQMT 连接失败后用新 trader instance 有界重试；
- 每个订单在提交前后写 durable JSONL journal 并 `fsync`；
- `order_stock` 抛异常时立即停止后续订单，因为远端副作用可能未知；
- 查询已提交订单并记录 partial fill / remaining volume；
- 缺失订单、部分成交或不确定提交只标记人工 reconciliation，不自动撤单。

本仓库 CI 不连接真实账户，不发送真实订单。

## 验收必须绑定同一策略

策略验收不得再依赖隐式 V3 路径。验收输入必须显式指定，并提供精确策略 SHA256；生成的 `acceptance_report.json` 会记录证据路径与该 SHA，供执行器核对。

研究 Gate、2026 holdout Gate 与 live acceptance 是不同层次的契约，不能互相替代。

## 自动化测试与依赖

依赖使用单一 exact-pin `requirements.txt`，Python 3.10 / 3.11 / 3.12 CI 都安装同一份文件并运行 `pip check`。BaoStock 固定：

```text
baostock==0.9.3
```

离线测试：

```bat
run_automated_tests.bat
```

CI 测试数量会随仓库演进变化，因此 RELEASE_MANIFEST 不再硬编码旧的固定测试数量；以当前 GitHub Actions 结果为准。

## 目录

```text
qmt_quant/          核心数据、研究、回测、信号、执行适配
research_lineage/  冻结研究 run/artifact/SHA lineage
research/           研究层扩展
risk/               pre-trade 风控
execution/          执行层扩展
monitoring/         运行状态检查
config/             本地配置；不要提交 token/账号
 tests/              自动测试
output/             本地运行产物
```

## 不可破坏的安全边界

- 不 force-push 研究历史。
- 不提交或输出 Secrets、账号、token。
- 不在 CI 连接真实交易账户。
- 不重新获取已成功冻结的 2017-2025 bars。
- 不把 accidental/non-authoritative recovery artifacts 混入 nested research。
- 不降低 PIT/ST/suspension/limit/raw-reference fail-closed 约束。
- 不降低 0.98 symbol / 0.97 session coverage thresholds。
- 不因为研究目标高而删掉交易成本、T+1、停牌或限价约束。
- pre-2026 Alpha Gate 未通过时，2026 策略绩效必须保持不可见。
