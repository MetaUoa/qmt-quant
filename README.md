# QMT Quant Research Suite V3-V7

面向 **QMT / MiniQMT + Tushare** 的 2018-2025 沪深 A 股量化研究、样本外验证、压力测试和受控实盘执行工程。

目标可以是“8 年 150 倍”，但工程不会把 150 倍硬编码成结果。只有真实 QMT 历史数据跑出的结果同时通过 **Point-in-time、Walk-forward、压力测试和风险门槛**，才会被判定为合格。

## 当前完成范围

### V2.2 数据验收

- 历史沪深 A 股 Point-in-time 股票池。
- 上市/退市日期过滤。
- 历史 ST 快照。
- 停牌、涨跌停约束。
- 前复权信号 / 不复权限价判断分离。
- QMT 日线和 raw limit bars 覆盖率审计。
- 新增逐股票“预期交易日 vs 实际行情行数”覆盖率检查。

入口：

```bat
python run_data_audit.py --start 20180101 --end 20251231 --reference-dir data\reference --download
```

输出：`output/v2_2_data_audit/`

### V2.5 全市场 baseline

严格使用 T 日收盘信号，默认 T+1 开盘执行：

```bat
run_baseline.bat
```

核心输出：总倍数、CAGR、最大回撤、Sharpe、Calmar、逐年收益、交易记录和数据质量报告。

### V3 多目标研究

策略仍以趋势/动量为核心，但支持：

- 20/60/120 日动量权重搜索；
- 波动率惩罚；
- 流动性过滤；
- 中证500趋势过滤；
- 全市场宽度过滤；
- Top N / 调仓周期 / 动量阈值搜索；
- 年度稳定性惩罚；
- 单股票成交额集中度惩罚；
- 参数邻域稳定性惩罚。

开发期默认 `2018-2022`，`2023-2025` 只作为 holdout 审计，不参与参数选择。

```bat
run_parameter_research_quick.bat
```

更大的参数网格：

```bat
run_parameter_research_balanced.bat
```

输出：`output/v3_research/`

其中：

- `candidate_summary.csv`
- `best_config.json`
- `selection_audit.json`
- `best_full_result/`

### V4 Walk-forward

默认：

```text
2018-2020 -> 2021
2019-2021 -> 2022
2020-2022 -> 2023
2021-2023 -> 2024
2022-2024 -> 2025
```

参数只能用该验证年度之前的数据选择。

```bat
run_walk_forward.bat
```

输出：`output/walk_forward/`

### V4.5 压力测试

实际重新运行执行路径，而不是只对结果做数学缩放：

- 佣金 x2；
- 滑点 x2 / x3；
- T+2 执行；
- 95% / 90% 成交率模拟；
- 动量参数 ±20%；
- 波动率门槛 ±10%；
- 随机删除 10% 股票池，3 个 seed；
- 每日收益 bootstrap Monte Carlo。

```bat
run_stress_tests.bat
```

输出：`output/v4_5_stress/`

### V5 150x 自动验收

默认 A 级要求：

```text
2018-2025 multiple >= 150x
Max Drawdown <= 35%
Sharpe >= 1.50
OOS CAGR >= 30%
5 个 OOS 年份至少 4 个盈利
压力场景通过率 >= 75%
```

B 级偏向“50x+ 但更稳健”；C 级表示收益为正、样本外和压力测试具备基本可复制性。

```bat
run_acceptance.bat
```

输出：`output/v5_acceptance/acceptance_report.json`

**没有通过 A 级，就不能声称“150 倍目标完成”。**

### V6 模拟/纸面执行

先刷新到当前日期的 reference data，然后生成目标仓位：

```bat
python prepare_reference_data.py --start 20180101 --end 20260902 --output data\reference
python generate_live_targets.py --as-of 20260902 --download --reference-dir data\reference
```

或：

```bat
generate_live_targets.bat
```

输出：

```text
output/live_targets/target_weights.csv
output/live_targets/signal_diagnostics.json
```

### V7 MiniQMT 执行

默认 **dry-run**，连接 MiniQMT 后只查询资产/持仓/实时 tick 并生成订单计划，不发送订单：

```bat
python run_qmt_executor.py ^
  --userdata "D:\QMT\userdata_mini" ^
  --account YOUR_ACCOUNT
```

只有同时满足以下条件才允许发送订单：

1. 明确加 `--enable-live`；
2. 再加 `--confirm-live LIVE`；
3. 默认要求 V5 验收等级至少 C；
4. 实盘前风控必须 PASS。

示例：

```bat
python run_qmt_executor.py ^
  --userdata "D:\QMT\userdata_mini" ^
  --account YOUR_ACCOUNT ^
  --enable-live ^
  --confirm-live LIVE
```

执行模块使用 MiniQMT `XtQuantTrader` 的连接、资产、持仓和同步 `order_stock` 接口；订单以 100 股整数手处理。买单在真正发送前会再次查询当前可用现金并缩量，避免现金透支。

## 一键完整研究流程

首先设置 Tushare Token：

```bat
set TUSHARE_TOKEN=你的Token
```

然后：

```bat
run_full_research_pipeline.bat
```

流程：

```text
prepare reference
  -> V2.2 data audit
  -> V2.5 strict baseline
  -> baseline artifact validation
  -> V3 parameter research
  -> V4 walk-forward
  -> V4.5 stress test + Monte Carlo
  -> V5 acceptance grade
```

更大参数网格：

```bat
run_full_research_pipeline_balanced.bat
```

如果目标就是严格验收 **150x A 级**，直接运行：

```bat
run_full_research_pipeline_150x.bat
```

该脚本只有 V5 最终等级达到 A 才返回成功。

注意：这条完整链路需要在你本机能 `import xtquant` 的 QMT Python 环境执行。当前交付包内没有伪造 2018-2025 的真实收益数字。

## 自动化测试

离线自动测试：

```bat
run_automated_tests.bat
```

本机 QMT 真实 smoke：

```bat
run_automated_tests_full.bat
```

覆盖内容包括：

- Python compileall；
- T -> T+1 因果时序；
- 修改未来行情不改变过去结果；
- PIT 成员/ST/涨跌停；
- 停牌和 100 股整数手；
- 现金不透支；
- 印花税切换；
- QMT 缓存；
- Walk-forward 选择；
- 多目标研究；
- 压力测试；
- Monte Carlo；
- 150x 分级；
- live target；
- pre-trade risk。

GitHub Actions 会在 Python 3.10/3.11/3.12 跑离线测试。

## 目录

```text
qmt_quant/       核心回测、数据、研究、信号、压力测试、执行适配
research/        研究层扩展目录
risk/            实盘前风险闸门
execution/       执行层扩展目录
monitoring/      运行状态检查
config/          本地配置目录（不要提交 Token/账号）
tests/           自动测试
output/          运行产物
```

## 数据准备

```bat
pip install -r requirements.txt
set TUSHARE_TOKEN=你的Token
prepare_reference_data.bat
```

`prepare_reference_data.py` 默认会为调仓周期 `3/5/10` 和执行延迟 `1/2` 个交易日准备 ST/涨跌停日期并集，保证 Walk-forward 和 T+2 压力测试在 strict mode 下不因参考快照缺失而降级。

## 原则

- 不能用今天的股票池回测 2018 年。
- 不能用未来行情决定过去交易。
- 不能假定涨停能买、跌停能卖。
- 不能因为追求 150x 而取消滑点/手续费/停牌约束。
- holdout 和 OOS 结果不能反向参与参数选择。
- 回测高收益但 OOS 或压力测试失败，最终自动 `REJECT`。
- 实盘默认 dry-run；研究验收不通过，不应开启 live。
