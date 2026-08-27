# 因子风格状态监测框架 Factor Regime Monitor

基于因子 ETF 构建的市场风格状态监测框架，用滚动收益率、波动率与 Z-score 方法量化跟踪市场风格轮动（价值/成长/动量/低波动/大盘/小盘），并对照美国宏观经济数据做交叉验证。

## 项目背景

投研工作中常需要判断"当前市场处于什么风格阶段"（比如价值股占优还是成长股占优、防御性资产是否走强等）。本项目尝试用可复现的量化方法，把这种判断从"凭感觉"变成"有数据支撑的结构化结论"。

## 方法概述

1. **数据获取**：通过 yfinance 下载 6 只代表性因子 ETF（SPY 大盘、IVE 价值、IVW 成长、MTUM 动量、USMV 低波动、IWM 小盘）2015 年至今的历史价格
2. **指标计算**：计算 63 个交易日（约 3 个月）滚动年化收益率与波动率
3. **状态判断**：对每个因子的滚动收益率，相对其过去 252 个交易日（约 1 年）的历史均值和标准差计算 Z-score，据此将其状态分类为「扩张」「中性」「收缩」三种
4. **可视化输出**：生成状态热力图，并自动生成中文文字简报
5. **宏观交叉验证**：叠加美国 CPI 同比、工业生产指数等宏观数据，做滚动相关性分析（描述性观察，非因果推断）

## 文件说明

| 文件 | 作用 |
|---|---|
| `download_factor_etf_prices.py` | 下载并缓存因子 ETF 历史价格 |
| `calculate_factor_etf_metrics.py` | 计算滚动年化收益率与波动率 |
| `calculate_regime_zscore.py` | 计算 Z-score 并分类市场状态 |
| `generate_regime_report.py` | 生成状态热力图与文字简报 |
| `compare_regime_with_macro.py` | 叠加宏观数据做交叉验证（需配置 FRED API key） |

## 示例输出

**状态热力图**（近 252 个交易日各因子状态变化）

![状态热力图](regime_heatmap.png)

**与宏观数据交叉验证**

![宏观对照](regime_vs_macro.png)

## 如何运行

```bash
pip install pandas yfinance matplotlib fredapi

python download_factor_etf_prices.py --refresh
python calculate_factor_etf_metrics.py
python calculate_regime_zscore.py
python generate_regime_report.py

# 可选：宏观数据对照，需先设置环境变量 FRED_API_KEY
python compare_regime_with_macro.py
```

## 说明与局限

- 因子 ETF 仅作为对应投资风格的代理（proxy），而非严格的学术因子定义
- Z-score 阈值（±1）可通过命令行参数调整
- 宏观相关性分析样本有限（12 个月滚动窗口），仅作描述性观察，不构成预测信号或投资建议

## 后续计划

正在扩展为 A 股版本，使用 akshare 等本土数据源，替换美股 ETF 与 FRED 宏观数据。
