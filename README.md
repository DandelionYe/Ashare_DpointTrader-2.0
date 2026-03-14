# A-Share Dpoint Trader 2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A 股单股票机器学习交易策略系统**

基于机器学习预测次日涨跌概率（Dpoint），结合 walk-forward 验证和随机搜索超参数优化的量化交易回测框架。

---

## 📋 目录

- [功能特性](#-功能特性)
- [系统要求](#-系统要求)
- [安装步骤](#-安装步骤)
- [快速开始](#-快速开始)
- [使用说明](#-使用说明)
- [配置参数](#-配置参数)
- [输出文件](#-输出文件)
- [项目结构](#-项目结构)
- [核心算法](#-核心算法)
- [常见问题](#-常见问题)
- [免责声明](#-免责声明)

---

## ✨ 功能特性

### 核心功能

- **Dpoint 预测模型**: 预测次日收盘价上涨的概率 `P(close_{t+1} > close_t | X_t)`
- **Walk-forward 验证**: 时间序列交叉验证，避免前向偏差，确保样本外评估
- **随机搜索优化**: 自动搜索最优特征组合、模型参数和交易策略参数
- **多模型支持**: Logistic Regression, SGDClassifier, XGBoost (支持 GPU 加速)
- **A 股交易约束**: 支持 T+1、最小 100 股单位、仅做多等 A 股规则

### 技术特性

- **特征工程**: 动量、波动率、成交量、换手率、K 线形态等 5 大家族 80+ 特征
- **早停剪枝**: 自动淘汰劣质配置，加速搜索过程
- **并行搜索**: 利用多核 CPU 并行评估候选配置
- **CUDA 加速**: 自动检测并启用 XGBoost GPU 加速
- **持续学习**: 支持 `continue` 模式，基于历史最优配置继续优化

---

## 💻 系统要求

### 必需环境

- **操作系统**: Windows / Linux / macOS
- **Python**: 3.8 或更高版本
- **内存**: 建议 8GB 以上

### Python 依赖

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
openpyxl>=3.0.0        # Excel 读写
xgboost>=1.5.0         # 可选，用于 XGBoost 模型
joblib>=1.1.0          # 并行计算
```

---

## 📥 安装步骤

### 1. 克隆或下载项目

```bash
git clone <your-repo-url>
cd "Ashare_DpointTrader 2.0"
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install pandas numpy scikit-learn openpyxl joblib

# 可选：安装 XGBoost（启用 GPU 加速）
pip install xgboost

# 如需 GPU 加速，额外安装 CUDA 版本的 XGBoost
pip install xgboost-cu112  # 根据 CUDA 版本选择
```

### 4. 验证安装

```bash
python -c "import pandas, sklearn, xgboost; print('OK')"
```

---

## 🚀 快速开始

### 默认运行（使用示例数据）

项目已包含示例数据文件 `data/600698_5Y_daily_qfq_20210302_20260302.xlsx`，可直接运行：

```bash
python main_cli.py
```

### 自定义运行

```bash
# 基础运行 - 100 次迭代
python main_cli.py --mode first --runs 100 --seed 42

# 深度搜索 - 1000 次迭代
python main_cli.py --mode first --runs 1000 --seed 42

# 继续模式 - 基于上次结果继续优化
python main_cli.py --mode continue --runs 500

# 使用自己的数据文件
python main_cli.py --data_path "path/to/your/data.xlsx" --runs 100
```

---

## 📖 使用说明

### 命令行参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | `first` / `continue` | `first` | 运行模式。`first` 从头开始，`continue` 基于历史最优继续 |
| `--data_path` | 字符串 | 内置默认路径 | Excel 数据文件路径 |
| `--output_dir` | 字符串 | `./output` | 输出目录 |
| `--runs` | 整数 | `100` | 随机搜索迭代次数（建议 100/500/1000/5000） |
| `--seed` | 整数 | `42` | 随机种子，用于复现结果 |
| `--initial_cash` | 浮点数 | `100000` | 初始资金（元） |

### 运行模式说明

#### `first` 模式
- 从头开始随机搜索
- 适用于首次运行或更换数据后

#### `continue` 模式
- 读取 `output/` 目录下最新运行的最优配置
- 在此基础上继续优化
- 适用于迭代改进策略

### 数据文件格式

Excel 文件需包含以下列：

| 列名 | 说明 | 示例 |
|------|------|------|
| `date` | 交易日期 | `2021-03-02` |
| `open_qfq` | 前复权开盘价 | `10.5` |
| `high_qfq` | 前复权最高价 | `11.2` |
| `low_qfq` | 前复权最低价 | `10.3` |
| `close_qfq` | 前复权收盘价 | `11.0` |
| `volume` | 成交量 | `1000000` |
| `amount` | 成交额 | `11000000` |
| `turnover_rate` | 换手率 | `2.5` |

> 💡 **提示**: 数据可通过 Tushare、Baostock、聚宽等数据源获取，导出为 Excel 格式即可。

---

## ⚙️ 配置参数

### 特征配置（自动搜索）

| 参数 | 说明 | 搜索范围 |
|------|------|----------|
| `windows` | 时间窗口 | `[3,5,10,20]`, `[5,10,20,60]` 等 |
| `use_momentum` | 动量特征 | `True` / `False` |
| `use_volatility` | 波动率特征 | `True` / `False` |
| `use_volume` | 成交量特征 | `True` / `False` |
| `use_candle` | K 线形态特征 | `True` / `False` |
| `use_turnover` | 换手率特征 | `True` / `False` |
| `vol_metric` | 波动率度量 | `std` / `mad` |
| `liq_transform` | 流动性变换 | `ratio` / `zscore` |

### 模型配置（自动搜索）

| 模型类型 | 搜索参数 |
|----------|----------|
| **Logistic Regression** | penalty (l1/l2/elasticnet), solver, C, class_weight, l1_ratio |
| **SGD Classifier** | alpha, penalty, class_weight, l1_ratio |
| **XGBoost** | n_estimators, max_depth, learning_rate, subsample, colsample_bytree |

### 交易配置（自动搜索）

| 参数 | 说明 | 搜索范围 |
|------|------|----------|
| `buy_threshold` | 买入概率阈值 | `0.50` ~ `0.70` |
| `sell_threshold` | 卖出概率阈值 | `0.30` ~ `0.50` |
| `confirm_days` | 信号确认天数 | `1` ~ `5` |
| `min_hold_days` | 最小持仓天数 | `1` ~ `5` |
| `max_hold_days` | 最大持仓天数 | `10` ~ `60` |
| `take_profit` | 止盈阈值（可选） | `0.05` ~ `0.20` |
| `stop_loss` | 止损阈值（可选） | `0.05` ~ `0.15` |

---

## 📊 输出文件

运行完成后，`output/` 目录将生成以下文件：

### run_XXX.xlsx - 回测报告

| Sheet 名称 | 内容 |
|------------|------|
| `EquityCurve` | 净值曲线（日期、净值、持仓等） |
| `Trades` | 交易记录（买卖日期、价格、收益等） |
| `SearchLog` | Walk-forward 验证日志（各折样本外指标） |
| `Data` | 清洗后的输入数据 |
| `LogNotes` | 运行日志和诊断信息 |

### run_XXX_config.json - 配置文件

包含最优配置的所有参数，可用于：
- 复现运行结果
- `continue` 模式加载
- 策略参数存档

### 关键指标说明

| 指标 | 说明 |
|------|------|
| `geom_mean_ratio` | 几何平均收益率（主要优化目标） |
| `total_return` | 总收益率 |
| `max_drawdown` | 最大回撤 |
| `win_rate` | 胜率 |
| `profit_factor` | 盈亏比 |
| `sharpe_ratio` | 夏普比率（年化） |
| `trade_count` | 交易次数 |

---

## 📁 项目结构

```
Ashare_DpointTrader 2.0/
├── main_cli.py              # 主入口，命令行接口
├── data_loader.py           # 数据加载与清洗
├── feature_dpoint.py        # 特征工程
├── model_builder.py         # 模型构建
├── splitter.py              # Walk-forward 数据分割
├── metrics.py               # 评估指标计算
├── search_engine.py         # 随机搜索引擎
├── trainer_optimizer.py     # 训练器 API
├── backtester_engine.py     # 回测执行引擎
├── reporter.py              # 结果报告生成
├── persistence.py           # 配置持久化
├── constants.py             # 全局常量
├── data/                    # 数据目录
│   └── 600698_5Y_daily_qfq_20210302_20260302.xlsx
└── output/                  # 输出目录（运行后生成）
    ├── run_001.xlsx
    ├── run_001_config.json
    └── ...
```

### 模块职责

| 模块 | 职责 |
|------|------|
| `data_loader.py` | Excel 读取、数据清洗、异常值过滤 |
| `feature_dpoint.py` | 构建 80+ 技术特征，生成 Dpoint 标签 |
| `model_builder.py` | 创建 sklearn/XGBoost 模型管道 |
| `splitter.py` | Walk-forward 时间序列分割 |
| `metrics.py` | 计算几何平均收益、惩罚项、交易统计 |
| `search_engine.py` | 随机搜索主循环、超参数采样 |
| `backtester_engine.py` | 信号生成、交易模拟、净值计算 |
| `reporter.py` | Excel 报告生成、配置保存 |
| `persistence.py` | 最优配置保存与加载 |

---

## 🔬 核心算法

### Dpoint 预测模型

```
Dpoint_t = P(close_{t+1} > close_t | X_t)
```

- **输入**: 截至 t 日的 OHLCV、成交额、换手率等特征
- **输出**: 次日上涨概率（0~1 之间）
- **关键**: 所有特征仅使用 t 日及之前数据，无未来函数

### Walk-forward 验证流程

```
┌─────────────────────────────────────────────────────────┐
│  Fold 1: 训练 [0:60%] → 验证 [60%:80%]                  │
│  Fold 2: 训练 [20%:80%] → 验证 [80%:100%]               │
│  ...                                                    │
└─────────────────────────────────────────────────────────┘
```

每折评估样本外交易表现，确保策略无过拟合。

### 随机搜索算法

```python
# 伪代码
for iteration in range(runs):
    if explore_mode:
        config = sample_global_space()  # 全局探索
    else:
        config = perturb_best_config()  # 局部利用
    
    score = evaluate_walkforward(config)  # Walk-forward 评估
    
    if score > best_score + epsilon:
        best_config = config
        best_score = score
```

### 交易执行逻辑

1. **信号生成**: `dpoint > buy_threshold` → 买入信号；`dpoint < sell_threshold` → 卖出信号
2. **信号确认**: 连续 N 日满足阈值才触发
3. **执行模拟**: T 日信号，T+1 日按 T 日收盘价执行（理想化）
4. **持仓管理**: 满足 `min_hold_days` 后才允许平仓

---

## ❓ 常见问题

### Q1: 运行时报错 "No module named 'xgboost'"

**A**: XGBoost 是可选依赖。如不使用 XGBoost 模型，可忽略此错误；如需使用：
```bash
pip install xgboost
```

### Q2: 运行速度太慢

**A**: 建议：
- 减少 `--runs` 参数（如从 1000 改为 100）
- 启用并行计算（默认开启，使用 `n_jobs=6`）
- 安装 XGBoost 并启用 GPU 加速

### Q3: 如何用自己的数据？

**A**: 
1. 准备 Excel 文件，包含必需列（见 [数据文件格式](#数据文件格式)）
2. 运行：
   ```bash
   python main_cli.py --data_path "your_data.xlsx"
   ```

### Q4: 结果如何复现？

**A**: 固定随机种子：
```bash
python main_cli.py --seed 42 --runs 100
```
使用相同种子、相同数据、相同代码版本可复现结果。

### Q5: `continue` 模式如何使用？

**A**: 
```bash
# 第一次运行
python main_cli.py --mode first --runs 100

# 基于第一次结果继续优化
python main_cli.py --mode continue --runs 100
```

### Q6: 输出文件中的 "IN-SAMPLE" 警告是什么意思？

**A**: 最终报告的净值曲线是全样本拟合结果（训练集=测试集），存在前向偏差，数值偏乐观。**真实样本外表现请查看 `SearchLog` sheet 中的 walk-forward 验证指标**。

---

## ⚠️ 免责声明

1. **本软件仅供学习和研究使用**，不构成任何投资建议或推荐。

2. **历史回测表现不代表未来收益**。量化策略存在过拟合、市场变化、执行偏差等风险。

3. **实盘交易风险极高**，可能导致本金损失。请在充分理解策略逻辑和风险的前提下谨慎使用。

4. **作者不对任何直接或间接损失承担责任**。用户使用本软件产生的一切后果由用户自行承担。

5. **请勿将本软件用于非法用途**。遵守当地法律法规和证券监管要求。

---

## 📧 联系方式

如有问题或建议，请通过 GitHub Issues 联系。


