# A-Share Dpoint Trader 2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A 股单股票机器学习交易策略系统**

基于机器学习预测次日涨跌概率（Dpoint），结合 walk-forward 验证和随机搜索超参数优化的量化交易回测框架。

---

## 📋 目录

- [快速开始](#-快速开始)
- [功能特性](#-功能特性)
- [系统要求](#-系统要求)
- [安装步骤](#-安装步骤)
- [使用说明](#-使用说明)
- [配置参数](#-配置参数)
- [真实执行模型](#-真实执行模型)
- [输出文件](#-输出文件)
- [项目结构](#-项目结构)
- [核心算法](#-核心算法)
- [测试](#-测试)
- [工程化特性](#-工程化特性)
- [常见问题](#-常见问题)
- [免责声明](#-免责声明)

---

## 🚀 快速开始

### 5 分钟开始使用

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd "Ashare_DpointTrader 2.0"

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证安装
python main_cli.py --help

# 4. 运行示例（使用内置示例数据）
python main_cli.py

# 5. 查看结果（在 output/ 目录）
# - run_001.xlsx          # 回测报告
# - run_001_config.json   # 最优配置
# - run_001_metadata.json # 完整元数据（用于复现）
```

### 打开 Excel 报告

1. **首先查看** `WalkForwardSummary` Sheet（样本外指标 - 主 KPI）
2. **然后查看** `Trades` Sheet（交易记录）
3. `EquityCurve` Sheet 仅供参考（样本内结果）

---

## ✨ 功能特性

### 核心功能

- **Dpoint 预测模型**: 预测次日收盘价上涨的概率 `P(close_{t+1} > close_t | X_t)`
- **Walk-forward 验证**: 时间序列交叉验证，避免前向偏差，确保样本外评估
- **随机搜索优化**: 自动搜索最优特征组合、模型参数和交易策略参数
- **多模型支持**: Logistic Regression, SGDClassifier, XGBoost (支持 GPU 加速)
- **A 股交易约束**: 支持 T+1、最小 100 股单位、仅做多等 A 股规则
- **真实执行模型**: 交易成本（佣金、印花税、过户费）、滑点、多种执行价模型

### 技术特性

- **特征工程**: 动量、波动率、成交量、换手率、K 线形态等 5 大家族 80+ 特征
- **早停剪枝**: 自动淘汰劣质配置，加速搜索过程
- **并行搜索**: 利用多核 CPU 并行评估候选配置
- **CUDA 加速**: 自动检测并启用 XGBoost GPU 加速
- **持续学习**: 支持 `continue` 模式，基于历史最优配置继续优化
- **结构化日志**: JSON 格式日志，支持性能追踪
- **完整元数据**: 记录代码版本、依赖版本、git commit，支持精确复现

---

## 💻 系统要求

### 必需环境

- **操作系统**: Windows / Linux / macOS
- **Python**: 3.8 或更高版本
- **内存**: 建议 8GB 以上

### Python 依赖

**必需：**
```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
openpyxl>=3.0.0        # Excel 读写
xlsxwriter>=3.0.0      # Excel 输出（必需）
joblib>=1.1.0          # 并行计算
```

**可选：**
```
xgboost>=1.5.0         # XGBoost 模型（启用 GPU 加速）
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

**方式 A：使用 requirements.txt（推荐）**
```bash
pip install -r requirements.txt
```

**方式 B：使用 pyproject.toml**
```bash
pip install .
```

**方式 C：手动安装**
```bash
pip install pandas numpy scikit-learn openpyxl xlsxwriter joblib

# 可选：安装 XGBoost（启用 GPU 加速）
pip install xgboost
```

### 4. 验证安装

```bash
# 快速验证（检查依赖和数据文件）
python main_cli.py --help

# 或运行启动检查（不执行回测）
python setup_check.py
```

---

## 🎯 使用说明

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

# 真实执行模式 - 使用开盘价 + 交易成本 + 滑点（推荐）
python main_cli.py --mode first --runs 100 --exec_price_model next_open --slippage_bps 10

# 复现上次运行
python main_cli.py --config output/run_001_metadata.json
```

### 命令行参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | `first` / `continue` | `first` | 运行模式。`first` 从头开始，`continue` 基于历史最优继续 |
| `--data_path` | 字符串 | 内置默认路径 | Excel 数据文件路径 |
| `--output_dir` | 字符串 | `./output` | 输出目录 |
| `--runs` | 整数 | `100` | 随机搜索迭代次数（建议 100/500/1000/5000） |
| `--seed` | 整数 | `42` | 随机种子，用于复现结果 |
| `--initial_cash` | 浮点数 | `100000` | 初始资金（元） |
| `--exec_price_model` | 字符串 | `next_open` | 执行价模型：`same_close_idealized`, `next_open`, `next_close` |
| `--slippage_bps` | 浮点数 | `10.0` | 滑点（基点），默认 10 bps（0.1%） |
| `--commission_rate` | 浮点数 | `0.00025` | 佣金率，默认万分之 2.5 |
| `--commission_min` | 浮点数 | `5.0` | 最低佣金，默认 5 元 |
| `--config` | 字符串 | `None` | 从配置文件加载配置（用于复现） |
| `--record_metadata` | 布尔 | `True` | 记录完整元数据（默认开启） |
| `--log_dir` | 字符串 | `./logs` | 结构化日志目录 |

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

## 🔬 真实执行模型

Version 2.0 引入**真实执行模型**，考虑：
- **执行价模型**: `same_close_idealized`, `next_open`, `next_close`
- **交易成本**: 佣金、过户费、印花税
- **滑点**: 固定基点（bps）滑点模型

### 执行价模型

| 模型 | 信号日 | 执行日 | 使用价格 | 说明 |
|------|--------|--------|----------|------|
| `same_close_idealized` | t | t+1 | t 日收盘价 | 理想化（旧版行为），低估隔夜跳空 |
| `next_open` | t | t+1 | t+1 日开盘价 | **推荐**，更真实，捕捉隔夜风险 |
| `next_close` | t | t+1 | t+1 日收盘价 | 保守估计 |

### 交易成本（A 股标准）

| 费用类型 | 费率 | 方向 | 最低 |
|----------|------|------|------|
| **佣金** | 0.025% (2.5‱) | 双向 | 5 元 |
| **过户费** | 0.001% (0.1‱) | 双向 | - |
| **印花税** | 0.05% (5‱) | 仅卖出 | - |

**计算示例：**

买入 100 股 @ 10 元：
- 成交额：100 × 10 = 1,000 元
- 佣金：max(1,000 × 0.025%, 5) = 5 元
- 过户费：1,000 × 0.001% = 0.01 元
- **总成本**: 5.01 元

卖出 100 股 @ 10 元：
- 成交额：100 × 10 = 1,000 元
- 佣金：max(1,000 × 0.025%, 5) = 5 元
- 过户费：1,000 × 0.001% = 0.01 元
- 印花税：1,000 × 0.05% = 0.5 元
- **总成本**: 5.51 元

### 滑点模型

滑点以固定基点（bps）调整：

- **买入**: 执行价 = 基准价 × (1 + slippage_bps / 10000)
- **卖出**: 执行价 = 基准价 × (1 - slippage_bps / 10000)

**默认**: 10 bps (0.1%)

**示例**:
- 基准价：10.00 元
- 滑点：10 bps
- 买入执行价：10.00 × 1.001 = **10.01 元**
- 卖出执行价：10.00 × 0.999 = **9.99 元**

### 预期影响

从 `same_close_idealized` 切换到 `next_open + fee + slippage` 后：

| 组件 | 典型影响 |
|------|----------|
| 隔夜跳空（close→open） | -0.5% ~ -2% / 每笔交易 |
| 滑点（10 bps） | -0.1% / 每笔交易 |
| 交易成本 | -0.5% ~ -1% / 每笔往返 |
| **总计** | **-1% ~ -3% / 每笔交易** |

**高频策略**（年交易次数多）累计影响更大。

---

## 📊 输出文件

运行完成后，`output/` 目录将生成以下文件：

### run_XXX.xlsx - 回测报告

**重要：Sheet 按重要性排序，请首先查看 [WalkForwardSummary]。**

| Sheet 名称 | 内容 | 优先级 |
|------------|------|--------|
| `WalkForwardSummary` | **Walk-Forward 样本外验证指标摘要**（主 KPI） | ⭐⭐⭐⭐⭐ |
| `Trades` | 交易记录（买卖日期、价格、收益、成本、滑点等） | ⭐⭐⭐⭐ |
| `EquityCurve` | 净值曲线（日期、净值、持仓、回撤、信号等） | ⭐⭐⭐ |
| `FinalFit_InSample` | **样本内结果警告**（仅供参考，不代表样本外表现） | ⚠️ |
| `SearchLog` | 完整的随机搜索迭代日志（每折详细指标、配置参数） | ⭐⭐⭐⭐ |
| `Config` | 配置参数（特征、模型、交易参数、约束条件） | ⭐⭐⭐ |
| `ModelParams` | 模型参数（特征系数、标准化参数、截距） | ⭐⭐ |
| `Log` | 运行日志和诊断信息 | ⭐⭐ |

### 为什么 [WalkForwardSummary] 最重要？

- **样本外验证**: Walk-Forward 每折验证集都是未见过的数据，反映真实可期望表现
- **无信息泄露**: 训练集和验证集严格分离，避免前向偏差
- **稳健性评估**: 多折验证可评估策略在不同市场条件下的稳定性

### ⚠️ 为什么 [EquityCurve] 仅供参考？

`EquityCurve` 展示的是**全样本拟合结果**（In-Sample Fit）：
- 模型在全部数据上训练并预测，存在信息泄露
- 未考虑隔夜跳空、滑点和成交偏差（除非使用真实执行模型）
- 数值偏乐观，**不代表未来实盘表现**

**真实样本外表现请查看 [WalkForwardSummary] 和 [SearchLog] 中的指标。**

### run_XXX_config.json - 配置文件

包含最优配置的所有参数，可用于：
- 复现运行结果
- `continue` 模式加载
- 策略参数存档

### run_XXX_metadata.json - 完整元数据（新增）

包含完整运行元数据：
- 代码版本（git commit）
- Python 版本
- 所有依赖版本
- 数据文件哈希
- 随机种子
- 主机名和时间戳

用于**精确复现**运行。

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
├── config_schema.py         # 配置 Schema 定义和验证（工程化）
├── structured_logging.py    # 结构化日志配置（工程化）
├── setup_check.py           # 启动前检查脚本
├── requirements.txt         # Python 依赖列表
├── pyproject.toml           # 项目配置文件
├── pytest.ini               # pytest 测试配置
├── data/                    # 数据目录
│   └── 600698_5Y_daily_qfq_20210302_20260302.xlsx
├── output/                  # 输出目录（运行后生成）
│   ├── run_001.xlsx
│   ├── run_001_config.json
│   ├── run_001_metadata.json
│   └── ...
├── logs/                    # 日志目录（运行后生成）
│   └── dpoint_trader_*.log
└── tests/                   # 测试目录
    ├── test_backtester.py
    ├── test_features.py
    ├── test_splitter.py
    ├── test_reporter.py
    ├── test_main_cli.py
    └── test_integration.py
```

### 模块职责

| 模块 | 职责 |
|------|------|
| `main_cli.py` | 主入口、CLI 参数解析、启动检查、元数据记录 |
| `data_loader.py` | Excel 读取、数据清洗、异常值过滤 |
| `feature_dpoint.py` | 构建 80+ 技术特征，生成 Dpoint 标签 |
| `model_builder.py` | 创建 sklearn/XGBoost 模型管道 |
| `splitter.py` | Walk-forward 时间序列分割 |
| `metrics.py` | 计算几何平均收益、惩罚项、交易统计 |
| `search_engine.py` | 随机搜索主循环、超参数采样 |
| `trainer_optimizer.py` | 训练器 API、最终模型训练 |
| `backtester_engine.py` | 信号生成、交易模拟、净值计算（含交易成本、滑点） |
| `reporter.py` | Excel 报告生成、配置保存 |
| `persistence.py` | 最优配置保存与加载 |
| `config_schema.py` | 配置 Schema 定义和验证（工程化） |
| `structured_logging.py` | 结构化日志配置（工程化） |
| `setup_check.py` | 启动前检查（依赖、数据、输出目录） |

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

## 🧪 测试

### 运行所有测试

```bash
cd "Ashare_DpointTrader 2.0"
pytest tests/ -v
```

### 运行特定模块

```bash
# 回测引擎测试
pytest tests/test_backtester.py -v

# 特征工程测试
pytest tests/test_features.py -v

# 数据分割测试
pytest tests/test_splitter.py -v

# 报告生成测试
pytest tests/test_reporter.py -v

# CLI 测试
pytest tests/test_main_cli.py -v

# 集成测试（最慢）
pytest tests/test_integration.py -v
```

### 生成覆盖率报告

```bash
# 安装覆盖率工具
pip install pytest-cov

# 运行并生成覆盖率报告
pytest --cov=. --cov-report=html

# 查看覆盖率报告
# Windows: start htmlcov/index.html
# Linux/macOS: open htmlcov/index.html
```

### 测试覆盖的关键路径

| 测试模块 | 覆盖内容 |
|----------|----------|
| `test_backtester.py` | T+1 执行、交易成本、滑点、持仓约束、止盈止损 |
| `test_features.py` | 标签构建、特征族、无未来函数 |
| `test_splitter.py` | Walk-forward 切分、无数据泄露 |
| `test_reporter.py` | Sheet 名称、配置 JSON、公式转义 |
| `test_main_cli.py` | 默认路径、参数解析、启动检查 |
| `test_integration.py` | 完整流程集成测试 |

---

## 🔧 工程化特性

### 1. 配置 Schema 化

使用严格的配置 Schema，替代松散的 dict：

```python
from config_schema import FullConfig, FeatureConfig, ModelConfig, TradeConfig

config = FullConfig(
    feature_config=FeatureConfig(windows=[3, 5], ...),  # 类型检查
    model_config=ModelConfig(model_type="logreg", ...),  # 验证
    trade_config=TradeConfig(...),
)
errors = config.validate()  # 显式验证
```

### 2. 结构化日志

统一日志格式，输出到文件和控制台：

```python
from structured_logging import setup_logger

logger = setup_logger(
    name="dpoint_trader",
    level="INFO",
    log_dir="./logs",
    console_output=True,
    file_output=True,
)

# 记录带额外字段的日志
logger.info("Starting training", extra={"runs": 100, "seed": 42})
```

**日志输出示例：**
```json
{"timestamp": "2026-03-14T10:30:00", "level": "INFO", "message": "Starting training", "runs": 100, "seed": 42}
```

### 3. 完整元数据记录

记录完整运行元数据用于复现：
- 代码版本（git commit）
- Python 版本
- 所有依赖版本
- 数据文件哈希
- 随机种子
- 主机名和时间戳

**使用方式：**
```bash
# 默认自动记录元数据
python main_cli.py --runs 100

# 从元数据文件复现
python main_cli.py --config output/run_001_metadata.json
```

### 4. 一键复现

```bash
# 精确复现（相同种子，相同配置）
python main_cli.py --config output/run_042_metadata.json

# 加载配置但修改参数
python main_cli.py --config output/run_042_config.json --runs 500 --seed 123
```

---

## ❓ 常见问题

### Q1: 运行时报错 "No module named 'xlsxwriter'"

**A**: xlsxwriter 是必需依赖。安装：
```bash
pip install xlsxwriter
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

**A**: 固定随机种子并使用元数据文件：
```bash
python main_cli.py --seed 42 --runs 100
# 或
python main_cli.py --config output/run_001_metadata.json
```

### Q5: `continue` 模式如何使用？

**A**:
```bash
# 第一次运行
python main_cli.py --mode first --runs 100

# 基于第一次结果继续优化
python main_cli.py --mode continue --runs 100
```

### Q6: 输出文件中的 "IN-SAMPLE" 警告是什么意思？

**A**: 最终报告的净值曲线是全样本拟合结果（训练集=测试集），存在前向偏差，数值偏乐观。**真实样本外表现请查看 `WalkForwardSummary` sheet 中的 walk-forward 验证指标**。

### Q7: 找不到数据文件？

**A**:
1. 确认 `data/600698_5Y_daily_qfq_20210302_20260302.xlsx` 存在
2. 或使用 `--data_path` 指定你的数据文件
3. 或设置环境变量：
   ```bash
   # Windows
   set ASHARE_DATA_PATH=path/to/your/data.xlsx
   
   # Linux/macOS
   export ASHARE_DATA_PATH=path/to/your/data.xlsx
   ```

### Q8: 如何调整交易成本和滑点？

**A**:
```bash
# 自定义佣金和滑点
python main_cli.py --commission_rate 0.0003 --slippage_bps 15
```

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

---

## 📝 版本历史

### Version 2.0 (当前版本)

**新增功能：**
- ✅ 真实执行模型（交易成本、滑点、多种执行价模型）
- ✅ 样本内/样本外结果分离（WalkForwardSummary sheet）
- ✅ 配置 Schema 化（类型检查、验证）
- ✅ 结构化日志（JSON 格式、性能追踪）
- ✅ 完整元数据记录（用于复现）
- ✅ 测试套件（100+ 测试用例）
- ✅ 启动检查（依赖、数据、输出目录）

**改进：**
- ✅ 默认数据路径改为相对路径
- ✅ 补全 requirements.txt 和 pyproject.toml
- ✅ 文档全面更新，与实际代码对齐
