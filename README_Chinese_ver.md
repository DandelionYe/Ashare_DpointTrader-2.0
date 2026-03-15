# A-Share Dpoint Trader 2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Language / 语言**: [English](README.md) | [简体中文](README_Chinese_ver.md)

---

**A 股单股票机器学习交易策略系统**

基于机器学习预测次日涨跌概率（Dpoint），结合 walk-forward 验证和随机搜索超参数优化的量化交易回测框架。

---

## 📋 目录

- [快速开始](#-快速开始)
- [功能特性](#-功能特性)
- [系统要求](#-系统要求)
- [安装步骤](#-安装步骤)
- [使用说明](#-使用说明)
- [配置 Schema](#-配置-schema)
- [命令行参数](#-命令行参数)
- [配置优先级](#-配置优先级)
- [复现文件说明](#-复现文件说明)
- [风险规则说明](#-风险规则说明)
- [真实执行模型](#-真实执行模型)
- [数据文件](#-数据文件)
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
git clone https://github.com/DandelionYe/Ashare_DpointTrader-2.0.git
cd Ashare_DpointTrader-2.0

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

### Windows 用户虚拟环境（可选）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# PowerShell:
.\venv\Scripts\Activate.ps1
# CMD:
.\venv\Scripts\activate.bat

# 安装依赖
pip install -r requirements.txt

# 运行
python main_cli.py
```

### 打开 Excel 报告

1. **首先查看** `WalkForwardSummary` Sheet（样本外指标 - 主 KPI）
2. **查看** `ExecutionAssumptions` Sheet（执行假设说明）
3. **然后查看** `Trades` Sheet（交易记录）
4. `EquityCurve` Sheet 仅供参考（样本内结果）

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
- **配置 Schema**: 使用 Pydantic 定义严格的配置结构，支持配置验证和序列化

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
git clone https://github.com/DandelionYe/Ashare_DpointTrader-2.0.git
cd Ashare_DpointTrader-2.0
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv venv
.\venv\Scripts\activate.bat

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

# 禁用元数据记录
python main_cli.py --no-record-metadata
```

---

## 🧬 配置 Schema

配置通过 `FullConfig` 管理，包含以下子配置：

### 1. FeatureConfig（特征工程）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `windows` | 整数列表 | `[3, 5, 10, 20]` | 滚动窗口大小 |
| `use_momentum` | 布尔 | `True` | 是否使用动量特征 |
| `use_volatility` | 布尔 | `True` | 是否使用波动率特征 |
| `use_volume` | 布尔 | `True` | 是否使用成交量特征 |
| `use_candle` | 布尔 | `True` | 是否使用 K 线形态特征 |
| `use_turnover` | 布尔 | `True` | 是否使用换手率特征 |
| `vol_metric` | `std`/`mad` | `std` | 波动率计算方式 |
| `liq_transform` | `ratio`/`zscore` | `ratio` | 流动性转换方式 |

### 2. ModelConfig（模型配置）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_type` | `logreg`/`sgd`/`xgb` | `logreg` | 模型类型 |
| `C` | 浮点数 | `1.0` | Logistic 回归正则化强度 |
| `penalty` | `l1`/`l2`/`elasticnet` | `l2` | 正则化类型 |
| `solver` | 字符串 | `lbfgs` | 优化算法 |
| `alpha` | 浮点数 | `1e-4` | SGD 正则化强度 |
| `n_estimators` | 整数 | `100` | XGBoost 树数量 |
| `max_depth` | 整数 | `3` | XGBoost 树最大深度 |
| `learning_rate` | 浮点数 | `0.1` | XGBoost 学习率 |

### 3. TradeConfig（交易执行）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `initial_cash` | 浮点数 | `100000` | 初始资金（元） |
| `buy_threshold` | 浮点数 | `0.55` | 买入 Dpoint 阈值 |
| `sell_threshold` | 浮点数 | `0.45` | 卖出 Dpoint 阈值 |
| `confirm_days` | 整数 | `2` | 信号确认天数 |
| `min_hold_days` | 整数 | `1` | 最短持仓天数 |
| `max_hold_days` | 整数 | `20` | 最长持仓天数 |
| `take_profit` | 浮点数 | `None` | 止盈阈值（EOD-based） |
| `stop_loss` | 浮点数 | `None` | 止损阈值（EOD-based） |
| `exec_price_model` | 字符串 | `next_open` | 执行价模型 |
| `slippage_bps` | 浮点数 | `10.0` | 滑点（基点） |
| `commission_rate` | 浮点数 | `0.00025` | 佣金率 |
| `commission_min` | 浮点数 | `5.0` | 最低佣金 |

### 4. SearchConfig（随机搜索超参数）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `runs` | 整数 | `100` | 随机搜索迭代次数 |
| `epsilon` | 浮点数 | `0.01` | 最小改进阈值 |
| `exploit_ratio` | 浮点数 | `0.7` | 局部扰动比例（70% 利用，30% 探索） |
| `top_k` | 整数 | `10` | Top-K 池大小 |
| `max_features` | 整数 | `80` | 最大特征数 |
| `n_jobs` | 整数 | `-1` | 并行进程数（-1=全部核心，1=单线程） |

---

## 🔧 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | `first` / `continue` | `first` | 运行模式 |
| `--data_path` | 字符串 | 内置默认路径 | Excel 数据文件路径 |
| `--output_dir` | 字符串 | `./output` | 输出目录 |
| `--runs` | 整数 | `100` | 随机搜索迭代次数 |
| `--seed` | 整数 | `42` | 随机种子（base seed） |
| `--initial_cash` | 浮点数 | `100000` | 初始资金（元） |
| `--exec_price_model` | 字符串 | `next_open` | 执行价模型 |
| `--slippage_bps` | 浮点数 | `10.0` | 滑点（基点） |
| `--commission_rate` | 浮点数 | `0.00025` | 佣金率 |
| `--commission_min` | 浮点数 | `5.0` | 最低佣金 |
| `--config` | 字符串 | `None` | 从配置文件加载（用于复现） |
| `--record-metadata` | 布尔 | `True` | 记录元数据（使用 `--no-record-metadata` 关闭） |
| `--log_dir` | 字符串 | `./logs` | 日志目录 |

### 运行模式说明

#### `first` 模式
- 从头开始随机搜索
- 适用于首次运行或更换数据后

#### `continue` 模式
- 读取 `output/` 目录下最新运行的最优配置
- 在此基础上继续优化
- 适用于迭代改进策略

### 随机种子语义

系统使用三种种子确保可复现性：

| 种子名称 | 说明 | 计算方式 |
|----------|------|----------|
| **Base Seed** | CLI 传入的基础种子（`--seed`） | 用户指定，默认 42 |
| **Search Seed** | 实际用于随机搜索的种子 | `base_seed + latest_run_id` |
| **Final Train Seed** | 用于最终全样本模型训练的种子 | 始终等于 `base_seed` |

**复现策略**：
- **首次运行（first 模式）**：三种种子相同
- **继续运行（continue 模式）**：Search Seed = Base Seed + 上次 run_id
- **精确复现**：使用 `--config run_XXX_metadata.json` 加载配置，并使用相同的 `--seed` 值

---

## 🔀 配置优先级

系统使用三级配置优先级，确保灵活性和可复现性：

```
CLI overrides > config file > built-in defaults
```

| 优先级 | 来源 | 说明 | 示例 |
|--------|------|------|------|
| **最高** | CLI 参数 | 命令行传入的参数 | `--runs 1000 --seed 42` |
| **中等** | 配置文件 | `--config run_XXX_metadata.json` | 从上次运行加载 |
| **最低** | 内置默认值 | 代码中的默认值 | `runs=100`, `seed=42` |

**重要说明**：
- `--runs` 参数：如果同时使用 `--config` 和 `--runs`，CLI 的 `--runs` 值会覆盖配置文件中的 `search_config.runs`
- `--seed` 参数：CLI 的 `--seed` 作为 base seed，search seed 会根据模式自动计算
- 执行成本参数（`--slippage_bps`、`--commission_rate` 等）：CLI 非默认值会覆盖配置文件

**推荐实践**：
- 复现上次运行：`python main_cli.py --config output/run_001_metadata.json`（不传 `--runs`，使用文件中的值）
- 修改参数重新运行：`python main_cli.py --config output/run_001_metadata.json --runs 500`（覆盖 runs）

---

## 📋 复现文件说明

系统生成两种配置文件，用途不同：

### run_XXX_metadata.json - 完整运行元数据

**用途**：精确复现整个运行，包含所有运行时上下文。

**包含字段**：
```json
{
  "run_id": 1,
  "created_at": "2026-03-15T10:00:00",
  "code_version": "abc12345",
  "python_version": "3.13.7",
  "dependency_versions": {...},
  "data_hash": "...",
  "data_path": "data/600698_5Y_daily_qfq_*.xlsx",
  "base_seed": 42,
  "search_seed": 42,
  "final_train_seed": 42,
  "mode": "first",
  "effective_runs": 100,
  "effective_config_source": "CLI/default",
  "config": {...},  // 完整复现配置 (repro_config)
  "git_commit": "...",
  "hostname": "...",
  "notes": [...]
}
```

**使用方法**：
```bash
python main_cli.py --config output/run_001_metadata.json
```

### run_XXX_config.json - 配置快照

**用途**：快速查看和对比配置，便于人工阅读。

**包含字段**：
```json
{
  "run_id": 1,
  "created_at": "...",
  "data_hash": "...",
  "best_strategy_config": {...},  // 最佳策略参数（仅特征/模型/交易阈值）
  "repro_config": {...},          // 完整复现配置（包含执行假设/search_config/分割参数）
  "run_context": {...},           // 运行上下文（mode/seeds/config_source 等）
  "feature_meta": {...},
  "notes": {...}
}
```

**关键区别**：
- `best_strategy_config`：仅包含可优化的策略参数（特征窗口、模型超参、买卖阈值）
- `repro_config`：包含完整复现所需的所有配置（执行成本、滑点、search_config、n_folds 等）

**推荐实践**：
- 人工查看/对比配置：打开 `run_XXX_config.json`
- 程序化复现：使用 `run_XXX_metadata.json` 或 `run_XXX_config.json` 均可

---

## ⚠️ 风险规则说明

### 止盈/止损触发机制（EOD-based）

系统当前使用 **EOD-based（End-of-Day，收盘价判断）** 机制，**不是** intraday-based（盘中高低点判断）。

**触发逻辑**：
1. 每日收盘后，使用当日 `close_qfq` 计算盈亏比例
2. 若 `pnl_ratio >= take_profit`，标记为 "EOD take_profit reached"，次日执行卖出
3. 若 `pnl_ratio <= -stop_loss`，标记为 "EOD stop_loss reached"，次日执行卖出

**不会触发的情况**：
- 盘中 `high_qfq` 触及止盈，但 `close_qfq` 未触及 → **不触发**
- 盘中 `low_qfq` 跌破止损，但 `close_qfq` 未跌破 → **不触发**

**示例**：
```
第 5 天数据：
  open_qfq:  10.0 元
  high_qfq:  11.5 元  (+15%，盘中触及 10% 止盈)
  low_qfq:   10.0 元
  close_qfq: 10.5 元  (+5%，收盘未触及 10% 止盈)

结果：止盈不触发，持仓继续保持
```

**设计理由**：
- EOD-based 更稳定，避免盘中噪音触发
- 实盘更容易执行（无需实时监控）
- 回测更保守，避免高估止盈/止损效果

**未来可能增强**：
- 支持 `intraday` 模式（使用 `high_qfq`/`low_qfq` 判断）
- 支持 `mixed` 模式（EOD 为主，intraday 为辅）

---

## 📊 数据文件

### 文件格式

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

### 缺失值处理

对于非核心列（`volume`、`amount`、`turnover_rate`）的缺失值，系统提供以下处理策略：

| 策略 | 说明 | 标记列 |
|------|------|--------|
| `zero` (默认) | 填充为 0，并添加 `_was_missing` 标记列 | `volume_was_missing` 等 |
| `ffill` | 使用前一日值向前填充 | `volume_was_missing` 等 |
| `drop` | 直接删除缺失行 | - |
| `keep_nan` | 保持 NaN，由特征工程处理 | `volume_was_missing` 等 |

**默认策略说明**：
- 填充为 0 时，会同时添加 `volume_was_missing`、`amount_was_missing`、`turnover_rate_was_missing` 布尔列
- 这些标记列可用于区分"真实 0 成交"和"数据缺失补 0"

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
| **印花税** | 0.05% (5‱) | 仅卖出 | -

### 滑点模型

滑点以固定基点（bps）调整：
- **买入**: 执行价 = 基准价 × (1 + slippage_bps / 10000)
- **卖出**: 执行价 = 基准价 × (1 - slippage_bps / 10000)

**默认**: 10 bps (0.1%)

### 止盈/止损触发机制

系统使用 **EOD-based（收盘价判断）** 机制：
- 在每日收盘后，使用当日收盘价计算盈亏比例
- 若 `pnl_ratio >= take_profit`，触发止盈，次日执行卖出
- 若 `pnl_ratio <= -stop_loss`，触发止损，次日执行卖出
- **不会**使用盘中高低点判断触发

---

## 📁 输出文件

运行完成后，`output/` 目录将生成以下文件：

### run_XXX.xlsx - 回测报告

**重要：Sheet 按重要性排序，请首先查看 [WalkForwardSummary]。**

| Sheet 名称 | 内容 | 优先级 |
|------------|------|--------|
| `WalkForwardSummary` | **Walk-Forward 样本外验证指标摘要**（主 KPI） | ⭐⭐⭐⭐⭐ |
| `ExecutionAssumptions` | **执行假设说明**（执行价模型、滑点、交易成本、交易规则） | ⭐⭐⭐⭐⭐ |
| `Trades` | 交易记录（买卖日期、价格、收益、成本、滑点等） | ⭐⭐⭐⭐ |
| `EquityCurve` | 净值曲线（日期、净值、持仓、回撤、信号等） | ⭐⭐⭐ |
| `FinalFit_InSample` | **样本内结果警告**（仅供参考，不代表样本外表现） | ⚠️ |
| `SearchLog` | 完整的随机搜索迭代日志（每折详细指标、配置参数） | ⭐⭐⭐⭐ |
| `Config` | 配置参数（特征、模型、交易参数、约束条件） | ⭐⭐⭐ |
| `ModelParams` | 模型参数（特征系数、标准化参数、截距） | ⭐⭐ |
| `Log` | 运行日志和诊断信息 | ⭐⭐ |

### run_XXX_config.json - 配置文件

包含最优配置和特征元数据的 JSON 文件，可用于后续运行的 `--config` 参数。

### run_XXX_metadata.json - 元数据文件

包含完整运行元数据，用于精确复现：
- 代码版本（git commit）
- Python 版本
- 所有依赖版本
- 数据文件哈希
- 随机种子（base seed）
- 主机名和时间戳

---

## 📂 项目结构

```
Ashare_DpointTrader-2.0/
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
├── config_schema.py         # 配置 Schema 定义和验证
├── structured_logging.py    # 结构化日志配置
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
    ├── test_integration.py
    └── test_e2e_config.py
```

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
│  Walk-forward 切分 (n_folds=4, train_start_ratio=0.5)   │
│                                                         │
│  Fold 1: 训练 [0%~50%]  →  验证 [50%~62.5%]             │
│  Fold 2: 训练 [0%~62%]  →  验证 [62.5%~75%]             │
│  Fold 3: 训练 [0%~75%]  →  验证 [75%~87.5%]             │
│  Fold 4: 训练 [0%~87%]  →  验证 [87.5%~100%]            │
│                                                         │
│  特点：训练集累积扩展 (expanding window)                │
│        验证集不重叠                                     │
└─────────────────────────────────────────────────────────┘
```

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
2. **信号确认**: 连续 N 日满足阈值才触发（`confirm_days`）
3. **执行模拟**: T 日信号，T+1 日执行（使用配置的执行价模型）
   - `same_close_idealized`: 用 T 日收盘价（理想化，旧版行为）
   - `next_open`: 用 T+1 日开盘价（推荐，更真实）
   - `next_close`: 用 T+1 日收盘价（保守）
4. **持仓管理**: 满足 `min_hold_days` 后才允许平仓

---

## 🧪 测试

### 运行所有测试

```bash
cd Ashare_DpointTrader-2.0
pytest tests/ -v
```

### 运行特定测试

```bash
# 回测引擎测试
pytest tests/test_backtester.py -v

# 特征工程测试
pytest tests/test_features.py -v

# 端到端配置测试
pytest tests/test_e2e_config.py -v

# 集成测试（最慢）
pytest tests/test_integration.py -v
```

### 生成覆盖率报告

```bash
# 安装覆盖率工具
pip install pytest-cov

# 运行并生成覆盖率报告
pytest --cov=. --cov-report=html
```

---

## ⚙️ 工程化特性

- **配置 Schema**: 使用 Pydantic 定义严格的配置结构，支持验证和序列化
- **配置持久化**: 最优配置自动保存至 JSON，支持复现
- **结构化日志**: JSON 格式日志，支持性能追踪和问题诊断
- **启动检查**: 运行前自动检查依赖、数据文件和输出目录
- **版本追踪**: 记录代码版本、Python 版本、依赖版本、git commit
- **数据哈希**: 使用 SHA-256 计算数据文件哈希，确保数据一致性
- **测试覆盖**: 端到端测试保护核心功能，防止重构引入回归

---

## ❓ 常见问题

### Q: 为什么 WalkForwardSummary 最重要？

**A**: Walk-Forward 每折验证集都是未见过的数据，反映真实可期望表现。训练集和验证集严格分离，避免前向偏差。

### Q: 为什么 EquityCurve 仅供参考？

**A**: `EquityCurve` 展示的是**全样本拟合结果**（In-Sample Fit）：
- 模型在全部数据上训练并预测，存在信息泄露
- 未考虑隔夜跳空、滑点和成交偏差（除非使用真实执行模型）
- 数值偏乐观，**不代表未来实盘表现**

**真实样本外表现请查看 [WalkForwardSummary] 和 [SearchLog] 中的指标。**

### Q: 如何复现上次运行？

**A**: 使用 `--config` 参数加载上次的 metadata 文件：
```bash
python main_cli.py --config output/run_001_metadata.json
```

### Q: continue 模式和 first 模式有什么区别？

**A**: 
- `first` 模式：从头开始随机搜索
- `continue` 模式：读取 `output/` 目录下最新运行的最优配置，在此基础上继续优化

---

## ⚠️ 免责声明

本软件仅供学术研究和教育用途，不构成投资建议。使用本软件进行的任何交易行为，风险由用户自行承担。作者不对任何直接或间接的损失负责。

金融市场交易存在重大风险，包括但不限于：
- 市场波动风险
- 流动性风险
- 技术故障风险
- 模型失效风险

请在充分理解风险并评估自身承受能力后谨慎使用。
