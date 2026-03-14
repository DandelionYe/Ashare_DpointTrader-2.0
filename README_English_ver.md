# A-Share Dpoint Trader 2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A-Share Single-Stock Machine Learning Trading Strategy System**

A quantitative trading backtesting framework based on machine learning to predict next-day price movement probability (Dpoint), combined with walk-forward validation and random search hyperparameter optimization.

---

## Table of Contents

- [Quick Start](#-quick-start)
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Realistic Execution](#-realistic-execution)
- [Output Files](#-output-files)
- [Project Structure](#-project-structure)
- [Core Algorithms](#-core-algorithms)
- [Testing](#-testing)
- [Engineering Features](#-engineering-features)
- [FAQ](#-faq)
- [Disclaimer](#-disclaimer)

---

## 🚀 Quick Start

### 5-Minute Setup

```bash
# 1. Clone project
git clone <your-repo-url>
cd "Ashare_DpointTrader 2.0"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python main_cli.py --help

# 4. Run example (using built-in sample data)
python main_cli.py

# 5. View results (in output/ directory)
# - run_001.xlsx          # Backtest report
# - run_001_config.json   # Best configuration
# - run_001_metadata.json # Full metadata (for reproduction)
```

### Opening Excel Report

1. **First check** `WalkForwardSummary` sheet (Out-of-sample metrics - PRIMARY KPIs)
2. **Then check** `Trades` sheet (Trade records)
3. `EquityCurve` sheet is for reference only (In-sample results)

---

## ✨ Features

### Core Capabilities

- **Dpoint Prediction Model**: Predicts probability of next-day close price increase `P(close_{t+1} > close_t | X_t)`
- **Walk-forward Validation**: Time-series cross-validation to avoid look-ahead bias, ensures out-of-sample evaluation
- **Random Search Optimization**: Auto-search for optimal feature combinations, model parameters, and trading strategy parameters
- **Multi-Model Support**: Logistic Regression, SGDClassifier, XGBoost (with GPU acceleration)
- **A-Share Trading Constraints**: Supports T+1, minimum 100-share units, long-only, and other A-share rules
- **Realistic Execution Model**: Transaction costs (commission, stamp tax, transfer fee), slippage, multiple execution price models

### Technical Features

- **Feature Engineering**: 80+ features across 5 families - momentum, volatility, volume, turnover rate, candlestick patterns
- **Early Stopping Pruning**: Automatically eliminates poor configurations to accelerate search
- **Parallel Search**: Utilizes multi-core CPU for parallel candidate evaluation
- **CUDA Acceleration**: Auto-detects and enables XGBoost GPU acceleration
- **Continuous Learning**: Supports `continue` mode to optimize based on historical best configurations
- **Structured Logging**: JSON format logs with performance tracking
- **Full Metadata Recording**: Records code version, dependency versions, git commit for exact reproduction

---

## 💻 Requirements

### System Requirements

- **OS**: Windows / Linux / macOS
- **Python**: 3.8 or higher
- **Memory**: 8GB or more recommended

### Python Dependencies

**Required:**
```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
openpyxl>=3.0.0        # Excel read/write
xlsxwriter>=3.0.0      # Excel output (required)
joblib>=1.1.0          # Parallel computing
```

**Optional:**
```
xgboost>=1.5.0         # XGBoost model (GPU acceleration)
```

---

## 📥 Installation

### 1. Clone or Download Project

```bash
git clone <your-repo-url>
cd "Ashare_DpointTrader 2.0"
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

**Option A: Using requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Using pyproject.toml**
```bash
pip install .
```

**Option C: Manual Installation**
```bash
pip install pandas numpy scikit-learn openpyxl xlsxwriter joblib

# Optional: Install XGBoost (for GPU acceleration)
pip install xgboost
```

### 4. Verify Installation

```bash
# Quick verification (check dependencies and data file)
python main_cli.py --help

# Or run startup checks (without backtest)
python setup_check.py
```

---

## 🎯 Usage

### Run with Default Sample Data

The project includes a sample data file `data/600698_5Y_daily_qfq_20210302_20260302.xlsx`, you can run directly:

```bash
python main_cli.py
```

### Custom Run

```bash
# Basic run - 100 iterations
python main_cli.py --mode first --runs 100 --seed 42

# Deep search - 1000 iterations
python main_cli.py --mode first --runs 1000 --seed 42

# Continue mode - Continue optimization based on previous results
python main_cli.py --mode continue --runs 500

# Use your own data file
python main_cli.py --data_path "path/to/your/data.xlsx" --runs 100

# Realistic execution mode - Use open price + costs + slippage (recommended)
python main_cli.py --mode first --runs 100 --exec_price_model next_open --slippage_bps 10

# Reproduce previous run
python main_cli.py --config output/run_001_metadata.json
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | `first` / `continue` | `first` | Run mode. `first` starts from scratch, `continue` continues based on historical best |
| `--data_path` | String | Built-in default path | Excel data file path |
| `--output_dir` | String | `./output` | Output directory |
| `--runs` | Integer | `100` | Random search iterations (recommended: 100/500/1000/5000) |
| `--seed` | Integer | `42` | Random seed for reproducibility |
| `--initial_cash` | Float | `100000` | Initial capital (CNY) |
| `--exec_price_model` | String | `next_open` | Execution price model: `same_close_idealized`, `next_open`, `next_close` |
| `--slippage_bps` | Float | `10.0` | Slippage in basis points, default 10 bps (0.1%) |
| `--commission_rate` | Float | `0.00025` | Commission rate, default 0.025% |
| `--commission_min` | Float | `5.0` | Minimum commission, default 5 CNY |
| `--config` | String | `None` | Load configuration from file (for reproduction) |
| `--record_metadata` | Boolean | `True` | Record full metadata (default: enabled) |
| `--log_dir` | String | `./logs` | Structured logging directory |

### Run Modes

#### `first` Mode
- Starts random search from scratch
- Suitable for first run or after changing data

#### `continue` Mode
- Reads the best configuration from the latest run in `output/` directory
- Continues optimization based on it
- Suitable for iterative strategy improvement

### Data File Format

Excel file must contain the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `date` | Trading date | `2021-03-02` |
| `open_qfq` | Adjusted open price | `10.5` |
| `high_qfq` | Adjusted high price | `11.2` |
| `low_qfq` | Adjusted low price | `10.3` |
| `close_qfq` | Adjusted close price | `11.0` |
| `volume` | Trading volume | `1000000` |
| `amount` | Trading amount | `11000000` |
| `turnover_rate` | Turnover rate | `2.5` |

> **Tip**: Data can be obtained from Tushare, Baostock, JoinQuant, etc. Export to Excel format.

---

## ⚙️ Configuration

### Feature Configuration (Auto-Search)

| Parameter | Description | Search Range |
|-----------|-------------|--------------|
| `windows` | Time windows | `[3,5,10,20]`, `[5,10,20,60]`, etc. |
| `use_momentum` | Momentum features | `True` / `False` |
| `use_volatility` | Volatility features | `True` / `False` |
| `use_volume` | Volume features | `True` / `False` |
| `use_candle` | Candlestick pattern features | `True` / `False` |
| `use_turnover` | Turnover rate features | `True` / `False` |
| `vol_metric` | Volatility metric | `std` / `mad` |
| `liq_transform` | Liquidity transformation | `ratio` / `zscore` |

### Model Configuration (Auto-Search)

| Model Type | Search Parameters |
|------------|-------------------|
| **Logistic Regression** | penalty (l1/l2/elasticnet), solver, C, class_weight, l1_ratio |
| **SGD Classifier** | alpha, penalty, class_weight, l1_ratio |
| **XGBoost** | n_estimators, max_depth, learning_rate, subsample, colsample_bytree |

### Trading Configuration (Auto-Search)

| Parameter | Description | Search Range |
|-----------|-------------|--------------|
| `buy_threshold` | Buy probability threshold | `0.50` ~ `0.70` |
| `sell_threshold` | Sell probability threshold | `0.30` ~ `0.50` |
| `confirm_days` | Signal confirmation days | `1` ~ `5` |
| `min_hold_days` | Minimum holding days | `1` ~ `5` |
| `max_hold_days` | Maximum holding days | `10` ~ `60` |
| `take_profit` | Take-profit threshold (optional) | `0.05` ~ `0.20` |
| `stop_loss` | Stop-loss threshold (optional) | `0.05` ~ `0.15` |

---

## 🔬 Realistic Execution

Version 2.0 introduces a **realistic execution model** that accounts for:
- **Execution price models**: `same_close_idealized`, `next_open`, `next_close`
- **Transaction costs**: Commission, transfer fee, stamp tax
- **Slippage**: Fixed basis points (bps) slippage model

### Execution Price Models

| Model | Signal Day | Execution Day | Price Used | Note |
|-------|------------|---------------|------------|------|
| `same_close_idealized` | t | t+1 | Day t close | Idealized (legacy), underestimates overnight gap |
| `next_open` | t | t+1 | Day t+1 open | **Recommended**, more realistic, captures overnight risk |
| `next_close` | t | t+1 | Day t+1 close | Conservative estimate |

### Transaction Costs (A-Share Standard)

| Fee Type | Rate | Direction | Minimum |
|----------|------|-----------|---------|
| **Commission** | 0.025% (2.5‱) | Buy & Sell | 5 CNY |
| **Transfer Fee** | 0.001% (0.1‱) | Buy & Sell | - |
| **Stamp Tax** | 0.05% (5‱) | Sell only | - |

**Calculation Example:**

Buy 100 shares @ 10 CNY:
- Turnover: 100 × 10 = 1,000 CNY
- Commission: max(1,000 × 0.025%, 5) = 5 CNY
- Transfer fee: 1,000 × 0.001% = 0.01 CNY
- **Total cost**: 5.01 CNY

Sell 100 shares @ 10 CNY:
- Turnover: 100 × 10 = 1,000 CNY
- Commission: max(1,000 × 0.025%, 5) = 5 CNY
- Transfer fee: 1,000 × 0.001% = 0.01 CNY
- Stamp tax: 1,000 × 0.05% = 0.5 CNY
- **Total cost**: 5.51 CNY

### Slippage Model

Slippage is applied as a fixed basis points (bps) adjustment:

- **Buy**: Execution price = Base price × (1 + slippage_bps / 10000)
- **Sell**: Execution price = Base price × (1 - slippage_bps / 10000)

**Default**: 10 bps (0.1%)

**Example**:
- Base price: 10.00 CNY
- Slippage: 10 bps
- Buy execution: 10.00 × 1.001 = **10.01 CNY**
- Sell execution: 10.00 × 0.999 = **9.99 CNY**

### Expected Impact

When switching from `same_close_idealized` to `next_open + fee + slippage`:

| Component | Typical Impact |
|-----------|----------------|
| Overnight gap (close→open) | -0.5% ~ -2% per trade |
| Slippage (10 bps) | -0.1% per trade |
| Transaction costs | -0.5% ~ -1% per round trip |
| **Total** | **-1% ~ -3% per trade** |

**High-frequency strategies** (many trades per year) will see a larger cumulative impact.

---

## 📊 Output Files

After running, the `output/` directory will generate the following files:

### run_XXX.xlsx - Backtest Report

**Important: Sheets are listed in order of importance. Check [WalkForwardSummary] first.**

| Sheet Name | Content | Priority |
|------------|---------|----------|
| `WalkForwardSummary` | **Walk-Forward Out-of-Sample Validation Metrics** (Key KPIs) | ⭐⭐⭐⭐⭐ |
| `Trades` | Trade records (buy/sell dates, prices, returns, costs, slippage, etc.) | ⭐⭐⭐⭐ |
| `EquityCurve` | Equity curve (date, equity, position, drawdown, signals, etc.) | ⭐⭐⭐ |
| `FinalFit_InSample` | **In-Sample Result Warning** (For reference only, not out-of-sample performance) | ⚠️ |
| `SearchLog` | Complete random search iteration log (detailed metrics per fold, config params) | ⭐⭐⭐⭐ |
| `Config` | Configuration parameters (features, model, trading params, constraints) | ⭐⭐⭐ |
| `ModelParams` | Model parameters (feature coefficients, scaler params, intercept) | ⭐⭐ |
| `Log` | Run logs and diagnostic information | ⭐⭐ |

### Why is [WalkForwardSummary] Most Important?

- **Out-of-Sample Validation**: Each Walk-Forward fold uses unseen data, reflecting real expected performance
- **No Information Leakage**: Training and validation sets are strictly separated, avoiding look-ahead bias
- **Robustness Assessment**: Multi-fold validation evaluates strategy stability across different market conditions

### ⚠️ Why is [EquityCurve] For Reference Only?

`EquityCurve` shows the **Full-Sample Fit Result** (In-Sample):
- Model is trained and predicted on the same data (information leakage)
- Does not account for overnight gaps, slippage, and fill deviation (unless using realistic execution)
- Results are overly optimistic and **do not represent future live trading performance**

**For real out-of-sample performance, check the metrics in [WalkForwardSummary] and [SearchLog] sheets.**

### run_XXX_config.json - Configuration File

Contains all parameters of the best configuration, used for:
- Reproducing run results
- `continue` mode loading
- Strategy parameter archiving

### run_XXX_metadata.json - Full Metadata (NEW)

Contains complete run metadata:
- Code version (git commit)
- Python version
- All dependency versions
- Data file hash
- Random seed
- Hostname and timestamp

Used for **exact reproduction**.

### Key Metrics Explanation

| Metric | Description |
|--------|-------------|
| `geom_mean_ratio` | Geometric mean return rate (main optimization target) |
| `total_return` | Total return |
| `max_drawdown` | Maximum drawdown |
| `win_rate` | Win rate |
| `profit_factor` | Profit factor |
| `sharpe_ratio` | Sharpe ratio (annualized) |
| `trade_count` | Number of trades |

---

## 📁 Project Structure

```
Ashare_DpointTrader 2.0/
├── main_cli.py              # Main entry point, CLI interface
├── data_loader.py           # Data loading and cleaning
├── feature_dpoint.py        # Feature engineering
├── model_builder.py         # Model construction
├── splitter.py              # Walk-forward data splitting
├── metrics.py               # Evaluation metrics calculation
├── search_engine.py         # Random search engine
├── trainer_optimizer.py     # Trainer API
├── backtester_engine.py     # Backtest execution engine
├── reporter.py              # Report generation
├── persistence.py           # Configuration persistence
├── constants.py             # Global constants
├── config_schema.py         # Configuration schema and validation (engineering)
├── structured_logging.py    # Structured logging configuration (engineering)
├── setup_check.py           # Startup verification script
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project configuration file
├── pytest.ini               # pytest test configuration
├── data/                    # Data directory
│   └── 600698_5Y_daily_qfq_20210302_20260302.xlsx
├── output/                  # Output directory (generated after running)
│   ├── run_001.xlsx
│   ├── run_001_config.json
│   ├── run_001_metadata.json
│   └── ...
├── logs/                    # Log directory (generated after running)
│   └── dpoint_trader_*.log
└── tests/                   # Test directory
    ├── test_backtester.py
    ├── test_features.py
    ├── test_splitter.py
    ├── test_reporter.py
    ├── test_main_cli.py
    └── test_integration.py
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `main_cli.py` | Main entry, CLI parsing, startup checks, metadata recording |
| `data_loader.py` | Excel reading, data cleaning, outlier filtering |
| `feature_dpoint.py` | Build 80+ technical features, generate Dpoint labels |
| `model_builder.py` | Create sklearn/XGBoost model pipelines |
| `splitter.py` | Walk-forward time-series splitting |
| `metrics.py` | Calculate geometric mean return, penalty, trade statistics |
| `search_engine.py` | Random search main loop, hyperparameter sampling |
| `trainer_optimizer.py` | Trainer API, final model training |
| `backtester_engine.py` | Signal generation, trade simulation, equity calculation (with costs & slippage) |
| `reporter.py` | Excel report generation, configuration saving |
| `persistence.py` | Best configuration saving and loading |
| `config_schema.py` | Configuration schema definition and validation (engineering) |
| `structured_logging.py` | Structured logging configuration (engineering) |
| `setup_check.py` | Startup checks (dependencies, data, output dir) |

---

## 🔬 Core Algorithms

### Dpoint Prediction Model

```
Dpoint_t = P(close_{t+1} > close_t | X_t)
```

- **Input**: OHLCV, trading amount, turnover rate, and other features up to day t
- **Output**: Probability of next-day increase (between 0~1)
- **Key**: All features use only data up to day t, no look-ahead bias

### Walk-forward Validation Flow

```
┌─────────────────────────────────────────────────────────┐
│  Fold 1: Train [0:60%] → Validate [60%:80%]             │
│  Fold 2: Train [20%:80%] → Validate [80%:100%]          │
│  ...                                                    │
└─────────────────────────────────────────────────────────┘
```

Each fold evaluates out-of-sample trading performance to ensure no overfitting.

### Random Search Algorithm

```python
# Pseudocode
for iteration in range(runs):
    if explore_mode:
        config = sample_global_space()  # Global exploration
    else:
        config = perturb_best_config()  # Local exploitation

    score = evaluate_walkforward(config)  # Walk-forward evaluation

    if score > best_score + epsilon:
        best_config = config
        best_score = score
```

### Trading Execution Logic

1. **Signal Generation**: `dpoint > buy_threshold` → Buy signal; `dpoint < sell_threshold` → Sell signal
2. **Signal Confirmation**: Trigger only after N consecutive days meeting threshold
3. **Execution Simulation**: Day t signal executes at Day t+1 using Day t's close price (idealized)
4. **Position Management**: Allow closing only after satisfying `min_hold_days`

---

## 🧪 Testing

### Run All Tests

```bash
cd "Ashare_DpointTrader 2.0"
pytest tests/ -v
```

### Run Specific Test File

```bash
# Backtest engine tests
pytest tests/test_backtester.py -v

# Feature engineering tests
pytest tests/test_features.py -v

# Data splitter tests
pytest tests/test_splitter.py -v

# Reporter tests
pytest tests/test_reporter.py -v

# CLI tests
pytest tests/test_main_cli.py -v

# Integration tests (slowest)
pytest tests/test_integration.py -v
```

### Run Specific Test

```bash
# Run a single test function
pytest tests/test_backtester.py::TestTransactionCosts::test_buy_commission_minimum -v
```

### Run with Coverage

```bash
# Install coverage tool
pip install pytest-cov

# Run with coverage report
pytest --cov=. --cov-report=html

# Open coverage report
# On Windows: start htmlcov/index.html
# On Linux/macOS: open htmlcov/index.html
```

### Test Coverage Summary

| Test Module | Coverage |
|-------------|----------|
| `test_backtester.py` | T+1 execution, transaction costs, slippage, position constraints, take-profit/stop-loss |
| `test_features.py` | Label construction, feature families, no look-ahead bias |
| `test_splitter.py` | Walk-forward splitting, no data leakage |
| `test_reporter.py` | Sheet names, config JSON, formula escaping |
| `test_main_cli.py` | Default path, parameter parsing, startup checks |
| `test_integration.py` | End-to-end pipeline test |

---

## 🔧 Engineering Features

### 1. Configuration Schema

Strict configuration schema with type checking and validation:

```python
from config_schema import FullConfig, FeatureConfig, ModelConfig, TradeConfig

config = FullConfig(
    feature_config=FeatureConfig(windows=[3, 5], ...),  # Type-checked
    model_config=ModelConfig(model_type="logreg", ...),  # Validated
    trade_config=TradeConfig(...),
)
errors = config.validate()  # Explicit validation
```

### 2. Structured Logging

Unified logging with JSON output to files and colored console output:

```python
from structured_logging import setup_logger

logger = setup_logger(
    name="dpoint_trader",
    level="INFO",
    log_dir="./logs",
    console_output=True,
    file_output=True,
)

# Log with extra fields
logger.info("Starting training", extra={"runs": 100, "seed": 42})
```

**Log output example:**
```json
{"timestamp": "2026-03-14T10:30:00", "level": "INFO", "message": "Starting training", "runs": 100, "seed": 42}
```

### 3. Full Metadata Recording

Records complete run metadata for reproducibility:
- Code version (git commit)
- Python version
- All dependency versions
- Data file hash
- Random seed
- Hostname and timestamp

**Usage:**
```bash
# Metadata recording enabled by default
python main_cli.py --runs 100

# Reproduce from metadata file
python main_cli.py --config output/run_001_metadata.json
```

### 4. One-Command Reproduction

```bash
# Exact reproduction (same seed, same config)
python main_cli.py --config output/run_042_metadata.json

# Load config but override parameters
python main_cli.py --config output/run_042_config.json --runs 500 --seed 123
```

---

## ❓ FAQ

### Q1: Error "No module named 'xlsxwriter'" at runtime

**A**: xlsxwriter is a required dependency. Install:
```bash
pip install xlsxwriter
```

### Q2: Running too slowly

**A**: Recommendations:
- Reduce `--runs` parameter (e.g., from 1000 to 100)
- Enable parallel computing (enabled by default, uses `n_jobs=6`)
- Install XGBoost and enable GPU acceleration

### Q3: How to use my own data?

**A**:
1. Prepare Excel file with required columns (see [Data File Format](#data-file-format))
2. Run:
   ```bash
   python main_cli.py --data_path "your_data.xlsx"
   ```

### Q4: How to reproduce results?

**A**: Fix random seed and use metadata file:
```bash
python main_cli.py --seed 42 --runs 100
# or
python main_cli.py --config output/run_001_metadata.json
```

### Q5: How to use `continue` mode?

**A**:
```bash
# First run
python main_cli.py --mode first --runs 100

# Continue optimization based on first run results
python main_cli.py --mode continue --runs 100
```

### Q6: What does "IN-SAMPLE" warning in output files mean?

**A**: The final report's equity curve is a full-sample fit result (training set = test set), which has look-ahead bias and is overly optimistic. **For real out-of-sample performance, check the walk-forward validation metrics in the `WalkForwardSummary` sheet**.

### Q7: Data file not found?

**A**:
1. Confirm `data/600698_5Y_daily_qfq_20210302_20260302.xlsx` exists
2. Or use `--data_path` to specify your data file
3. Or set environment variable:
   ```bash
   # Windows
   set ASHARE_DATA_PATH=path/to/your/data.xlsx
   
   # Linux/macOS
   export ASHARE_DATA_PATH=path/to/your/data.xlsx
   ```

### Q8: How to adjust transaction costs and slippage?

**A**:
```bash
# Customize commission and slippage
python main_cli.py --commission_rate 0.0003 --slippage_bps 15
```

---

## ⚠️ Disclaimer

1. **This software is for learning and research purposes only** and does not constitute investment advice or recommendation.

2. **Historical backtest performance does not represent future results**. Quantitative strategies carry risks of overfitting, market changes, execution deviations, etc.

3. **Live trading carries high risk** and may result in loss of principal. Please use with caution after fully understanding the strategy logic and risks.

4. **The author is not responsible for any direct or indirect losses**. All consequences arising from the use of this software are borne by the user.

5. **Do not use this software for illegal purposes**. Comply with local laws, regulations, and securities regulatory requirements.

---

## 📧 Contact

For questions or suggestions, please contact via GitHub Issues.

---

## 📝 Version History

### Version 2.0 (Current)

**New Features:**
- ✅ Realistic execution model (transaction costs, slippage, multiple execution price models)
- ✅ In-sample / out-of-sample results separation (WalkForwardSummary sheet)
- ✅ Configuration schema (type checking, validation)
- ✅ Structured logging (JSON format, performance tracking)
- ✅ Full metadata recording (for reproduction)
- ✅ Test suite (100+ test cases)
- ✅ Startup checks (dependencies, data, output directory)

**Improvements:**
- ✅ Default data path changed to relative path
- ✅ Completed requirements.txt and pyproject.toml
- ✅ Documentation fully updated and aligned with actual code
