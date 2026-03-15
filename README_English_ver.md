# A-Share Dpoint Trader 2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A-Share Single-Stock ML Trading Strategy System**

A quantitative trading backtesting framework based on machine learning prediction of next-day price movement (Dpoint), combined with walk-forward validation and random search hyperparameter optimization.

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration Schema](#-configuration-schema)
- [CLI Parameters](#-cli-parameters)
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
git clone https://github.com/Ashare_DpointTrader-2.0/Ashare_DpointTrader-2.0.git
cd Ashare_DpointTrader-2.0

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

### Windows Virtual Environment (Optional)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# PowerShell:
.\venv\Scripts\Activate.ps1
# CMD:
.\venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Run
python main_cli.py
```

### Opening Excel Report

1. **First check** `WalkForwardSummary` sheet (Out-of-sample metrics - PRIMARY KPIs)
2. **Check** `ExecutionAssumptions` sheet (Execution assumptions)
3. **Then check** `Trades` sheet (Trade records)
4. `EquityCurve` sheet is for reference only (In-sample results)

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
- **Configuration Schema**: Strict configuration structure using Pydantic, supports validation and serialization

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
git clone https://github.com/Ashare_DpointTrader-2.0/Ashare_DpointTrader-2.0.git
cd Ashare_DpointTrader-2.0
```

### 2. Create Virtual Environment (Recommended)

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

# Optional: Install XGBoost (GPU acceleration)
pip install xgboost
```

### 4. Verify Installation

```bash
# Quick verification (check dependencies and data files)
python main_cli.py --help

# Or run startup checks (without backtesting)
python setup_check.py
```

---

## 🎯 Usage

### Default Run (Using Sample Data)

Project includes sample data file `data/600698_5Y_daily_qfq_20210302_20260302.xlsx`, can run directly:

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

# Realistic execution mode - Use open price + transaction costs + slippage (recommended)
python main_cli.py --mode first --runs 100 --exec_price_model next_open --slippage_bps 10

# Reproduce previous run
python main_cli.py --config output/run_001_metadata.json

# Disable metadata recording
python main_cli.py --no-record-metadata
```

---

## 🧬 Configuration Schema

Configuration is managed through `FullConfig`, containing the following sub-configurations:

### 1. FeatureConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `windows` | Integer list | `[3, 5, 10, 20]` | Rolling window sizes |
| `use_momentum` | Boolean | `True` | Use momentum features |
| `use_volatility` | Boolean | `True` | Use volatility features |
| `use_volume` | Boolean | `True` | Use volume features |
| `use_candle` | Boolean | `True` | Use candlestick pattern features |
| `use_turnover` | Boolean | `True` | Use turnover rate features |
| `vol_metric` | `std`/`mad` | `std` | Volatility calculation method |
| `liq_transform` | `ratio`/`zscore` | `ratio` | Liquidity transformation method |

### 2. ModelConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_type` | `logreg`/`sgd`/`xgb` | `logreg` | Model type |
| `C` | Float | `1.0` | Logistic regression regularization strength |
| `penalty` | `l1`/`l2`/`elasticnet` | `l2` | Regularization type |
| `solver` | String | `lbfgs` | Optimization algorithm |
| `alpha` | Float | `1e-4` | SGD regularization strength |
| `n_estimators` | Integer | `100` | XGBoost tree count |
| `max_depth` | Integer | `3` | XGBoost tree max depth |
| `learning_rate` | Float | `0.1` | XGBoost learning rate |

### 3. TradeConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `initial_cash` | Float | `100000` | Initial capital (CNY) |
| `buy_threshold` | Float | `0.55` | Buy Dpoint threshold |
| `sell_threshold` | Float | `0.45` | Sell Dpoint threshold |
| `confirm_days` | Integer | `2` | Signal confirmation days |
| `min_hold_days` | Integer | `1` | Minimum holding days |
| `max_hold_days` | Integer | `20` | Maximum holding days |
| `take_profit` | Float | `None` | Take-profit threshold (EOD-based) |
| `stop_loss` | Float | `None` | Stop-loss threshold (EOD-based) |
| `exec_price_model` | String | `next_open` | Execution price model |
| `slippage_bps` | Float | `10.0` | Slippage (basis points) |
| `commission_rate` | Float | `0.00025` | Commission rate |
| `commission_min` | Float | `5.0` | Minimum commission |

### 4. SearchConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `runs` | Integer | `100` | Random search iterations |
| `epsilon` | Float | `0.01` | Minimum improvement threshold |
| `exploit_ratio` | Float | `0.7` | Local exploitation ratio (70% exploit, 30% explore) |
| `top_k` | Integer | `10` | Top-K pool size |
| `max_features` | Integer | `80` | Maximum features |
| `n_jobs` | Integer | `-1` | Parallel processes (-1=all cores, 1=single thread) |

---

## 🔧 CLI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--mode` | `first` / `continue` | `first` | Run mode |
| `--data_path` | String | Built-in default | Excel data file path |
| `--output_dir` | String | `./output` | Output directory |
| `--runs` | Integer | `100` | Random search iterations |
| `--seed` | Integer | `42` | Random seed (base seed) |
| `--initial_cash` | Float | `100000` | Initial capital (CNY) |
| `--exec_price_model` | String | `next_open` | Execution price model |
| `--slippage_bps` | Float | `10.0` | Slippage (basis points) |
| `--commission_rate` | Float | `0.00025` | Commission rate |
| `--commission_min` | Float | `5.0` | Minimum commission |
| `--config` | String | `None` | Load from config file (for reproduction) |
| `--record-metadata` | Boolean | `True` | Record metadata (use `--no-record-metadata` to disable) |
| `--log_dir` | String | `./logs` | Log directory |

### Run Modes

#### `first` Mode
- Starts random search from scratch
- Suitable for first run or after changing data

#### `continue` Mode
- Reads the best configuration from the latest run in `output/` directory
- Continues optimization based on it
- Suitable for iterative strategy improvement

### Random Seed Semantics

The system uses three seeds to ensure reproducibility:

| Seed Name | Description | Calculation |
|-----------|-------------|-------------|
| **Base Seed** | CLI input base seed (`--seed`) | User specified, default 42 |
| **Search Seed** | Actual seed for random search | `base_seed + latest_run_id` |
| **Final Train Seed** | Seed for final full-sample model training | Always equals `base_seed` |

**Reproduction Strategy**:
- **First run (first mode)**: All three seeds are the same
- **Continue run (continue mode)**: Search Seed = Base Seed + previous run_id
- **Exact reproduction**: Use `--config run_XXX_metadata.json` with the same `--seed` value

---

## 📊 Data Files

### File Format

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

> 💡 **Tip**: Data can be obtained from Tushare, Baostock, JoinQuant, etc. Export to Excel format.

### Missing Value Handling

For non-core columns (`volume`, `amount`, `turnover_rate`), the system provides the following handling strategies:

| Strategy | Description | Marker Columns |
|----------|-------------|----------------|
| `zero` (default) | Fill with 0, add `_was_missing` marker columns | `volume_was_missing`, etc. |
| `ffill` | Forward fill from previous day | `volume_was_missing`, etc. |
| `drop` | Drop rows with missing values | - |
| `keep_nan` | Keep as NaN, handled by feature engineering | `volume_was_missing`, etc. |

**Default Strategy Notes**:
- When filling with 0, boolean marker columns (`volume_was_missing`, `amount_was_missing`, `turnover_rate_was_missing`) are added
- These markers distinguish "true 0 volume" from "missing data filled with 0"

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
| **Commission** | 0.025% (2.5‱) | Both sides | 5 CNY |
| **Transfer Fee** | 0.001% (0.1‱) | Both sides | - |
| **Stamp Tax** | 0.05% (5‱) | Sell only | - |

### Slippage Model

Slippage adjusts by fixed basis points (bps):
- **Buy**: Execution Price = Base Price × (1 + slippage_bps / 10000)
- **Sell**: Execution Price = Base Price × (1 - slippage_bps / 10000)

**Default**: 10 bps (0.1%)

### Take-Profit/Stop-Loss Trigger Mechanism

The system uses **EOD-based (End-of-Day)** mechanism:
- After market close, calculates PnL ratio using closing price
- If `pnl_ratio >= take_profit`, triggers take-profit, executes sell next day
- If `pnl_ratio <= -stop_loss`, triggers stop-loss, executes sell next day
- Does **NOT** use intraday high/low for trigger

---

## 📁 Output Files

After running, the `output/` directory will contain:

### run_XXX.xlsx - Backtest Report

**Important: Sheets are sorted by priority. First check [WalkForwardSummary].**

| Sheet Name | Content | Priority |
|------------|---------|----------|
| `WalkForwardSummary` | **Walk-Forward Out-of-Sample Validation Metrics** (Key KPIs) | ⭐⭐⭐⭐⭐ |
| `ExecutionAssumptions` | **Execution Assumptions** (price model, slippage, costs, rules) | ⭐⭐⭐⭐⭐ |
| `Trades` | Trade records (buy/sell dates, prices, returns, costs, slippage, etc.) | ⭐⭐⭐⭐ |
| `EquityCurve` | Equity curve (date, equity, position, drawdown, signals, etc.) | ⭐⭐⭐ |
| `FinalFit_InSample` | **In-Sample Result Warning** (For reference only, not out-of-sample performance) | ⚠️ |
| `SearchLog` | Complete random search iteration log (detailed metrics per fold, config params) | ⭐⭐⭐⭐ |
| `Config` | Configuration parameters (features, model, trading params, constraints) | ⭐⭐⭐ |
| `ModelParams` | Model parameters (feature coefficients, scaler params, intercept) | ⭐⭐ |
| `Log` | Run logs and diagnostic information | ⭐⭐ |

### run_XXX_config.json - Configuration File

JSON file containing the best configuration and feature metadata, can be used for `--config` parameter in subsequent runs.

### run_XXX_metadata.json - Metadata File

Contains complete run metadata for exact reproduction:
- Code version (git commit)
- Python version
- All dependency versions
- Data file hash
- Random seed (base seed)
- Hostname and timestamp

---

## 📂 Project Structure

```
Ashare_DpointTrader-2.0/
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
├── config_schema.py         # Configuration schema and validation
├── structured_logging.py    # Structured logging configuration
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
    ├── test_integration.py
    └── test_e2e_config.py
```

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
│  Walk-forward Splits (n_folds=4, train_start_ratio=0.5) │
│                                                         │
│  Fold 1: Train [0%~50%]  →  Validate [50%~62.5%]        │
│  Fold 2: Train [0%~62%]  →  Validate [62.5%~75%]        │
│  Fold 3: Train [0%~75%]  →  Validate [75%~87.5%]        │
│  Fold 4: Train [0%~87%]  →  Validate [87.5%~100%]       │
│                                                         │
│  Features: Training set expands cumulatively            │
│            (expanding window)                           │
│            Validation sets do not overlap               │
└─────────────────────────────────────────────────────────┘
```

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
2. **Signal Confirmation**: Trigger only after N consecutive days meeting threshold (`confirm_days`)
3. **Execution Simulation**: Day t signal executes on Day t+1 (using configured execution price model)
   - `same_close_idealized`: Uses Day t's close price (idealized, legacy behavior)
   - `next_open`: Uses Day t+1's open price (recommended, more realistic)
   - `next_close`: Uses Day t+1's close price (conservative)
4. **Position Management**: Allow closing only after satisfying `min_hold_days`

---

## 🧪 Testing

### Run All Tests

```bash
cd Ashare_DpointTrader-2.0
pytest tests/ -v
```

### Run Specific Tests

```bash
# Backtester engine tests
pytest tests/test_backtester.py -v

# Feature engineering tests
pytest tests/test_features.py -v

# End-to-end configuration tests
pytest tests/test_e2e_config.py -v

# Integration tests (slowest)
pytest tests/test_integration.py -v
```

### Generate Coverage Report

```bash
# Install coverage tool
pip install pytest-cov

# Run and generate coverage report
pytest --cov=. --cov-report=html
```

---

## ⚙️ Engineering Features

- **Configuration Schema**: Strict configuration structure using Pydantic, supports validation and serialization
- **Configuration Persistence**: Best configuration auto-saved to JSON, supports reproduction
- **Structured Logging**: JSON format logs with performance tracking and diagnostics
- **Startup Checks**: Automatically checks dependencies, data files, and output directory before running
- **Version Tracking**: Records code version, Python version, dependency versions, git commit
- **Data Hash**: Uses SHA-256 to calculate data file hash, ensures data consistency
- **Test Coverage**: End-to-end tests protect core functionality, prevent regression from refactoring

---

## ❓ FAQ

### Q: Why is WalkForwardSummary most important?

**A**: Each Walk-Forward fold uses unseen data, reflecting real expected performance. Training and validation sets are strictly separated, avoiding look-ahead bias.

### Q: Why is EquityCurve for reference only?

**A**: `EquityCurve` shows the **Full-Sample Fit result** (In-Sample):
- Model is trained and predicted on the same data (information leakage)
- Does not account for overnight gaps, slippage, and fill deviation (unless using realistic execution model)
- Results are overly optimistic, **does not represent future live trading performance**

**For real out-of-sample performance, see metrics in [WalkForwardSummary] and [SearchLog].**

### Q: How to reproduce a previous run?

**A**: Use `--config` parameter to load the previous metadata file:
```bash
python main_cli.py --config output/run_001_metadata.json
```

### Q: What's the difference between continue mode and first mode?

**A**: 
- `first` mode: Starts random search from scratch
- `continue` mode: Reads the best configuration from the latest run in `output/` directory and continues optimization

---

## ⚠️ Disclaimer

This software is for academic research and educational purposes only, and does not constitute investment advice. Any trading behavior conducted using this software is at the user's own risk. The author is not responsible for any direct or indirect losses.

Financial market trading involves significant risks, including but not limited to:
- Market volatility risk
- Liquidity risk
- Technical failure risk
- Model failure risk

Please use with caution after fully understanding the risks and evaluating your own risk tolerance.
