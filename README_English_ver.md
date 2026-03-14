# A-Share Dpoint Trader 2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A-Share Single-Stock Machine Learning Trading Strategy System**

A quantitative trading backtesting framework based on machine learning to predict next-day price movement probability (Dpoint), combined with walk-forward validation and random search hyperparameter optimization.

---

## Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Output Files](#-output-files)
- [Project Structure](#-project-structure)
- [Core Algorithms](#-core-algorithms)
- [FAQ](#-faq)
- [Disclaimer](#-disclaimer)

---

## Features

### Core Capabilities

- **Dpoint Prediction Model**: Predicts the probability of next-day closing price increase `P(close_{t+1} > close_t | X_t)`
- **Walk-forward Validation**: Time-series cross-validation to avoid look-ahead bias and ensure out-of-sample evaluation
- **Random Search Optimization**: Automatic search for optimal feature combinations, model parameters, and trading strategy parameters
- **Multi-Model Support**: Logistic Regression, SGDClassifier, XGBoost (with GPU acceleration)
- **A-Share Trading Constraints**: Supports T+1, minimum 100-share units, long-only, and other A-share rules

### Technical Features

- **Feature Engineering**: 80+ features across 5 families - momentum, volatility, volume, turnover rate, candlestick patterns
- **Early Stopping Pruning**: Automatically eliminates poor configurations to accelerate search
- **Parallel Search**: Utilizes multi-core CPU for parallel candidate evaluation
- **CUDA Acceleration**: Auto-detects and enables XGBoost GPU acceleration
- **Continuous Learning**: Supports `continue` mode to optimize based on historical best configurations

---

## Requirements

### System Requirements

- **OS**: Windows / Linux / macOS
- **Python**: 3.8 or higher
- **Memory**: 8GB or more recommended

### Python Dependencies

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
openpyxl>=3.0.0        # Excel read/write
xgboost>=1.5.0         # Optional, for XGBoost model
joblib>=1.1.0          # Parallel computing
```

---

## Installation

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

```bash
pip install pandas numpy scikit-learn openpyxl joblib

# Optional: Install XGBoost (for GPU acceleration)
pip install xgboost

# For GPU acceleration, install CUDA version of XGBoost
pip install xgboost-cu112  # Select according to your CUDA version
```

### 4. Verify Installation

```bash
python -c "import pandas, sklearn, xgboost; print('OK')"
```

---

## Quick Start

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
```

---

## Usage

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | `first` / `continue` | `first` | Run mode. `first` starts from scratch, `continue` continues based on historical best |
| `--data_path` | String | Built-in default path | Excel data file path |
| `--output_dir` | String | `./output` | Output directory |
| `--runs` | Integer | `100` | Random search iterations (recommended: 100/500/1000/5000) |
| `--seed` | Integer | `42` | Random seed for reproducibility |
| `--initial_cash` | Float | `100000` | Initial capital (CNY) |

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

## Configuration

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

## Output Files

After running, the `output/` directory will generate the following files:

### run_XXX.xlsx - Backtest Report

| Sheet Name | Content |
|------------|---------|
| `EquityCurve` | Equity curve (date, equity, position, etc.) |
| `Trades` | Trade records (buy/sell dates, prices, returns, etc.) |
| `SearchLog` | Walk-forward validation log (out-of-sample metrics per fold) |
| `Config` | Configuration parameters |
| `Log` | Run logs and diagnostic information |
| `ModelParams` | Model parameters (feature coefficients, scaler parameters) |

### run_XXX_config.json - Configuration File

Contains all parameters of the best configuration, used for:
- Reproducing run results
- `continue` mode loading
- Strategy parameter archiving

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

## Project Structure

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
├── data/                    # Data directory
│   └── 600698_5Y_daily_qfq_20210302_20260302.xlsx
└── output/                  # Output directory (generated after running)
    ├── run_001.xlsx
    ├── run_001_config.json
    └── ...
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `data_loader.py` | Excel reading, data cleaning, outlier filtering |
| `feature_dpoint.py` | Build 80+ technical features, generate Dpoint labels |
| `model_builder.py` | Create sklearn/XGBoost model pipelines |
| `splitter.py` | Walk-forward time-series splitting |
| `metrics.py` | Calculate geometric mean return, penalty, trade statistics |
| `search_engine.py` | Random search main loop, hyperparameter sampling |
| `backtester_engine.py` | Signal generation, trade simulation, equity calculation |
| `reporter.py` | Excel report generation, configuration saving |
| `persistence.py` | Best configuration saving and loading |

---

## Core Algorithms

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

## FAQ

### Q1: Error "No module named 'xgboost'" at runtime

**A**: XGBoost is an optional dependency. If not using XGBoost model, you can ignore this error; if needed:
```bash
pip install xgboost
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

**A**: Fix random seed:
```bash
python main_cli.py --seed 42 --runs 100
```
Use the same seed, same data, and same code version for reproducibility.

### Q5: How to use `continue` mode?

**A**:
```bash
# First run
python main_cli.py --mode first --runs 100

# Continue optimization based on first run results
python main_cli.py --mode continue --runs 100
```

### Q6: What does "IN-SAMPLE" warning in output files mean?

**A**: The final report's equity curve is a full-sample fit result (training set = test set), which has look-ahead bias and is overly optimistic. **For real out-of-sample performance, check the walk-forward validation metrics in the `SearchLog` sheet**.

---

## Disclaimer

1. **This software is for learning and research purposes only** and does not constitute investment advice or recommendation.

2. **Historical backtest performance does not represent future results**. Quantitative strategies carry risks of overfitting, market changes, execution deviations, etc.

3. **Live trading carries high risk** and may result in loss of principal. Please use with caution after fully understanding the strategy logic and risks.

4. **The author is not responsible for any direct or indirect losses**. All consequences arising from the use of this software are borne by the user.

5. **Do not use this software for illegal purposes**. Comply with local laws, regulations, and securities regulatory requirements.

---

## Contact

For questions or suggestions, please contact via GitHub Issues.
