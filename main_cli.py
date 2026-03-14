# main_cli.py
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from data_loader import load_stock_excel
from trainer_optimizer import random_search_train, train_final_model_and_dpoint
from backtester_engine import backtest_from_dpoint
from reporter import save_run_outputs, find_latest_run
from config_schema import (
    FullConfig, FeatureConfig, ModelConfig, TradeConfig,
    RunMetadata, compute_data_hash, get_code_version,
    get_dependency_versions, get_python_version, get_git_commit, get_hostname
)
from structured_logging import setup_logger, log_context, info_extra, error_extra


# =========================================================
# Startup Checks (确保开箱即用)
# =========================================================
def check_dependencies() -> List[str]:
    """
    检查必需依赖是否已安装，返回缺失的依赖列表。
    """
    missing = []
    
    # 核心依赖
    core_packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("openpyxl", "openpyxl"),
        ("xlsxwriter", "xlsxwriter"),
        ("joblib", "joblib"),
    ]
    
    for import_name, pkg_name in core_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)
    
    return missing


def check_data_file(data_path: str) -> tuple[bool, str]:
    """
    检查数据文件是否存在。
    返回 (是否存在，错误信息)。
    """
    if not os.path.exists(data_path):
        return False, f"Data file not found: {data_path}"
    if not os.path.isfile(data_path):
        return False, f"Path is not a file: {data_path}"
    if not data_path.endswith(('.xlsx', '.xls')):
        return False, f"File does not appear to be an Excel file: {data_path}"
    return True, ""


def check_output_dir(output_dir: str) -> tuple[bool, str]:
    """
    检查输出目录是否可写。
    返回 (是否可写，错误信息)。
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        # 测试写入权限
        test_file = os.path.join(output_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True, ""
    except PermissionError:
        return False, f"Permission denied: cannot write to {output_dir}"
    except Exception as e:
        return False, f"Cannot access output directory: {e}"


def run_startup_checks(data_path: str, output_dir: str) -> bool:
    """
    运行所有启动前检查。
    返回是否通过所有检查。
    """
    print("=" * 60)
    print("A-Share Dpoint Trader 2.0 - Startup Checks")
    print("=" * 60)
    
    all_passed = True
    
    # 1. 检查依赖
    print("\n[1/3] Checking dependencies...")
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"  ❌ Missing dependencies: {', '.join(missing_deps)}")
        print(f"  💡 Install with: pip install {' '.join(missing_deps)}")
        all_passed = False
    else:
        print("  ✅ All required dependencies installed")
    
    # 2. 检查数据文件
    print(f"\n[2/3] Checking data file...")
    print(f"  Path: {data_path}")
    exists, err_msg = check_data_file(data_path)
    if not exists:
        print(f"  ❌ {err_msg}")
        print(f"  💡 Use --data_path to specify your data file")
        print(f"  💡 Or set ASHARE_DATA_PATH environment variable")
        all_passed = False
    else:
        print(f"  ✅ Data file found")
    
    # 3. 检查输出目录
    print(f"\n[3/3] Checking output directory...")
    print(f"  Path: {output_dir}")
    writable, err_msg = check_output_dir(output_dir)
    if not writable:
        print(f"  ❌ {err_msg}")
        all_passed = False
    else:
        print(f"  ✅ Output directory is writable")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All startup checks passed")
    else:
        print("❌ Some startup checks failed. Please fix the issues above.")
    print("=" * 60)
    
    return all_passed


# =========================================================
# Data Path Configuration
# =========================================================
def get_default_data_path() -> str:
    """
    Get default data file path. Uses relative path based on script location.
    Falls back to environment variable ASHARE_DATA_PATH if set.
    """
    # Check environment variable first
    env_path = os.environ.get("ASHARE_DATA_PATH")
    if env_path:
        return env_path

    # Use relative path based on script location
    script_dir = Path(__file__).resolve().parent
    default_relative_path = script_dir / "data" / "600698_5Y_daily_qfq_20210302_20260302.xlsx"
    return str(default_relative_path)


DEFAULT_DATA_PATH = get_default_data_path()

def _get_latest_run_id(output_dir: str) -> int:
    latest = find_latest_run(output_dir)
    if latest is None:
        return 0
    run_id, _, _ = latest
    return int(run_id)

def _load_previous_best(output_dir: str) -> Optional[Dict[str, object]]:
    latest = find_latest_run(output_dir)
    if latest is None:
        return None
    _, cfg_path, _ = latest
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        return blob.get("best_config")
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="A-share single-stock ML Dpoint trader (2.0).")
    parser.add_argument("--mode", choices=["first", "continue"], default="first", help="Run mode.")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to Excel data file.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for runs.")
    parser.add_argument("--runs", type=int, default=100, help="Random search iterations (100/1000/5000...).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--initial_cash", type=float, default=100000.0, help="Initial cash for simulation.")

    # Realistic execution parameters
    parser.add_argument(
        "--exec_price_model",
        type=str,
        default="next_open",
        choices=["same_close_idealized", "next_open", "next_close"],
        help="Execution price model. Default: next_open (realistic)."
    )
    parser.add_argument(
        "--slippage_bps",
        type=float,
        default=10.0,
        help="Slippage in basis points. Default: 10 bps (0.1%)."
    )
    parser.add_argument(
        "--commission_rate",
        type=float,
        default=0.00025,
        help="Commission rate. Default: 0.00025 (0.025%)."
    )
    parser.add_argument(
        "--commission_min",
        type=float,
        default=5.0,
        help="Minimum commission. Default: 5 CNY."
    )

    # Reproducibility and engineering
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file for reproduction (from previous run)."
    )
    parser.add_argument(
        "--record_metadata",
        action="store_true",
        default=True,
        help="Record full run metadata for reproducibility (default: True)."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory for structured log files."
    )
    args = parser.parse_args()

    # =========================================================
    # Setup Structured Logging
    # =========================================================
    logger = setup_logger(
        name="dpoint_trader",
        level="INFO",
        log_dir=args.log_dir,
        console_output=True,
        file_output=True,
    )

    # =========================================================
    # Load Config from File (if specified)
    # =========================================================
    if args.config:
        # Reproduction mode: load config from JSON
        logger.info(f"Loading config from: {args.config}")
        try:
            full_config = FullConfig.from_json_file(args.config)
            # Override with CLI arguments if provided
            if args.runs != 100:
                logger.info(f"Overriding runs from config: {args.runs}")
            if args.seed != 42:
                logger.info(f"Overriding seed from config: {args.seed}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
    else:
        # Build config from CLI arguments
        full_config = FullConfig(
            feature_config=FeatureConfig(),
            model_config=ModelConfig(),
            trade_config=TradeConfig(
                initial_cash=args.initial_cash,
                buy_threshold=0.55,
                sell_threshold=0.45,
                confirm_days=2,
                min_hold_days=1,
                exec_price_model=args.exec_price_model,
                slippage_bps=args.slippage_bps,
                commission_rate=args.commission_rate,
                commission_min=args.commission_min,
            ),
        )
        logger.info("Using default/CLI config")

    # =========================================================
    # Run Startup Checks (before any processing)
    # =========================================================
    if not run_startup_checks(args.data_path, args.output_dir):
        logger.error("Startup checks failed")
        sys.exit(1)

    logger.info("Startup checks passed")
    df_clean, data_report = load_stock_excel(args.data_path)
    print(f"[INFO] Loaded clean data rows: {len(df_clean)}")
    if len(df_clean) == 0:
        raise ValueError("DataLoader produced 0 rows after cleaning. Check Excel columns and date parsing.")

    base_best_config = None
    if args.mode == "continue":
        base_best_config = _load_previous_best(args.output_dir)
        if base_best_config is None:
            print("[WARN] Continue mode but no previous config found. Falling back to first mode behavior.")
        else:
            print("[INFO] Loaded previous best_config as incumbent.")

    trade_params = {
        "initial_cash": float(args.initial_cash),
        "buy_threshold": 0.55,
        "sell_threshold": 0.45,
        "confirm_days": 2,
        "min_hold_days": 1,
    }

    # compute effective seed based on mode and latest run id
    latest_run_id = _get_latest_run_id(args.output_dir) if args.mode == "continue" else 0
    seed_effective = int(args.seed) + int(latest_run_id)

    train_res = random_search_train(
        df_clean=df_clean,
        runs=int(args.runs),
        seed=int(seed_effective),
        base_best_config=base_best_config,  # 保留也行，trainer会优先best_so_far
        output_dir=str(args.output_dir),
        epsilon=0.01,
        exploit_ratio=0.7,
        top_k=10,
        trade_params=trade_params,
        max_features=80,
        n_jobs=6,
        exec_price_model=args.exec_price_model,
        slippage_bps=args.slippage_bps,
    )

    best_config = train_res.best_config
    print(f"[INFO] Best validation metric (geom mean ratio): {train_res.best_val_metric:.6f}")
    print(f"[INFO] Best validation equity proxy (mean): {train_res.best_val_final_equity_proxy:.2f}")

    dpoint, artifacts = train_final_model_and_dpoint(df_clean, best_config, seed=int(args.seed))

    tc = best_config["trade_config"]  # 提取一次，避免重复索引
    bt = backtest_from_dpoint(
        df=df_clean,
        dpoint=dpoint,
        initial_cash=float(tc["initial_cash"]),
        buy_threshold=float(tc["buy_threshold"]),
        sell_threshold=float(tc["sell_threshold"]),
        confirm_days=int(tc["confirm_days"]),
        min_hold_days=int(tc["min_hold_days"]),
        max_hold_days=int(tc.get("max_hold_days", 20)),    # ✅ 补全
        take_profit=tc.get("take_profit", None),            # ✅ 补全
        stop_loss=tc.get("stop_loss", None),                # ✅ 补全
        exec_price_model=args.exec_price_model,
        slippage_bps=args.slippage_bps,
        commission_rate=args.commission_rate,
        commission_min=args.commission_min,
    )

    final_equity = float(bt.equity_curve["total_equity"].iloc[-1]) if not bt.equity_curve.empty else float(args.initial_cash)
    print(f"[INFO] Full-sample final equity: {final_equity:.2f}")
    print(f"[INFO] Trades executed: {len(bt.trades)}")

    log_notes: List[str] = []
    log_notes.append("=== DataLoader Report ===")
    log_notes.append(f"Data path: {args.data_path}")
    log_notes.append(f"Sheet used: {data_report.sheet_used}")
    log_notes.append(f"Rows raw: {data_report.rows_raw}")
    log_notes.append(f"Rows after dropna core: {data_report.rows_after_dropna_core}")
    log_notes.append(f"Rows after filters: {data_report.rows_after_filters}")
    log_notes.append(f"Duplicate dates: {data_report.duplicate_dates}")
    log_notes.append(f"Bad OHLC rows dropped: {data_report.bad_ohlc_rows}")
    log_notes.extend(data_report.notes)

    log_notes.append("")
    log_notes.append("=== Training Summary / Improvement Confirmation ===")
    log_notes.append(f"Mode: {args.mode}")
    log_notes.append(f"Runs (search iterations): {args.runs}")
    log_notes.append(f"Base seed (CLI): {args.seed}")
    log_notes.append(f"Effective seed (base + latest_run_id): {seed_effective} (latest_run_id={latest_run_id})")
    log_notes.append(f"Best validation metric (geom-mean ratio): {train_res.best_val_metric:.6f}")
    log_notes.append(f"Best validation equity proxy (mean): {train_res.best_val_final_equity_proxy:.2f}")
    log_notes.append(f"Global best metric prev: {train_res.global_best_metric_prev:.6f}")
    log_notes.append(f"Candidate best metric this run: {train_res.candidate_best_metric:.6f}")
    log_notes.append(f"Global best metric new: {train_res.global_best_metric_new:.6f}")
    log_notes.append(f"Epsilon (min improvement): {train_res.epsilon:.6f}")
    log_notes.append(f"Global best updated: {train_res.global_best_updated}")
    log_notes.append(f"Not-updated reason: {train_res.not_updated_reason}")
    log_notes.append(f"Best-so-far file: {train_res.best_so_far_path}")
    log_notes.append(f"Best pool file: {train_res.best_pool_path}")
    log_notes.extend(train_res.training_notes)

    log_notes.append("")
    log_notes.append("=== Backtest Notes ===")
    log_notes.append(
        "⚠️  WARNING: This equity curve is an IN-SAMPLE result (model trained & predicted on same data). "
        "It overstates real performance. See SearchLog for out-of-sample walk-forward metrics."
    )
    log_notes.append(
        "⚠️  警告：此净值曲线为全样本内拟合结果，模型在训练集上预测，存在前向偏差，数值偏乐观。"
        "真实样本外表现请查看 SearchLog sheet 中的 walk-forward 验证指标。"
    )
    log_notes.append(f"Full-sample final equity: {final_equity:.2f}")
    log_notes.append(f"Trades executed: {len(bt.trades)}")
    log_notes.extend(bt.notes)

    excel_path, config_path, run_id = save_run_outputs(
        output_dir=str(args.output_dir),
        df_clean=df_clean,
        log_notes=log_notes,
        trades=bt.trades,
        equity_curve=bt.equity_curve,
        config=best_config,
        feature_meta=artifacts["feature_meta"],
        search_log=train_res.search_log,
        model_params=artifacts.get("model_params"),
    )

    # =========================================================
    # Record Full Run Metadata (for reproducibility)
    # =========================================================
    if args.record_metadata:
        try:
            data_hash = compute_data_hash(df_clean)
            metadata = RunMetadata(
                run_id=run_id,
                created_at=datetime.now().isoformat(timespec="seconds"),
                code_version=get_code_version(),
                python_version=get_python_version(),
                dependency_versions=get_dependency_versions(),
                data_hash=data_hash,
                data_path=args.data_path,
                random_seed=args.seed,
                config=FullConfig.from_dict(best_config),
                git_commit=get_git_commit(),
                hostname=get_hostname(),
                notes=[
                    f"Mode: {args.mode}",
                    f"Runs: {args.runs}",
                    f"Effective seed: {seed_effective}",
                ],
            )
            
            # Save metadata JSON
            metadata_path = os.path.join(str(args.output_dir), f"run_{run_id:03d}_metadata.json")
            metadata.save_json(metadata_path)
            
            logger.info(f"Saved metadata to: {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    print(f"[DONE] Saved run {run_id:03d}")
    print(f"  - Excel : {os.path.abspath(excel_path)}")
    print(f"  - Config: {os.path.abspath(config_path)}")
    if args.record_metadata:
        print(f"  - Metadata: {os.path.abspath(metadata_path)}")
    
    logger.info("Run completed successfully")


if __name__ == "__main__":
    main()
