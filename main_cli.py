# main_cli.py
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Any

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from data_loader import load_stock_excel
from trainer_optimizer import random_search_train, train_final_model_and_dpoint
from backtester_engine import backtest_from_dpoint
from reporter import save_run_outputs, find_latest_run
from config_schema import (
    FullConfig, FeatureConfig, ModelConfig, TradeConfig, SearchConfig,
    RunMetadata, compute_data_hash, get_code_version,
    get_dependency_versions, get_python_version, get_git_commit, get_hostname
)
from structured_logging import setup_logger, log_context, info_extra, error_extra


# =========================================================
# Config Resolution (统一配置入口)
# =========================================================
def resolve_effective_config(args: argparse.Namespace) -> FullConfig:
    """
    统一配置解析入口。
    
    规则：
        1. 有 --config：先从文件加载 FullConfig
        2. 再把明确传入的 CLI override 覆盖到这个 config 上
        3. 没有 --config：再从 CLI/default 生成 FullConfig
    
    后续所有地方只吃 effective_config，不要再散落使用 args.xxx 和硬编码值。
    """
    # 默认 CLI 参数（用于检测用户是否明确传入）
    DEFAULT_INITIAL_CASH = 100000.0
    DEFAULT_EXEC_PRICE_MODEL = "next_open"
    DEFAULT_SLIPPAGE_BPS = 10.0
    DEFAULT_COMMISSION_RATE = 0.00025
    DEFAULT_COMMISSION_MIN = 5.0

    if args.config:
        # 从文件加载配置
        full_config = FullConfig.from_json_file(args.config)

        # 应用 CLI 覆盖（只有当 CLI 参数不是默认值时才覆盖）
        # 注意：runs 和 seed 不是 FullConfig 的一部分，它们只在 main() 中使用
        overrides = {}
        if args.initial_cash != DEFAULT_INITIAL_CASH:
            overrides["initial_cash"] = args.initial_cash
        if args.exec_price_model != DEFAULT_EXEC_PRICE_MODEL:
            overrides["exec_price_model"] = args.exec_price_model
        if args.slippage_bps != DEFAULT_SLIPPAGE_BPS:
            overrides["slippage_bps"] = args.slippage_bps
        if args.commission_rate != DEFAULT_COMMISSION_RATE:
            overrides["commission_rate"] = args.commission_rate
        if args.commission_min != DEFAULT_COMMISSION_MIN:
            overrides["commission_min"] = args.commission_min

        if overrides:
            full_config = full_config.apply_cli_overrides(**overrides)

        return full_config
    else:
        # 从 CLI/default 生成 FullConfig
        return FullConfig(
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


def resolve_runtime_values(args: argparse.Namespace, effective_config: FullConfig, latest_run_id: int) -> dict:
    """
    统一解析运行时变量，确保所有地方使用一致的值。

    语义说明：
        1. effective_runs: 从 effective_config.search_config.runs 读取，优先于 args.runs
        2. base_seed: CLI 传入的基础种子（--seed）
        3. search_seed: 实际用于随机搜索的种子（base_seed + latest_run_id，仅 continue 模式不同）
        4. final_train_seed: 用于最终全样本模型训练的种子（始终 = base_seed）
        5. effective_initial_cash: 从 effective_config.trade_config.initial_cash 读取
    """
    base_seed = int(args.seed)
    return {
        "effective_runs": int(effective_config.search_config.runs),
        "base_seed": base_seed,
        "search_seed": base_seed + int(latest_run_id),
        "final_train_seed": base_seed,
        "effective_initial_cash": float(effective_config.trade_config.initial_cash),
    }


def build_repro_config(best_config: dict, effective_config: FullConfig) -> FullConfig:
    """
    构造用于复现的完整配置。

    问题：search_engine.py 返回的 best_config 只包含 feature_config / model_config / trade_config，
         但 trade_config 只包含阈值、持有期、TP/SL，不包含 exec_price_model、滑点、佣金等执行假设。

    解决方案：
        1. 从 effective_config 复制完整配置（包含所有执行假设）
        2. 用 best_config 中的策略参数覆盖对应部分
        3. 返回一个真正可复现的 FullConfig
    """
    repro = FullConfig.from_dict(effective_config.to_dict())

    # 用搜索得到的最优策略参数覆盖
    repro.feature_config = FeatureConfig.from_dict(best_config["feature_config"])
    repro.model_config = ModelConfig.from_dict(best_config["model_config"])

    # trade_config: 只覆盖阈值/持有期/tp/sl 等策略参数，保留执行假设
    merged_trade = repro.trade_config.to_dict()
    merged_trade.update(best_config["trade_config"])
    repro.trade_config = TradeConfig.from_dict(merged_trade)

    return repro


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
        help="Execution price model: same_close_idealized (t signal, t+1 exec at t close), next_open (t signal, t+1 exec at t+1 open, recommended), next_close (t signal, t+1 exec at t+1 close). Default: next_open."
    )
    parser.add_argument(
        "--slippage_bps",
        type=float,
        default=10.0,
        help="Slippage in basis points (1 bp = 0.01%%). Default: 10 bps (0.1%%)."
    )
    parser.add_argument(
        "--commission_rate",
        type=float,
        default=0.00025,
        help="Commission rate (default: 0.00025 = 0.025%% = 万分之 2.5)."
    )
    parser.add_argument(
        "--commission_min",
        type=float,
        default=5.0,
        help="Minimum commission in CNY (default: 5 yuan)."
    )

    # Reproducibility and engineering
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file for reproduction (from previous run)."
    )
    parser.add_argument(
        "--record-metadata",
        dest="record_metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record full run metadata for reproducibility. Use --no-record-metadata to disable (default: True)."
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
    # Resolve Effective Config (统一配置入口)
    # =========================================================
    effective_config = resolve_effective_config(args)

    # 打印配置加载优先级
    if args.config:
        logger.info(f"Loaded config from: {args.config}")
        logger.info("Config priority: CLI overrides > config file > defaults")
        if args.runs != 100:
            logger.info(f"Overriding runs from CLI: runs={args.runs} (config file value ignored)")
        if args.seed != 42:
            logger.info(f"Overriding seed from CLI: seed={args.seed} (config file value ignored)")
    else:
        logger.info("Using default/CLI config")
        logger.info("Config priority: CLI args > defaults")

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

    # 从 effective_config 提取 trade_params，不再硬编码
    trade_params = effective_config.trade_config.to_dict()

    # compute effective seed based on mode and latest run id
    latest_run_id = _get_latest_run_id(args.output_dir) if args.mode == "continue" else 0

    # 统一解析运行时变量
    rt = resolve_runtime_values(args, effective_config, latest_run_id)
    effective_runs = rt["effective_runs"]
    base_seed = rt["base_seed"]
    search_seed = rt["search_seed"]
    final_train_seed = rt["final_train_seed"]
    effective_initial_cash = rt["effective_initial_cash"]

    # 从 effective_config.search_config 提取随机搜索参数
    search_cfg = effective_config.search_config
    epsilon = search_cfg.epsilon
    exploit_ratio = search_cfg.exploit_ratio
    top_k = search_cfg.top_k
    max_features = search_cfg.max_features
    n_jobs = search_cfg.n_jobs

    # 打印执行假设（Execution Assumptions）
    print("\n" + "=" * 60)
    print("Execution Assumptions (训练/验证/最终回测一致)")
    print("=" * 60)
    print(f"  Price Model:    {effective_config.trade_config.exec_price_model}")
    print(f"  Slippage:       {effective_config.trade_config.slippage_bps:.1f} bps ({effective_config.trade_config.slippage_bps/10000:.2%})")
    print(f"  Commission:     {effective_config.trade_config.commission_rate*100:.3f}% (min ¥{effective_config.trade_config.commission_min:.2f})")
    print(f"  Transfer Fee:   0.001% (万分之 0.1)")
    print(f"  Stamp Tax:      0.05% (千分之 0.5, 仅卖出)")
    print(f"  Buy Threshold:  {effective_config.trade_config.buy_threshold:.2f}")
    print(f"  Sell Threshold: {effective_config.trade_config.sell_threshold:.2f}")
    print(f"  Confirm Days:   {effective_config.trade_config.confirm_days}")
    print(f"  Min Hold Days:  {effective_config.trade_config.min_hold_days}")
    print("=" * 60)
    print("\nSearch Hyperparameters (from search_config)")
    print("=" * 60)
    print(f"  Runs:           {search_cfg.runs}")
    print(f"  Epsilon:        {search_cfg.epsilon:.4f} (min improvement)")
    print(f"  Exploit Ratio:  {search_cfg.exploit_ratio:.2f} ({int(search_cfg.runs * search_cfg.exploit_ratio)} exploit, {search_cfg.runs - int(search_cfg.runs * search_cfg.exploit_ratio)} explore)")
    print(f"  Top-K:          {search_cfg.top_k}")
    print(f"  Max Features:   {search_cfg.max_features}")
    print(f"  N-Jobs:         {search_cfg.n_jobs} (-1=all cores, 1=single, >0=specified)")
    print("=" * 60 + "\n")

    train_res = random_search_train(
        df_clean=df_clean,
        runs=effective_runs,
        seed=search_seed,
        base_best_config=base_best_config,
        output_dir=str(args.output_dir),
        epsilon=epsilon,
        exploit_ratio=exploit_ratio,
        top_k=top_k,
        trade_params=trade_params,
        max_features=max_features,
        n_jobs=n_jobs,
        exec_price_model=effective_config.trade_config.exec_price_model,
        slippage_bps=effective_config.trade_config.slippage_bps,
        commission_rate=effective_config.trade_config.commission_rate,
        commission_min=effective_config.trade_config.commission_min,
        n_folds=effective_config.n_folds,
        train_start_ratio=effective_config.train_start_ratio,
        wf_min_rows=effective_config.wf_min_rows,
    )

    # best_config: 完整配置（包含执行假设），用于向后兼容
    # best_strategy_config: 策略配置（仅含可优化的策略参数），推荐用于复现配置组装
    best_config = train_res.best_config
    best_strategy_config = train_res.best_strategy_config if train_res.best_strategy_config else best_config

    print(f"[INFO] Best validation metric (geom mean ratio): {train_res.best_val_metric:.6f}")
    print(f"[INFO] Best validation equity proxy (mean): {train_res.best_val_final_equity_proxy:.2f}")

    dpoint, artifacts = train_final_model_and_dpoint(df_clean, best_config, seed=final_train_seed)

    tc = best_config["trade_config"]
    bt = backtest_from_dpoint(
        df=df_clean,
        dpoint=dpoint,
        initial_cash=float(tc["initial_cash"]),
        buy_threshold=float(tc["buy_threshold"]),
        sell_threshold=float(tc["sell_threshold"]),
        confirm_days=int(tc["confirm_days"]),
        min_hold_days=int(tc["min_hold_days"]),
        max_hold_days=int(tc.get("max_hold_days", 20)),
        take_profit=tc.get("take_profit", None),
        stop_loss=tc.get("stop_loss", None),
        exec_price_model=effective_config.trade_config.exec_price_model,
        slippage_bps=effective_config.trade_config.slippage_bps,
        commission_rate=effective_config.trade_config.commission_rate,
        commission_min=effective_config.trade_config.commission_min,
    )

    final_equity = float(bt.equity_curve["total_equity"].iloc[-1]) if not bt.equity_curve.empty else effective_initial_cash
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
    log_notes.append(f"Runs (search iterations): {effective_runs}")
    log_notes.append(f"Base seed (CLI --seed): {base_seed}")
    log_notes.append(f"Search seed (base + latest_run_id={latest_run_id}): {search_seed}")
    log_notes.append(f"Final train seed (same as base): {final_train_seed}")
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

    # 构造复现配置用于保存和 metadata
    # 使用 best_strategy_config（仅含策略参数）而非 best_config，确保正确组装
    repro_config = build_repro_config(best_strategy_config, effective_config)

    # 构造运行上下文
    run_context = {
        "mode": args.mode,
        "base_seed": base_seed,
        "search_seed": search_seed,
        "final_train_seed": final_train_seed,
        "effective_runs": effective_runs,
        "config_source": args.config if args.config else "CLI/default",
        "git_commit": get_git_commit(),
        "hostname": get_hostname(),
    }

    excel_path, config_path, run_id = save_run_outputs(
        output_dir=str(args.output_dir),
        df_clean=df_clean,
        log_notes=log_notes,
        trades=bt.trades,
        equity_curve=bt.equity_curve,
        best_strategy_config=best_strategy_config,
        repro_config=repro_config.to_dict(),
        run_context=run_context,
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
            # 使用上面已构造的 repro_config

            # 使用结构化字段记录三种种子语义和运行时上下文
            metadata = RunMetadata(
                run_id=run_id,
                created_at=datetime.now().isoformat(timespec="seconds"),
                code_version=get_code_version(),
                python_version=get_python_version(),
                dependency_versions=get_dependency_versions(),
                data_hash=data_hash,
                data_path=args.data_path,
                base_seed=base_seed,
                search_seed=search_seed,
                final_train_seed=final_train_seed,
                mode=args.mode,
                effective_runs=effective_runs,
                effective_config_source=args.config if args.config else "CLI/default",
                config=repro_config,
                git_commit=get_git_commit(),
                hostname=get_hostname(),
                notes=[
                    f"Base seed (CLI --seed): {base_seed}",
                    f"Search seed (base + latest_run_id={latest_run_id}): {search_seed}",
                    f"Final train seed (same as base): {final_train_seed}",
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
