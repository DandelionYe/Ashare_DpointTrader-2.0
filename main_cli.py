# main_cli.py
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional, List

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from data_loader import load_stock_excel
from trainer_optimizer import random_search_train, train_final_model_and_dpoint
from backtester_engine import backtest_from_dpoint
from reporter import save_run_outputs, find_latest_run


# ====== 你要求的：留出一个位置让你粘贴数据路径 ======
# ⚠️  本地路径，不要提交到版本控制。建议改用环境变量：
#     export ASHARE_DATA_PATH="/path/to/your/data.xlsx"
#     或在 .env 文件中配置后由 python-dotenv 加载。
DEFAULT_DATA_PATH = r"I:\交易机器学习\项目文件夹\Ashare_DpointTrader 2.0\data\600698_5Y_daily_qfq_20210302_20260302.xlsx"
# ===================================================

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
    args = parser.parse_args()

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

    print(f"[DONE] Saved run {run_id:03d}")
    print(f"  - Excel : {os.path.abspath(excel_path)}")
    print(f"  - Config: {os.path.abspath(config_path)}")


if __name__ == "__main__":
    main()
