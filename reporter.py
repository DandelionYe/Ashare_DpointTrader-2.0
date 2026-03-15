# reporter.py
"""
结果报告生成（Walk-Forward 分离版本）。

关键改进：
    - WalkForwardSummary sheet：仅展示每折样本外指标（主 KPI）
    - FinalFit_InSample sheet：明确标注"仅供参考，不代表样本外表现"
    - Log sheet：仅包含运行日志，不再混入搜索日志
    - SearchLog sheet：完整的随机搜索迭代日志

用户第一眼看到的是样本外表现，而不是样本内拟合曲线。
"""
from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import pandas as pd

from constants import MIN_CLOSED_TRADES_PER_FOLD, TARGET_CLOSED_TRADES_PER_FOLD, LAMBDA_TRADE_PENALTY


def escape_excel_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prevent Excel from treating text as formulas (which can trigger repair prompts),
    by prefixing strings starting with = + - @ with a single quote.
    """
    df2 = df.copy()
    for col in df2.columns:
        # Check for both 'object' (old pandas) and 'string' (new pandas) dtypes
        if df2[col].dtype == "object" or df2[col].dtype == "string":
            df2[col] = df2[col].apply(
                lambda v: ("'" + v) if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v
            )
    return df2


def _hash_dataframe(df: pd.DataFrame) -> str:
    """
    对 DataFrame 内容做 SHA-256 哈希，用于检测数据变化。
    使用 pandas 内置哈希（比 to_csv 快约 10x），结果为 16 进制字符串。
    """
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()


def _next_run_id(output_dir: str) -> int:
    """
    扫描 output_dir 中已有的 run_XXX_config.json，返回下一个可用的 run_id（从 1 开始）。
    若目录为空则返回 1。
    """
    os.makedirs(output_dir, exist_ok=True)
    existing = []
    for fn in os.listdir(output_dir):
        if fn.startswith("run_") and fn.endswith("_config.json"):
            try:
                n = int(fn.split("_")[1])
                existing.append(n)
            except Exception:
                pass
    return (max(existing) + 1) if existing else 1


def find_latest_run(output_dir: str) -> Optional[Tuple[int, str, str]]:
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for fn in os.listdir(output_dir):
        if fn.startswith("run_") and fn.endswith("_config.json"):
            try:
                run_id = int(fn.split("_")[1])
                cfg_path = os.path.join(output_dir, fn)
                xlsx_path = os.path.join(output_dir, f"run_{run_id:03d}.xlsx")
                candidates.append((run_id, cfg_path, xlsx_path))
            except Exception:
                continue
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0])[-1]


def _build_execution_assumptions(config: Dict[str, object]) -> pd.DataFrame:
    """
    构建执行假设说明页，展示交易成本、滑点、执行价模型等参数。
    """
    trade_cfg = config.get("trade_config", {})
    
    rows = [
        {"Parameter": "=== Execution Price Model ===", "Value": "", "Description": ""},
        {"Parameter": "exec_price_model", "Value": str(trade_cfg.get("exec_price_model", "N/A")), "Description": "Price used for order execution"},
        {"Parameter": "", "Value": "", "Description": ""},
        {"Parameter": "=== Slippage ===", "Value": "", "Description": ""},
        {"Parameter": "slippage_bps", "Value": f"{trade_cfg.get('slippage_bps', 0):.1f} bps", "Description": f"({trade_cfg.get('slippage_bps', 0)/10000:.2%})"},
        {"Parameter": "", "Value": "", "Description": ""},
        {"Parameter": "=== Transaction Costs ===", "Value": "", "Description": ""},
        {"Parameter": "commission_rate", "Value": f"{trade_cfg.get('commission_rate', 0)*100:.3f}%", "Description": "Commission rate (both sides)"},
        {"Parameter": "commission_min", "Value": f"¥{trade_cfg.get('commission_min', 0):.2f}", "Description": "Minimum commission"},
        {"Parameter": "transfer_fee_rate", "Value": "0.001%", "Description": "Transfer fee (both sides, 万分之 0.1)"},
        {"Parameter": "stamp_tax_rate", "Value": "0.05%", "Description": "Stamp tax (sell only, 千分之 0.5)"},
        {"Parameter": "", "Value": "", "Description": ""},
        {"Parameter": "=== Trading Rules ===", "Value": "", "Description": ""},
        {"Parameter": "buy_threshold", "Value": f"{trade_cfg.get('buy_threshold', 0):.2f}", "Description": "Dpoint threshold for buy signal"},
        {"Parameter": "sell_threshold", "Value": f"{trade_cfg.get('sell_threshold', 0):.2f}", "Description": "Dpoint threshold for sell signal"},
        {"Parameter": "confirm_days", "Value": str(trade_cfg.get("confirm_days", 0)), "Description": "Days to confirm signal"},
        {"Parameter": "min_hold_days", "Value": str(trade_cfg.get("min_hold_days", 0)), "Description": "Minimum holding period"},
        {"Parameter": "max_hold_days", "Value": str(trade_cfg.get("max_hold_days", 20)), "Description": "Maximum holding period"},
        {"Parameter": "take_profit", "Value": str(trade_cfg.get("take_profit", "None")), "Description": "Take-profit threshold (EOD-based)"},
        {"Parameter": "stop_loss", "Value": str(trade_cfg.get("stop_loss", "None")), "Description": "Stop-loss threshold (EOD-based)"},
    ]
    
    return pd.DataFrame(rows)


def _build_run_context(
    run_context: Dict[str, object],
    repro_config: Dict[str, object],
) -> pd.DataFrame:
    """
    构建运行上下文说明页，展示 mode/seeds/config_source 等运行时信息。
    """
    rows = [
        {"Parameter": "=== Run Mode ===", "Value": "", "Description": ""},
        {"Parameter": "mode", "Value": str(run_context.get("mode", "N/A")), "Description": "Run mode (first / continue)"},
        {"Parameter": "config_source", "Value": str(run_context.get("config_source", "CLI/default")), "Description": "Source of effective config"},
        {"Parameter": "", "Value": "", "Description": ""},
        {"Parameter": "=== Random Seeds ===", "Value": "", "Description": ""},
        {"Parameter": "base_seed", "Value": str(run_context.get("base_seed", "N/A")), "Description": "Base seed from CLI --seed"},
        {"Parameter": "search_seed", "Value": str(run_context.get("search_seed", "N/A")), "Description": "Search seed (base + latest_run_id)"},
        {"Parameter": "final_train_seed", "Value": str(run_context.get("final_train_seed", "N/A")), "Description": "Final training seed (same as base)"},
        {"Parameter": "", "Value": "", "Description": ""},
        {"Parameter": "=== Search Configuration ===", "Value": "", "Description": ""},
        {"Parameter": "effective_runs", "Value": str(run_context.get("effective_runs", "N/A")), "Description": "Actual search iterations from search_config.runs"},
        {"Parameter": "", "Value": "", "Description": ""},
        {"Parameter": "=== Walk-Forward Splits ===", "Value": "", "Description": ""},
        {"Parameter": "n_folds", "Value": str(repro_config.get("n_folds", "N/A")), "Description": "Number of walk-forward folds"},
        {"Parameter": "train_start_ratio", "Value": str(repro_config.get("train_start_ratio", "N/A")), "Description": "Initial training set ratio"},
        {"Parameter": "wf_min_rows", "Value": str(repro_config.get("wf_min_rows", "N/A")), "Description": "Minimum rows per fold"},
        {"Parameter": "", "Value": "", "Description": ""},
        {"Parameter": "=== System Info ===", "Value": "", "Description": ""},
        {"Parameter": "git_commit", "Value": str(run_context.get("git_commit", "N/A")), "Description": "Git commit hash"},
        {"Parameter": "hostname", "Value": str(run_context.get("hostname", "N/A")), "Description": "Hostname where run executed"},
    ]

    return pd.DataFrame(rows)


def _build_walkforward_summary(
    search_log: pd.DataFrame,
    config: Dict[str, object],
) -> pd.DataFrame:
    """
    从 search_log 中提取 Walk-Forward 验证的样本外指标摘要。

    优先解析 fold_details_json 列（如果存在），否则回退到旧逻辑。
    """
    if search_log.empty:
        return pd.DataFrame(columns=[
            "Fold", "Out-of-Sample Return (Geom Mean)", "Min Fold Ratio",
            "Avg Trades per Fold", "Equity Proxy Mean", "Metric (Raw)",
            "Metric (Final)", "Penalty"
        ])

    # 尝试解析 fold_details_json 列
    if "fold_details_json" in search_log.columns:
        # 检查是否有有效的 fold_details
        valid_fold_details = False
        for _, row in search_log.iterrows():
            fd_json = row.get("fold_details_json", "[]")
            if fd_json and fd_json != "[]":
                try:
                    fold_details = json.loads(fd_json)
                    if fold_details:
                        valid_fold_details = True
                        break
                except (json.JSONDecodeError, TypeError):
                    pass

        if valid_fold_details:
            # 使用 fold_details 构建详细的逐折摘要
            summary_rows = []
            
            # 获取最优配置（search_log 中 val_metric_final 最高的行）
            best_row = search_log.loc[search_log["val_metric_final"].idxmax()]
            best_fd_json = best_row.get("fold_details_json", "[]")
            
            try:
                best_fold_details = json.loads(best_fd_json) if best_fd_json else []
            except (json.JSONDecodeError, TypeError):
                best_fold_details = []

            if best_fold_details:
                for fd in best_fold_details:
                    summary_rows.append({
                        "Fold": f"Fold {fd.get('fold_id', '?') + 1}",
                        "Out-of-Sample Return (Geom Mean)": f"{fd.get('equity_ratio', 0):.4f}",
                        "Drawdown": "",  # 可从 fold_details 扩展
                        "Trades": str(fd.get('n_closed', 0)),
                        "Win Rate": "",  # 可从 fold_details 扩展
                        "Sharpe": "",  # 可从 fold_details 扩展
                        "Train Rows": str(fd.get('train_rows', 0)),
                        "Val Rows": str(fd.get('val_rows', 0)),
                        "Equity End": f"{fd.get('equity_end', 0):.2f}",
                        "Exec Price Model": str(fd.get('exec_price_model', 'N/A')),
                        "Slippage": f"{fd.get('slippage_bps', 0):.1f} bps",
                        "Commission": f"{fd.get('commission_rate', 0)*100:.3f}%",
                    })
                
                # 添加汇总行
                summary_rows.append({
                    "Fold": "---",
                    "Out-of-Sample Return (Geom Mean)": "",
                    "Drawdown": "",
                    "Trades": "",
                    "Win Rate": "",
                    "Sharpe": "",
                    "Train Rows": "",
                    "Val Rows": "",
                    "Equity End": "",
                    "Exec Price Model": "",
                    "Slippage": "",
                    "Commission": "",
                })

            # 添加聚合指标
            summary_rows.append({
                "Fold": "Overall (Aggregated)",
                "Out-of-Sample Return (Geom Mean)": f"{best_row.get('val_geom_mean_ratio', 0):.4f}",
                "Drawdown": "",
                "Trades": f"{best_row.get('val_avg_closed_trades_per_fold', 0):.2f}",
                "Win Rate": "",
                "Sharpe": "",
                "Train Rows": "",
                "Val Rows": "",
                "Equity End": "",
                "Exec Price Model": "",
                "Slippage": "",
                "Commission": "",
            })
            summary_rows.append({
                "Fold": "Metric (Raw)",
                "Out-of-Sample Return (Geom Mean)": f"{best_row.get('val_metric_raw', 0):.6f}",
                "Drawdown": "",
                "Trades": "",
                "Win Rate": "",
                "Sharpe": "",
                "Train Rows": "",
                "Val Rows": "",
                "Equity End": "",
                "Exec Price Model": "",
                "Slippage": "",
                "Commission": "",
            })
            summary_rows.append({
                "Fold": "Metric (Final)",
                "Out-of-Sample Return (Geom Mean)": f"{best_row.get('val_metric_final', 0):.6f}",
                "Drawdown": "",
                "Trades": "",
                "Win Rate": "",
                "Sharpe": "",
                "Train Rows": "",
                "Val Rows": "",
                "Equity End": "",
                "Exec Price Model": "",
                "Slippage": "",
                "Commission": "",
            })
            summary_rows.append({
                "Fold": "Penalty",
                "Out-of-Sample Return (Geom Mean)": f"{best_row.get('val_penalty', 0):.6f}",
                "Drawdown": "",
                "Trades": "",
                "Win Rate": "",
                "Sharpe": "",
                "Train Rows": "",
                "Val Rows": "",
                "Equity End": "",
                "Exec Price Model": "",
                "Slippage": "",
                "Commission": "",
            })

            return pd.DataFrame(summary_rows)

    # Fallback: 旧逻辑（兼容旧文件）
    summary_rows = []

    best_row = search_log.loc[search_log["val_metric_final"].idxmax()]

    summary_rows.append({
        "Fold": "Overall",
        "Out-of-Sample Return (Geom Mean)": f"{best_row.get('val_geom_mean_ratio', 0):.4f}",
        "Min Fold Ratio": f"{best_row.get('val_min_fold_ratio', 0):.4f}",
        "Avg Trades per Fold": f"{best_row.get('val_avg_closed_trades_per_fold', 0):.2f}",
        "Equity Proxy Mean": f"{best_row.get('val_equity_proxy_mean', 0):.2f}",
        "Metric (Raw)": f"{best_row.get('val_metric_raw', 0):.6f}",
        "Metric (Final)": f"{best_row.get('val_metric_final', 0):.6f}",
        "Penalty": f"{best_row.get('val_penalty', 0):.6f}",
    })

    # 添加警告说明（旧逻辑）
    summary_rows.append({
        "Fold": "⚠️ WARNING",
        "Out-of-Sample Return (Geom Mean)": "",
        "Min Fold Ratio": "",
        "Avg Trades per Fold": "",
        "Equity Proxy Mean": "",
        "Metric (Raw)": "",
        "Metric (Final)": "",
        "Penalty": "",
    })
    summary_rows.append({
        "Fold": "Derived from aggregate metrics; fold-level details unavailable",
        "Out-of-Sample Return (Geom Mean)": "",
        "Min Fold Ratio": "",
        "Avg Trades per Fold": "",
        "Equity Proxy Mean": "",
        "Metric (Raw)": "",
        "Metric (Final)": "",
        "Penalty": "",
    })
    summary_rows.append({
        "Fold": "以上指标均为 Walk-Forward 样本外验证结果",
        "Out-of-Sample Return (Geom Mean)": "",
        "Min Fold Ratio": "",
        "Avg Trades per Fold": "",
        "Equity Proxy Mean": "",
        "Metric (Raw)": "",
        "Metric (Final)": "",
        "Penalty": "",
    })
    summary_rows.append({
        "Fold": "Out-of-Sample = 真实可期望表现 | In-Sample = 仅供参考",
        "Out-of-Sample Return (Geom Mean)": "",
        "Min Fold Ratio": "",
        "Avg Trades per Fold": "",
        "Equity Proxy Mean": "",
        "Metric (Raw)": "",
        "Metric (Final)": "",
        "Penalty": "",
    })

    return pd.DataFrame(summary_rows)


def _build_insample_warning_sheet() -> pd.DataFrame:
    """
    构建样本内结果警告页。
    """
    rows = [
        {"Item": "⚠️ 警告 / WARNING"},
        {"Item": ""},
        {"Item": "此 Sheet 展示的是全样本拟合结果（In-Sample Fit）。"},
        {"Item": "This sheet shows the Full-Sample Fit result (In-Sample)."},
        {"Item": ""},
        {"Item": "⚠️ 此结果存在前向偏差（Look-Ahead Bias），数值偏乐观。"},
        {"Item": "⚠️ This result has Look-Ahead Bias and is overly optimistic."},
        {"Item": ""},
        {"Item": "📊 真实可期望表现请查看 [WalkForwardSummary] Sheet 中的样本外指标。"},
        {"Item": "📊 For real out-of-sample performance, see the [WalkForwardSummary] sheet."},
        {"Item": ""},
        {"Item": "=== 为什么样本内结果不可靠？ ==="},
        {"Item": "1. 模型在全部数据上训练并预测，存在信息泄露"},
        {"Item": "2. 未考虑隔夜跳空、滑点和成交偏差"},
        {"Item": "3. 无法反映策略在未来数据上的真实表现"},
        {"Item": ""},
        {"Item": "=== Why In-Sample Results Are Unreliable? ==="},
        {"Item": "1. Model is trained and predicted on the same data (information leakage)"},
        {"Item": "2. Does not account for overnight gaps, slippage, and fill deviation"},
        {"Item": "3. Cannot reflect true performance on future data"},
    ]
    return pd.DataFrame(rows)


def save_run_outputs(
    output_dir: str,
    df_clean: pd.DataFrame,
    log_notes: List[str],
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    best_strategy_config: Dict[str, object],
    repro_config: Dict[str, object],
    run_context: Dict[str, object],
    feature_meta: Dict[str, object],
    search_log: pd.DataFrame,
    model_params: Optional[Dict[str, object]] = None,
) -> Tuple[str, str, int]:
    """
    保存运行输出到 Excel 和 JSON 文件。

    参数说明：
        - best_strategy_config: 最佳策略配置（仅含可优化的策略参数）
        - repro_config: 完整复现配置（包含所有执行假设和 walk-forward 参数）
        - run_context: 运行上下文（mode/seeds/config_source/git_commit/hostname 等）

    Excel Sheet 结构（按重要性排序）：
        1. WalkForwardSummary — 样本外验证指标（主 KPI，用户应首先关注）
        2. RunContext — 运行上下文（mode/seeds/config_source 等）
        3. ExecutionAssumptions — 执行假设说明
        4. Trades — 交易记录
        5. EquityCurve — 净值曲线（样本内，仅供参考）
        6. FinalFit_InSample — 样本内结果警告页
        7. SearchLog — 完整的随机搜索迭代日志
        8. Config — 配置参数
        9. ModelParams — 模型参数（特征系数等）
        10. Log — 运行日志
    """
    run_id = _next_run_id(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    excel_path = os.path.join(output_dir, f"run_{run_id:03d}.xlsx")
    config_path = os.path.join(output_dir, f"run_{run_id:03d}_config.json")

    df_hash = _hash_dataframe(df_clean)

    # ---------- build config JSON ----------
    # 新结构：明确区分 best_strategy_config / repro_config / run_context
    config_blob = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_hash": df_hash,
        "best_strategy_config": best_strategy_config,
        "repro_config": repro_config,
        "run_context": run_context,
        "feature_meta": feature_meta,
        "notes": {
            "execution_assumption": "Signal uses day t data; order executes on t+1 at t+1 price (realistic).",
            "a_share_constraints": "Long-only, buy before sell, no short, min 100 shares, full-in/out, T+1 approximated via min_hold_days>=1.",
            "out_of_sample_note": "Walk-forward validation ensures out-of-sample evaluation. See WalkForwardSummary sheet.",
        },
    }

    # Config sheet dataframe（保持向后兼容，使用 repro_config）
    config_rows = []
    config_rows.append(("run_id", run_id))
    config_rows.append(("created_at", config_blob["created_at"]))
    config_rows.append(("data_hash", df_hash))
    config_rows.append(("split_mode", repro_config.get("split_mode", "")))

    for k, v in repro_config.get("feature_config", {}).items():
        config_rows.append((f"feature.{k}", str(v)))
    for k, v in repro_config.get("model_config", {}).items():
        config_rows.append((f"model.{k}", str(v)))
    for k, v in repro_config.get("trade_config", {}).items():
        config_rows.append((f"trade.{k}", str(v)))

    config_rows.append(("constraint.min_closed_trades_per_fold", MIN_CLOSED_TRADES_PER_FOLD))
    config_rows.append(("penalty.target_closed_trades_per_fold", TARGET_CLOSED_TRADES_PER_FOLD))
    config_rows.append(("penalty.lambda_trade_penalty", LAMBDA_TRADE_PENALTY))

    config_rows.append(("dpoint_definition", feature_meta.get("dpoint_explainer", "")))
    config_df = pd.DataFrame(config_rows, columns=["key", "value"])

    # Log notes dataframe
    notes_df = pd.DataFrame({"notes": log_notes})

    # ---------- Build specialized sheets ----------

    # 1. WalkForwardSummary: 样本外验证指标（主 KPI）
    walkforward_summary = _build_walkforward_summary(search_log, repro_config)
    walkforward_summary_safe = escape_excel_formulas(walkforward_summary)

    # 2. RunContext: 运行上下文（新增）
    run_context_sheet = _build_run_context(run_context, repro_config)
    run_context_sheet_safe = escape_excel_formulas(run_context_sheet)

    # 3. ExecutionAssumptions: 执行假设说明
    execution_assumptions = _build_execution_assumptions(repro_config)
    execution_assumptions_safe = escape_excel_formulas(execution_assumptions)

    # 4. FinalFit_InSample: 样本内结果警告页
    insample_warning = _build_insample_warning_sheet()
    insample_warning_safe = escape_excel_formulas(insample_warning)

    # ---------- escape Excel formulas BEFORE writing Excel ----------
    trades_safe = escape_excel_formulas(trades)
    equity_safe = escape_excel_formulas(equity_curve)
    config_safe = escape_excel_formulas(config_df)
    notes_safe = escape_excel_formulas(notes_df)
    search_safe = escape_excel_formulas(search_log)

    model_params_effective = model_params
    if model_params_effective is None and isinstance(feature_meta, dict):
        model_params_effective = feature_meta.get("model_params")

    model_params_df: Optional[pd.DataFrame] = None
    if isinstance(model_params_effective, dict):
        feature_names = list(model_params_effective.get("feature_names", []))
        coef = list(model_params_effective.get("coef", []))
        scaler_mean = model_params_effective.get("mean", model_params_effective.get("scaler_mean", []))
        scaler_scale = model_params_effective.get("scale", model_params_effective.get("scaler_scale", []))
        scaler_mean = list(scaler_mean) if isinstance(scaler_mean, (list, tuple)) else []
        scaler_scale = list(scaler_scale) if isinstance(scaler_scale, (list, tuple)) else []

        n = max(len(feature_names), len(coef), len(scaler_mean), len(scaler_scale))
        rows = []
        for i in range(n):
            rows.append(
                {
                    "feature_name": feature_names[i] if i < len(feature_names) else "",
                    "coef": coef[i] if i < len(coef) else "",
                    "scaler_mean": scaler_mean[i] if i < len(scaler_mean) else "",
                    "scaler_scale": scaler_scale[i] if i < len(scaler_scale) else "",
                }
            )
        intercept = model_params_effective.get("intercept")
        if intercept is not None:
            rows.append(
                {
                    "feature_name": "__intercept__",
                    "coef": intercept,
                    "scaler_mean": "",
                    "scaler_scale": "",
                }
            )
        if rows:
            model_params_df = pd.DataFrame(rows, columns=["feature_name", "coef", "scaler_mean", "scaler_scale"])
            model_params_df = escape_excel_formulas(model_params_df)

    # ---------- write config json (ONLY json) ----------
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_blob, f, ensure_ascii=False, indent=2)

    # ---------- write Excel ----------
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        # 第一优先级：样本外验证结果
        walkforward_summary_safe.to_excel(writer, sheet_name="WalkForwardSummary", index=False)

        # 运行上下文（新增）
        run_context_sheet_safe.to_excel(writer, sheet_name="RunContext", index=False)

        # 执行假设说明
        execution_assumptions_safe.to_excel(writer, sheet_name="ExecutionAssumptions", index=False)

        # 交易记录
        trades_safe.to_excel(writer, sheet_name="Trades", index=False)

        # 净值曲线（样本内，仅供参考）
        equity_safe.to_excel(writer, sheet_name="EquityCurve", index=False)

        # 样本内结果警告页
        insample_warning_safe.to_excel(writer, sheet_name="FinalFit_InSample", index=False)

        # 完整的搜索日志
        search_safe.to_excel(writer, sheet_name="SearchLog", index=False)

        # 配置参数
        config_safe.to_excel(writer, sheet_name="Config", index=False)

        # 模型参数
        if model_params_df is not None:
            model_params_df.to_excel(writer, sheet_name="ModelParams", index=False)

        # 运行日志
        notes_safe.to_excel(writer, sheet_name="Log", index=False)

    return excel_path, config_path, run_id
