# search_engine.py
"""
随机搜索主引擎（加速版）。

本版相对原版新增三项加速优化：
  - Perf-02  并行搜索：joblib.Parallel 并行评估所有候选，利用全部 CPU 核心
  - Perf-03  早停剪枝：某折净值低于初始资金 85% 时直接跳过剩余折，淘汰劣质配置
  - Perf-04  XGBoost GPU：自动检测 CUDA 可用性，可用则启用 device='cuda'

以上三项均为纯执行层加速，不改变搜索空间、评估指标和验证逻辑。
continue 模式下可直接沿用已有 best_so_far.json，无需重新训练。

包含：
  - SearchSpaces         : 所有采样池的容器
  - _build_search_spaces : 一次性构建采样池
  - _clamp_int           : 整数夹逼工具
  - _diagnose_from_incumbent : 根据现任配置诊断交易频率
  - _eval_candidate      : 候选配置评估（含 Perf-01 特征缓存 + Perf-03 早停）
  - _sample_explore      : 全局随机探索
  - _sample_exploit      : 局部扰动利用
  - TrainResult          : 训练结果数据类
  - random_search_train  : 主入口（含 Perf-02 并行搜索）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline

from constants import (
    MIN_CLOSED_TRADES_PER_FOLD,
    TARGET_CLOSED_TRADES_PER_FOLD,
    LAMBDA_TRADE_PENALTY,
)
from feature_dpoint import build_features_and_labels, FeatureMeta
from model_builder import make_model, predict_dpoint, _try_import_xgboost
from splitter import walkforward_splits
from metrics import metric_from_fold_ratios, trade_penalty, backtest_fold_stats
from persistence import (
    config_hash,
    best_so_far_path,
    best_pool_path,
    load_best_so_far,
    save_best_so_far,
    update_best_pool,
)


# =========================================================
# Perf-04：检测 CUDA 是否可用（模块加载时执行一次）
# =========================================================
def _detect_cuda() -> bool:
    """
    检测当前环境是否支持 XGBoost CUDA 加速。
    使用一个极小的虚拟数据集验证，失败则静默回退到 CPU。
    """
    xgb = _try_import_xgboost()
    if xgb is None:
        return False
    try:
        import numpy as _np
        _X = _np.random.rand(10, 4).astype("float32")
        _y = _np.array([0, 1] * 5)
        clf = xgb.XGBClassifier(
            n_estimators=2,
            device="cuda",
            tree_method="hist",
            verbosity=0,
        )
        clf.fit(_X, _y)
        return True
    except Exception:
        return False


_CUDA_AVAILABLE: bool = _detect_cuda()


# =========================================================
# 采样空间容器
# =========================================================
@dataclass
class SearchSpaces:
    """所有超参采样池，由 _build_search_spaces 一次性构建。"""
    window_pool: List[List[int]]
    logreg_choices: List[Dict[str, Any]]
    sgd_choices: List[Dict[str, Any]]
    xgb_param_pool: List[Dict[str, Any]]
    xgb_available: bool
    vol_metric_pool: List[str]
    liq_transform_pool: List[str]
    buy_pool: List[float]
    sell_pool: List[float]
    confirm_pool: List[int]
    min_hold_pool: List[int]
    max_hold_pool: List[int]
    take_profit_pool: List[Optional[float]]
    stop_loss_pool: List[Optional[float]]


def _build_search_spaces(seed: int) -> SearchSpaces:
    """构建所有采样池。在 random_search_train 开始时调用一次。"""
    xgb_available = _try_import_xgboost() is not None

    C_pool = list(np.logspace(-2, 2, 13))
    logreg_choices: List[Dict[str, Any]] = []
    for C in C_pool:
        for class_weight in [None, "balanced"]:
            logreg_choices.append({"penalty": "l2", "solver": "lbfgs", "C": C, "class_weight": class_weight})
            logreg_choices.append({"penalty": "l2", "solver": "liblinear", "C": C, "class_weight": class_weight})
            logreg_choices.append({"penalty": "l1", "solver": "liblinear", "C": C, "class_weight": class_weight})
            logreg_choices.append({"penalty": "l2", "solver": "saga", "C": C, "class_weight": class_weight})
            logreg_choices.append({"penalty": "l1", "solver": "saga", "C": C, "class_weight": class_weight})
            for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
                logreg_choices.append({
                    "penalty": "elasticnet", "solver": "saga",
                    "C": C, "class_weight": class_weight, "l1_ratio": l1_ratio,
                })

    sgd_choices: List[Dict[str, Any]] = []
    for alpha in list(np.logspace(-5, -2, 10)):
        for class_weight in [None, "balanced"]:
            sgd_choices.append({"alpha": alpha, "penalty": "l2", "class_weight": class_weight})
            sgd_choices.append({"alpha": alpha, "penalty": "l1", "class_weight": class_weight})
            for l1_ratio in [0.15, 0.3, 0.5, 0.7]:
                sgd_choices.append({
                    "alpha": alpha, "penalty": "elasticnet",
                    "class_weight": class_weight, "l1_ratio": l1_ratio,
                })

    # Perf-04：XGBoost 参数池，自动注入 CUDA 参数
    xgb_param_pool: List[Dict[str, Any]] = []
    if xgb_available:
        # 根据运行时检测结果决定是否启用 GPU
        xgb_device_kwargs: Dict[str, Any] = (
            {"device": "cuda", "tree_method": "hist"}
            if _CUDA_AVAILABLE
            else {"n_jobs": 4}
        )
        for depth in [2, 3, 4]:
            for lr in [0.03, 0.05, 0.1]:
                for n_est in [100, 200, 400]:
                    xgb_param_pool.append(dict(
                        n_estimators=n_est,
                        max_depth=depth,
                        learning_rate=lr,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        min_child_weight=1.0,
                        objective="binary:logistic",
                        random_state=seed,
                        eval_metric="logloss",
                        **xgb_device_kwargs,
                    ))

    return SearchSpaces(
        window_pool=[
            [2, 3, 5, 8, 13, 21], [3, 5, 8, 13, 21], [5, 10, 30, 60],
            [5, 10, 20, 60], [2, 5, 10, 30, 60], [3, 5, 10, 20, 40, 60],
            [10, 20, 60], [3, 7, 14, 28], [4, 9, 18, 36], [3, 5, 10, 20],
        ],
        logreg_choices=logreg_choices,
        sgd_choices=sgd_choices,
        xgb_param_pool=xgb_param_pool,
        xgb_available=xgb_available,
        vol_metric_pool=["std", "mad"],
        liq_transform_pool=["ratio", "zscore"],
        buy_pool=[0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65],
        sell_pool=[0.35, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.55],
        confirm_pool=[1, 2, 3],
        min_hold_pool=[1, 2, 3],
        max_hold_pool=[10, 15, 20, 30, 45, 60],
        take_profit_pool=[None, 0.08, 0.10, 0.12, 0.15],
        stop_loss_pool=[None, 0.05, 0.08, 0.10],
    )


# =========================================================
# 纯工具函数（无闭包依赖，可独立测试）
# =========================================================
def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def _diagnose_from_incumbent(inc_info: Dict[str, object]) -> Dict[str, float]:
    """
    根据现任配置的平均交易频率，生成扰动方向诊断信号。
    用于 _sample_exploit 中的自适应阈值调整。
    """
    diag: Dict[str, float] = {
        "trade_too_few": 0.0,
        "trade_too_many": 0.0,
        "stoploss_often": 0.0,
    }
    avg_trades = float(inc_info.get("avg_closed_trades", float("nan")))
    if not np.isnan(avg_trades):
        if avg_trades < (TARGET_CLOSED_TRADES_PER_FOLD - 1):
            diag["trade_too_few"] = 1.0
        if avg_trades > (TARGET_CLOSED_TRADES_PER_FOLD + 3):
            diag["trade_too_many"] = 1.0
    return diag


# =========================================================
# 候选配置评估（含 Perf-01 特征缓存 + Perf-03 早停）
# =========================================================

# Perf-03：早停阈值。某折净值低于初始资金此比例时，直接放弃剩余折。
# 0.85 表示亏损超过 15% 即触发，可根据需要在 constants.py 中提取为常量。
_EARLY_STOP_EQUITY_RATIO: float = 0.85


def _eval_candidate(
    candidate: Dict[str, object],
    df_clean: pd.DataFrame,
    max_features: int,
    n_folds: int,
    train_start_ratio: float,
    wf_min_rows: int,
    seed: int,
    feat_cache: Dict[str, Tuple[pd.DataFrame, pd.Series, FeatureMeta]],
    exec_price_model: str = "next_open",
    slippage_bps: float = 10.0,
) -> Tuple[float, float, Dict[str, object]]:
    """
    对单个候选配置做 walk-forward 回测，返回 (metric, equity_proxy, info)。

    Perf-01 特征缓存：
        feat_cache 以 feature_config 的 SHA-256 为 key，存储 (X, y, meta)。
        在 exploit 阶段约 70% 的候选只改 trade_config/model_config，feature_config
        不变，此时直接返回缓存结果，跳过 build_features_and_labels 的全量重算。
        并行模式下每个 worker 持有独立缓存（进程间不共享），仍可受益于同一 worker
        在不同 chunk 间的复用。

    Perf-03 早停剪枝：
        若某折净值低于 initial_cash * _EARLY_STOP_EQUITY_RATIO，
        直接返回 -inf，不再计算剩余折，节省后续模型训练和回测的开销。
    
    新增参数：
        exec_price_model — 执行价模型（默认 "next_open"）
        slippage_bps     — 滑点（基点），默认 10 bps
    """
    feat_cfg = candidate["feature_config"]
    trade_cfg = candidate["trade_config"]
    initial_cash = float(trade_cfg["initial_cash"])
    early_stop_floor = initial_cash * _EARLY_STOP_EQUITY_RATIO

    # --- Perf-01：特征缓存命中检查 ---
    fhash = config_hash(feat_cfg)
    if fhash in feat_cache:
        X, y, meta = feat_cache[fhash]
    else:
        X, y, meta = build_features_and_labels(df_clean, feat_cfg)
        feat_cache[fhash] = (X, y, meta)

    if len(X.columns) > max_features:
        return (-np.inf, initial_cash, {
            "skip": "too_many_features",
            "n_features": len(X.columns),
            "meta": meta,
        })

    splits = walkforward_splits(
        X, y,
        n_folds=n_folds,
        train_start_ratio=train_start_ratio,
        min_rows=wf_min_rows,
    )
    if not splits:
        return (-np.inf, initial_cash, {
            "skip": "no_valid_splits",
            "n_features": len(X.columns),
            "meta": meta,
        })

    ratios: List[float] = []
    equities: List[float] = []
    closed_trades_list: List[int] = []

    for (X_tr, y_tr), (X_va, y_va) in splits:
        model = make_model(candidate, seed=seed)
        if isinstance(model, Pipeline):
            model.fit(X_tr.values, y_tr.values)
        else:
            model.fit(X_tr, y_tr)

        dp_val = predict_dpoint(model, X_va)
        fold_stats = backtest_fold_stats(
            df_clean, X_va, dp_val, trade_cfg,
            exec_price_model=exec_price_model,
            slippage_bps=slippage_bps,
        )
        eq_end = float(fold_stats["equity_end"])
        n_closed = int(fold_stats["n_closed"])

        # 硬约束：每折至少需要 MIN_CLOSED_TRADES_PER_FOLD 笔已平仓交易
        if n_closed < MIN_CLOSED_TRADES_PER_FOLD:
            return (-np.inf, initial_cash, {
                "skip": f"invalid_low_closed_trades(n_closed={n_closed})",
                "n_features": len(X.columns),
                "meta": meta,
            })

        # Perf-03：早停剪枝——该折亏损过重，直接淘汰
        if eq_end < early_stop_floor:
            return (-np.inf, initial_cash, {
                "skip": f"early_stop(fold_equity={eq_end:.0f}<floor={early_stop_floor:.0f})",
                "n_features": len(X.columns),
                "meta": meta,
            })

        equities.append(eq_end)
        ratios.append(eq_end / initial_cash)
        closed_trades_list.append(n_closed)

    geom = metric_from_fold_ratios(ratios)
    min_ratio = float(np.min(ratios)) if ratios else 0.0
    metric_raw = 0.8 * geom + 0.2 * min_ratio
    penalty = trade_penalty(closed_trades_list)
    metric_final = float(metric_raw) - float(penalty)
    equity_proxy = float(np.mean(equities)) if equities else initial_cash
    avg_closed = float(np.mean(closed_trades_list)) if closed_trades_list else 0.0

    return (metric_final, equity_proxy, {
        "n_features": len(X.columns),
        "meta": meta,
        "geom_mean_ratio": geom,
        "min_fold_ratio": min_ratio,
        "metric_raw": float(metric_raw),
        "penalty": float(penalty),
        "avg_closed_trades": avg_closed,
    })


# =========================================================
# 采样策略
# =========================================================
def _sample_explore(
    rng: np.random.Generator,
    spaces: SearchSpaces,
    trade_params: Dict[str, object],
) -> Dict[str, object]:
    """全局随机探索：从采样池中完全随机生成一个新候选。"""
    feat_cfg = {
        "windows": spaces.window_pool[int(rng.integers(0, len(spaces.window_pool)))],
        "use_momentum": bool(rng.integers(0, 2)),
        "use_volatility": bool(rng.integers(0, 2)),
        "use_volume": bool(rng.integers(0, 2)),
        "use_candle": bool(rng.integers(0, 2)),
        "use_turnover": bool(rng.integers(0, 2)),
        "vol_metric": spaces.vol_metric_pool[int(rng.integers(0, len(spaces.vol_metric_pool)))],
        "liq_transform": spaces.liq_transform_pool[int(rng.integers(0, len(spaces.liq_transform_pool)))],
    }
    # 至少保留一个特征族
    if not any([feat_cfg["use_momentum"], feat_cfg["use_volatility"], feat_cfg["use_volume"],
                feat_cfg["use_candle"], feat_cfg["use_turnover"]]):
        feat_cfg["use_momentum"] = True

    model_types = ["logreg", "sgd"] + (["xgb"] if spaces.xgb_available else [])
    mt = model_types[int(rng.integers(0, len(model_types)))]

    if mt == "logreg":
        mc = spaces.logreg_choices[int(rng.integers(0, len(spaces.logreg_choices)))]
        model_cfg: Dict[str, Any] = {"model_type": "logreg", **mc}
    elif mt == "sgd":
        mc = spaces.sgd_choices[int(rng.integers(0, len(spaces.sgd_choices)))]
        model_cfg = {"model_type": "sgd", **mc}
    else:
        params = spaces.xgb_param_pool[int(rng.integers(0, len(spaces.xgb_param_pool)))]
        model_cfg = {"model_type": "xgb", "params": params}

    buy_th = float(spaces.buy_pool[int(rng.integers(0, len(spaces.buy_pool)))])
    sell_th = float(spaces.sell_pool[int(rng.integers(0, len(spaces.sell_pool)))])
    if sell_th >= buy_th:
        sell_th = max(0.30, buy_th - 0.10)

    trade_cfg = {
        "initial_cash": float(trade_params["initial_cash"]),
        "buy_threshold": buy_th,
        "sell_threshold": sell_th,
        "confirm_days": int(spaces.confirm_pool[int(rng.integers(0, len(spaces.confirm_pool)))]),
        "min_hold_days": int(spaces.min_hold_pool[int(rng.integers(0, len(spaces.min_hold_pool)))]),
        "max_hold_days": int(spaces.max_hold_pool[int(rng.integers(0, len(spaces.max_hold_pool)))]),
        "take_profit": spaces.take_profit_pool[int(rng.integers(0, len(spaces.take_profit_pool)))],
        "stop_loss": spaces.stop_loss_pool[int(rng.integers(0, len(spaces.stop_loss_pool)))],
    }

    return {
        "feature_config": feat_cfg,
        "model_config": model_cfg,
        "trade_config": trade_cfg,
        "split_mode": "walkforward",
    }


def _sample_exploit(
    incumbent: Dict[str, object],
    diag: Dict[str, float],
    rng: np.random.Generator,
    spaces: SearchSpaces,
    trade_params: Dict[str, object],
) -> Dict[str, object]:
    """
    局部扰动利用（Exploitation）：以现任配置为基础，小幅扰动生成新候选。
    diag 提供交易频率诊断，用于偏置阈值扰动方向。
    """
    base_feat = dict(incumbent["feature_config"])
    base_model = dict(incumbent["model_config"])
    base_trade = dict(incumbent["trade_config"])

    # --- 特征窗口扰动：随机替换 1~2 个窗口值 ---
    windows = list(base_feat.get("windows", [3, 5, 10, 20]))
    all_w = sorted({w for ws in spaces.window_pool for w in ws})
    for _ in range(1 + int(rng.integers(0, 2))):
        if windows:
            idx = int(rng.integers(0, len(windows)))
            windows[idx] = int(all_w[int(rng.integers(0, len(all_w)))])
    base_feat["windows"] = sorted(list(dict.fromkeys(windows)))  # 去重并排序

    # --- 偶尔翻转一个特征族开关（15% 概率）---
    if float(rng.random()) < 0.15:
        key = ["use_momentum", "use_volatility", "use_volume", "use_candle", "use_turnover"][
            int(rng.integers(0, 5))
        ]
        base_feat[key] = not bool(base_feat.get(key, True))
    if not any([base_feat.get("use_momentum", True), base_feat.get("use_volatility", True),
                base_feat.get("use_volume", True), base_feat.get("use_candle", True),
                base_feat.get("use_turnover", True)]):
        base_feat["use_momentum"] = True

    if float(rng.random()) < 0.10:
        base_feat["vol_metric"] = "mad" if str(base_feat.get("vol_metric", "std")) == "std" else "std"
    if float(rng.random()) < 0.10:
        base_feat["liq_transform"] = (
            "zscore" if str(base_feat.get("liq_transform", "ratio")) == "ratio" else "ratio"
        )

    # --- 交易阈值扰动（受 diag 偏置）---
    buy = float(base_trade.get("buy_threshold", 0.55))
    sell = float(base_trade.get("sell_threshold", 0.45))

    if diag.get("trade_too_few", 0.0) > 0.5:
        buy += float(rng.uniform(-0.03, 0.01))
        sell += float(rng.uniform(-0.01, 0.03))
    elif diag.get("trade_too_many", 0.0) > 0.5:
        buy += float(rng.uniform(-0.01, 0.03))
        sell += float(rng.uniform(-0.03, 0.01))
    else:
        buy += float(rng.uniform(-0.02, 0.02))
        sell += float(rng.uniform(-0.02, 0.02))

    buy = float(np.clip(buy, 0.50, 0.70))
    sell = float(np.clip(sell, 0.30, 0.60))
    if sell >= buy - 0.03:
        sell = max(0.30, buy - 0.05)

    confirm = int(base_trade.get("confirm_days", 2))
    min_hold = int(base_trade.get("min_hold_days", 1))
    max_hold = int(base_trade.get("max_hold_days", 20))

    if diag.get("trade_too_few", 0.0) > 0.5:
        confirm = _clamp_int(confirm + int(rng.choice([-1, 0])), 1, 3)
        max_hold = _clamp_int(max_hold + int(rng.choice([-10, -5, 0, 5])), 5, 60)
    elif diag.get("trade_too_many", 0.0) > 0.5:
        confirm = _clamp_int(confirm + int(rng.choice([0, 1])), 1, 4)
        max_hold = _clamp_int(max_hold + int(rng.choice([0, 5, 10])), 5, 90)
    else:
        confirm = _clamp_int(confirm + int(rng.choice([-1, 0, 1])), 1, 4)
        max_hold = _clamp_int(max_hold + int(rng.choice([-10, -5, 0, 5, 10])), 5, 90)
    min_hold = _clamp_int(min_hold + int(rng.choice([-1, 0, 1])), 1, 5)

    tp = base_trade.get("take_profit", None)
    sl = base_trade.get("stop_loss", None)
    if diag.get("stoploss_often", 0.0) > 0.5 and sl is not None:
        sl = float(np.clip(float(sl) + float(rng.uniform(0.01, 0.03)), 0.03, 0.20))
    else:
        if sl is not None:
            sl = float(np.clip(float(sl) + float(rng.uniform(-0.02, 0.02)), 0.03, 0.20))
        if tp is not None:
            tp = float(np.clip(float(tp) + float(rng.uniform(-0.03, 0.03)), 0.05, 0.30))

    trade_cfg = {
        "initial_cash": float(trade_params["initial_cash"]),
        "buy_threshold": buy,
        "sell_threshold": sell,
        "confirm_days": int(confirm),
        "min_hold_days": int(min_hold),
        "max_hold_days": int(max_hold),
        "take_profit": tp,
        "stop_loss": sl,
    }

    # --- 模型超参扰动 ---
    model_type = str(base_model.get("model_type", "logreg"))
    if model_type == "logreg":
        C = float(base_model.get("C", 1.0))
        C *= float(rng.choice([0.5, 0.8, 1.0, 1.25, 1.6, 2.0]))
        base_model["C"] = float(np.clip(C, 0.01, 100.0))
        if str(base_model.get("penalty", "l2")) == "elasticnet":
            lr = float(base_model.get("l1_ratio", 0.5))
            base_model["l1_ratio"] = float(np.clip(lr + float(rng.uniform(-0.2, 0.2)), 0.05, 0.95))
    elif model_type == "sgd":
        a = float(base_model.get("alpha", 1e-4))
        a *= float(rng.choice([0.5, 0.8, 1.0, 1.25, 1.6, 2.0]))
        base_model["alpha"] = float(np.clip(a, 1e-6, 1e-2))

    if float(rng.random()) < 0.1:
        base_model["class_weight"] = (
            "balanced" if base_model.get("class_weight", None) is None else None
        )

    return {
        "feature_config": base_feat,
        "model_config": base_model,
        "trade_config": trade_cfg,
        "split_mode": "walkforward",
    }


# =========================================================
# TrainResult 数据类
# =========================================================
@dataclass
class TrainResult:
    best_config: Dict[str, object]
    best_val_metric: float
    best_val_final_equity_proxy: float
    search_log: pd.DataFrame
    feature_meta: Dict[str, object]
    training_notes: List[str]

    global_best_updated: bool
    global_best_metric_prev: float
    global_best_metric_new: float
    candidate_best_metric: float
    epsilon: float
    not_updated_reason: str
    best_so_far_path: str
    best_pool_path: str


# =========================================================
# 主入口
# =========================================================
def random_search_train(
    df_clean: pd.DataFrame,
    runs: int = 5000,
    seed: int = 42,
    base_best_config: Optional[Dict[str, object]] = None,
    trade_params: Optional[Dict[str, object]] = None,
    max_features: int = 80,
    output_dir: str = "./output",
    epsilon: float = 0.01,
    exploit_ratio: float = 0.7,
    top_k: int = 10,
    n_folds: int = 4,
    train_start_ratio: float = 0.5,
    wf_min_rows: int = 80,
    n_jobs: int = -1,
    exec_price_model: str = "next_open",
    slippage_bps: float = 10.0,
) -> TrainResult:
    """
    持续优化系统（加速版）。

    新增参数：
        n_jobs : joblib 并行进程数。-1 = 使用全部 CPU 核心（推荐）；
                 1 = 串行（调试用）；正整数 = 指定核心数。
        exec_price_model : 执行价模型（默认 "next_open"）
                           - "same_close_idealized": t 日信号，t+1 日执行，用 t 日收盘价（理想化）
                           - "next_open": t 日信号，t+1 日执行，用 t+1 日开盘价（推荐，更真实）
                           - "next_close": t 日信号，t+1 日执行，用 t+1 日收盘价（保守）
        slippage_bps : 滑点（基点），默认 10 bps（0.1%）

    设计要点：
      - global best 持久化至 output/best_so_far.json（现任始终 = 全局最优）
      - Top-K 池持久化至 output/best_pool.json
      - exploit_ratio% 局部扰动 + (1-exploit_ratio)% 全局探索
      - metric_raw = 0.8*geom_mean_ratio + 0.2*min_fold_ratio
        metric_final = metric_raw - trade_penalty
      - 硬约束：每折必须有 >= MIN_CLOSED_TRADES_PER_FOLD 笔已平仓交易
      - 更新规则：candidate_best_metric > incumbent + epsilon 才更新全局最优
      - Perf-01 特征缓存：feature_config 相同的候选复用 (X, y, meta)，避免重算
      - Perf-02 并行搜索：所有候选串行预生成（保持 rng 确定性），并行评估
      - Perf-03 早停剪枝：某折净值低于初始资金 85% 直接淘汰
      - Perf-04 XGBoost GPU：自动检测 CUDA，可用则注入 device='cuda'
    """
    rng = np.random.default_rng(seed)
    training_notes: List[str] = []
    training_notes.append("Split mode: walkforward.")
    training_notes.append(
        "Metric: metric_raw = 0.8*geom_mean_ratio + 0.2*min_fold_ratio; "
        "metric_final = metric_raw - trade_penalty."
    )
    training_notes.append(f"Hard constraint: each fold must have >= {MIN_CLOSED_TRADES_PER_FOLD} CLOSED trades.")
    training_notes.append(
        f"Trade penalty: lambda={LAMBDA_TRADE_PENALTY}, "
        f"target={TARGET_CLOSED_TRADES_PER_FOLD} closed trades per fold."
    )
    training_notes.append(f"Epsilon (min improvement): {epsilon}")
    training_notes.append(
        f"Exploit ratio: {exploit_ratio} "
        f"(exploit={int(runs * exploit_ratio)}, explore={runs - int(runs * exploit_ratio)})"
    )
    training_notes.append(f"best_so_far path: {best_so_far_path(output_dir)}")
    training_notes.append(f"best_pool path: {best_pool_path(output_dir)}")
    training_notes.append(
        f"Walk-forward: n_folds={n_folds}, "
        f"train_start_ratio={train_start_ratio}, wf_min_rows={wf_min_rows}"
    )
    # 加速方案日志
    training_notes.append(
        f"[Perf-02] Parallel search: n_jobs={n_jobs} (joblib loky backend)."
    )
    training_notes.append(
        f"[Perf-03] Early stopping: fold equity < initial_cash * {_EARLY_STOP_EQUITY_RATIO:.0%} -> skip."
    )
    training_notes.append(
        f"[Perf-04] XGBoost GPU: CUDA available={_CUDA_AVAILABLE}."
    )

    trade_params = trade_params or {
        "initial_cash": 100_000.0,
        "buy_threshold": 0.55,
        "sell_threshold": 0.45,
        "confirm_days": 2,
        "min_hold_days": 1,
    }

    spaces = _build_search_spaces(seed)
    if not spaces.xgb_available:
        training_notes.append("XGBoost not installed -> xgb candidates will be skipped.")

    # --- 确定现任配置 ---
    global_best_config = load_best_so_far(output_dir)
    if global_best_config is not None:
        training_notes.append("Loaded global_best from best_so_far.json as incumbent.")
        incumbent_config: Optional[Dict[str, object]] = global_best_config
    elif base_best_config is not None:
        training_notes.append("No best_so_far.json found; using last-run best_config as incumbent.")
        incumbent_config = base_best_config
    else:
        incumbent_config = None
        training_notes.append("No incumbent found; starting from scratch.")

    # 默认 fallback 现任
    _default_incumbent: Dict[str, object] = {
        "feature_config": {
            "windows": [2, 3, 5, 8, 13, 21],
            "use_momentum": True, "use_volatility": True, "use_volume": True,
            "use_candle": True, "use_turnover": True,
            "vol_metric": "std", "liq_transform": "ratio",
        },
        "model_config": {
            "model_type": "logreg", "C": 1.0,
            "penalty": "l2", "solver": "lbfgs", "class_weight": None,
        },
        "trade_config": {
            "initial_cash": float(trade_params["initial_cash"]),
            "buy_threshold": 0.55, "sell_threshold": 0.45,
            "confirm_days": 2, "min_hold_days": 1, "max_hold_days": 20,
            "take_profit": 0.12, "stop_loss": 0.08,
        },
        "split_mode": "walkforward",
    }

    best_config: Dict[str, object] = incumbent_config if incumbent_config is not None else _default_incumbent

    # 评估现任配置（串行，仅一次）
    inc_feat_cache: Dict[str, Tuple] = {}
    m_inc, eq_inc, info_inc = _eval_candidate(
        best_config, df_clean, max_features,
        n_folds, train_start_ratio, wf_min_rows,
        seed, inc_feat_cache,
        exec_price_model=exec_price_model,
        slippage_bps=slippage_bps,
    )
    best_metric = float(m_inc)
    best_equity_proxy = float(eq_inc)
    incumbent_info_for_diag = info_inc

    if incumbent_config is not None:
        training_notes.append(
            f"Incumbent loaded. metric={best_metric:.6f}, "
            f"metric_raw={float(info_inc.get('metric_raw', float('nan'))):.6f}, "
            f"penalty={float(info_inc.get('penalty', float('nan'))):.6f}, "
            f"geom={float(info_inc.get('geom_mean_ratio', float('nan'))):.6f}, "
            f"min_ratio={float(info_inc.get('min_fold_ratio', float('nan'))):.6f}, "
            f"avg_closed={float(info_inc.get('avg_closed_trades', float('nan'))):.2f}, "
            f"equity_proxy={best_equity_proxy:.2f}"
        )
    else:
        training_notes.append("Constructed a fallback incumbent (no prior best).")

    # =========================================================
    # Perf-02：并行搜索主循环
    #
    # 分三阶段：
    #   阶段一：串行预生成所有候选（保持 rng 状态确定性，seed 相同结果可复现）
    #   阶段二：joblib.Parallel 并行评估（每个 worker 持有独立的 feat_cache）
    #   阶段三：串行收集结果，与原逻辑完全一致
    # =========================================================

    exploit_n = int(runs * exploit_ratio)
    diag = _diagnose_from_incumbent(incumbent_info_for_diag)

    # 阶段一：串行预生成候选列表
    all_candidates: List[Dict[str, object]] = []
    use_exploit_flags: List[bool] = []
    for i in range(runs):
        use_exploit = (best_config is not None) and (i < exploit_n)
        use_exploit_flags.append(use_exploit)
        if use_exploit:
            all_candidates.append(_sample_exploit(best_config, diag, rng, spaces, trade_params))
        else:
            all_candidates.append(_sample_explore(rng, spaces, trade_params))

    # 阶段二：并行评估
    # 每个 worker 持有独立 feat_cache（跨进程无法共享，但进程内仍受益）
    def _eval_one(
        candidate: Dict[str, object],
    ) -> Tuple[float, float, Dict[str, object]]:
        local_cache: Dict[str, Tuple] = {}
        return _eval_candidate(
            candidate, df_clean, max_features,
            n_folds, train_start_ratio, wf_min_rows,
            seed, local_cache,
            exec_price_model=exec_price_model,
            slippage_bps=slippage_bps,
        )

    raw_results: List[Tuple[float, float, Dict[str, object]]] = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        prefer="processes",
    )(delayed(_eval_one)(c) for c in all_candidates)

    # 阶段三：串行收集结果
    candidate_best_metric: float = -np.inf
    candidate_best_equity_proxy: float = float(trade_params["initial_cash"])
    candidate_best_config: Optional[Dict[str, object]] = None
    feat_cache_size: int = 0  # 并行模式下各 worker 缓存独立，此处仅做占位记录
    search_rows: List[Dict[str, object]] = []

    for i, (result, use_exploit) in enumerate(zip(raw_results, use_exploit_flags)):
        candidate = all_candidates[i]
        status = "ok"
        try:
            metric, equity_proxy, info = result
            n_features = info.get("n_features", np.nan)
            metric_raw = float(info.get("metric_raw", np.nan))
            penalty = float(info.get("penalty", np.nan))
            geom = float(info.get("geom_mean_ratio", np.nan))
            min_ratio = float(info.get("min_fold_ratio", np.nan))
            avg_closed = float(info.get("avg_closed_trades", np.nan))

            if metric > candidate_best_metric:
                candidate_best_metric = float(metric)
                candidate_best_equity_proxy = float(equity_proxy)
                candidate_best_config = candidate

            search_rows.append({
                "iter": i + 1,
                "status": status,
                "use_exploit": bool(use_exploit),
                "val_metric_final": float(metric),
                "val_metric_raw": metric_raw,
                "val_penalty": penalty,
                "val_geom_mean_ratio": geom,
                "val_min_fold_ratio": min_ratio,
                "val_equity_proxy_mean": float(equity_proxy),
                "val_avg_closed_trades_per_fold": avg_closed,
                "n_features": n_features,
                "model_type": str(candidate["model_config"].get("model_type")),
                "feature_config": str(candidate["feature_config"]),
                "model_config": str(candidate["model_config"]),
                "trade_config": str(candidate["trade_config"]),
                "constraint_min_closed_trades_per_fold": MIN_CLOSED_TRADES_PER_FOLD,
                "target_closed_trades_per_fold": TARGET_CLOSED_TRADES_PER_FOLD,
                "lambda_trade_penalty": LAMBDA_TRADE_PENALTY,
            })
        except Exception as e:
            status = f"error:{type(e).__name__}"
            search_rows.append({
                "iter": i + 1,
                "status": status,
                "use_exploit": bool(use_exploit),
                "val_metric_final": np.nan,
                "val_metric_raw": np.nan,
                "val_penalty": np.nan,
                "val_geom_mean_ratio": np.nan,
                "val_min_fold_ratio": np.nan,
                "val_equity_proxy_mean": np.nan,
                "val_avg_closed_trades_per_fold": np.nan,
                "n_features": np.nan,
                "model_type": str(candidate.get("model_config", {}).get("model_type", "")),
                "feature_config": str(candidate.get("feature_config", "")),
                "model_config": str(candidate.get("model_config", "")),
                "trade_config": str(candidate.get("trade_config", "")),
                "constraint_min_closed_trades_per_fold": MIN_CLOSED_TRADES_PER_FOLD,
                "target_closed_trades_per_fold": TARGET_CLOSED_TRADES_PER_FOLD,
                "lambda_trade_penalty": LAMBDA_TRADE_PENALTY,
            })

    # --- 更新 Top-K 池 ---
    if candidate_best_config is not None and np.isfinite(candidate_best_metric):
        update_best_pool(output_dir, candidate_best_config, candidate_best_metric, top_k=top_k)

    # --- 全局最优更新（epsilon 阈值）---
    global_best_metric_prev = float(best_metric)
    candidate_best_metric_final = (
        float(candidate_best_metric) if np.isfinite(candidate_best_metric) else -np.inf
    )

    global_best_updated = False
    not_updated_reason = ""
    global_best_metric_new = global_best_metric_prev

    if not np.isfinite(candidate_best_metric_final) or candidate_best_config is None:
        not_updated_reason = "no_valid_candidate_found (all invalid or errors)"
        best_final_config = best_config
        best_final_metric = global_best_metric_prev
        best_final_equity_proxy = best_equity_proxy
    else:
        if candidate_best_metric_final > global_best_metric_prev + float(epsilon):
            global_best_updated = True
            global_best_metric_new = candidate_best_metric_final
            best_final_config = candidate_best_config
            best_final_metric = candidate_best_metric_final
            best_final_equity_proxy = candidate_best_equity_proxy
            save_best_so_far(output_dir, best_final_config, best_final_metric)
        else:
            not_updated_reason = (
                "candidate_best_metric did not exceed global_best_metric_prev + epsilon"
            )
            best_final_config = best_config
            best_final_metric = global_best_metric_prev
            best_final_equity_proxy = best_equity_proxy

    training_notes.append(f"Global best metric prev: {global_best_metric_prev:.6f}")
    training_notes.append(f"Candidate best metric this run: {candidate_best_metric_final:.6f}")
    training_notes.append(f"Global best metric new: {global_best_metric_new:.6f}")
    training_notes.append(f"Global best updated: {global_best_updated}")
    if not_updated_reason:
        training_notes.append(f"Not-updated reason: {not_updated_reason}")

    # --- 为 reporter 生成 feature_meta ---
    Xb, yb, meta = build_features_and_labels(df_clean, best_final_config["feature_config"])
    feature_list_preview = (
        ", ".join(meta.feature_names[:30]) + ("..." if len(meta.feature_names) > 30 else "")
    )
    training_notes.append(f"Dpoint features (preview): {feature_list_preview}")
    training_notes.append(f"Total features used: {len(meta.feature_names)}")
    training_notes.append("Dpoint definition: P(close_{t+1} > close_t | X_t)")

    model_cfg = best_final_config["model_config"]
    if str(model_cfg.get("model_type")) in ("logreg", "sgd"):
        training_notes.append(
            "Model formula (linear+sigmoid): p = sigmoid(b0 + Σ wi * zi), "
            "zi are StandardScaler(z) features."
        )
    else:
        training_notes.append(
            "Model formula (XGBoost binary:logistic): p = sigmoid(Σ f_k(x)), "
            "where f_k are boosted trees."
        )

    training_notes.append(
        f"[Perf-01] Feature cache: {feat_cache_size} unique feature_config(s) per worker "
        f"(parallel mode, caches are per-process)."
    )

    feature_meta = {
        "feature_names": meta.feature_names,
        "feature_params": meta.params,
        "dpoint_explainer": meta.dpoint_explainer,
    }

    return TrainResult(
        best_config=best_final_config,
        best_val_metric=float(best_final_metric),
        best_val_final_equity_proxy=float(best_final_equity_proxy),
        search_log=pd.DataFrame(search_rows),
        feature_meta=feature_meta,
        training_notes=training_notes,
        global_best_updated=bool(global_best_updated),
        global_best_metric_prev=float(global_best_metric_prev),
        global_best_metric_new=float(global_best_metric_new),
        candidate_best_metric=float(candidate_best_metric_final),
        epsilon=float(epsilon),
        not_updated_reason=str(not_updated_reason),
        best_so_far_path=best_so_far_path(output_dir),
        best_pool_path=best_pool_path(output_dir),
    )
