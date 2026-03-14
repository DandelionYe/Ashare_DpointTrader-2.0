# feature_dpoint.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureMeta:
    feature_names: List[str]
    params: Dict[str, object]
    dpoint_explainer: str


def _safe_log1p(x: pd.Series) -> pd.Series:
    """对序列做 log1p 变换，先 clip 负值为 0，避免对数域报错。"""
    return np.log1p(np.clip(x.astype(float), 0.0, None))


def _rolling_mad(x: pd.Series, window: int) -> pd.Series:
    """滚动中位数绝对偏差（MAD），比标准差更鲁棒的波动率代理。"""
    med = x.rolling(window, min_periods=window).median()
    mad = (x - med).abs().rolling(window, min_periods=window).median()
    return mad


def _rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    """滚动 Z-score 标准化；标准差为 0 时返回 NaN 避免除零。"""
    mu = x.rolling(window, min_periods=window).mean()
    sd = x.rolling(window, min_periods=window).std()
    return (x - mu) / sd.replace(0, np.nan)


def build_features_and_labels(
    df: pd.DataFrame,
    config: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.Series, FeatureMeta]:
    """
    Dpoint_t = P(close_{t+1} > close_t | X_t)
    All features are computed using info <= t (no leakage).
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    windows: List[int] = list(config.get("windows", [3, 5, 10, 20]))

    # feature families
    use_momentum: bool = bool(config.get("use_momentum", True))
    use_volatility: bool = bool(config.get("use_volatility", True))
    use_volume: bool = bool(config.get("use_volume", True))
    use_candle: bool = bool(config.get("use_candle", True))
    use_turnover: bool = bool(config.get("use_turnover", True))

    # parameterized transforms
    vol_metric: str = str(config.get("vol_metric", "std")).lower()  # std|mad
    liq_transform: str = str(config.get("liq_transform", "ratio")).lower()  # ratio|zscore

    close = df["close_qfq"].astype(float)
    open_ = df["open_qfq"].astype(float)
    high = df["high_qfq"].astype(float)
    low = df["low_qfq"].astype(float)
    volume = df["volume"].astype(float)
    amount = df["amount"].astype(float)
    turnover = df["turnover_rate"].astype(float)

    feats: Dict[str, pd.Series] = {}

    # base return
    ret1 = close.pct_change(1)
    feats["ret_1"] = ret1

    if use_momentum:
        for k in windows:
            feats[f"ret_{k}"] = close.pct_change(k)
            ma = close.rolling(k, min_periods=k).mean()
            feats[f"ma_{k}_ratio"] = close / ma - 1.0

    if use_volatility:
        feats["hl_range"] = (high - low) / close.replace(0, np.nan)

        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        feats["true_range_norm"] = tr / close.replace(0, np.nan)

        for k in windows:
            if vol_metric == "mad":
                feats[f"vol_mad_{k}"] = _rolling_mad(ret1, k)
            else:
                feats[f"vol_std_{k}"] = ret1.rolling(k, min_periods=k).std()

    if use_volume:
        feats["log_volume"] = _safe_log1p(volume)
        feats["log_amount"] = _safe_log1p(amount)
        for k in windows:
            if liq_transform == "zscore":
                feats[f"volume_z_{k}"] = _rolling_zscore(volume, k)
                feats[f"amount_z_{k}"] = _rolling_zscore(amount, k)
            else:
                vma = volume.rolling(k, min_periods=k).mean()
                ama = amount.rolling(k, min_periods=k).mean()
                feats[f"volume_ma_{k}_ratio"] = volume / vma.replace(0, np.nan)
                feats[f"amount_ma_{k}_ratio"] = amount / ama.replace(0, np.nan)

    if use_turnover:
        feats["turnover"] = turnover
        for k in windows:
            if liq_transform == "zscore":
                feats[f"turnover_z_{k}"] = _rolling_zscore(turnover, k)
            else:
                feats[f"turnover_ma_{k}"] = turnover.rolling(k, min_periods=k).mean()
                feats[f"turnover_std_{k}"] = turnover.rolling(k, min_periods=k).std()

    if use_candle:
        feats["body"] = (close - open_) / open_.replace(0, np.nan)
        feats["upper_shadow"] = (high - np.maximum(open_, close)) / close.replace(0, np.nan)
        feats["lower_shadow"] = (np.minimum(open_, close) - low) / close.replace(0, np.nan)

    X = pd.DataFrame(feats)

    # label: next day up or not
    # # 先计算差值，保留 NaN（最后一行 shift(-1) 为 NaN）
    y_diff = close.shift(-1) - close        # 最后一行为 NaN，语义清晰
    y = (y_diff > 0).astype("Int64")        # 用 nullable int，NaN 行保持为 pd.NA

    # 过滤条件：X 所有特征非空 & y 有真实标签（即排除最后一行）
    valid = X.notna().all(axis=1) & y_diff.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].astype(int).copy()    # 转回普通 int 供 sklearn 使用

    X.index = df.loc[X.index, "date"].values
    y.index = X.index

    meta = FeatureMeta(
        feature_names=list(X.columns),
        params={
            "windows": windows,
            "use_momentum": use_momentum,
            "use_volatility": use_volatility,
            "use_volume": use_volume,
            "use_candle": use_candle,
            "use_turnover": use_turnover,
            "vol_metric": vol_metric,
            "liq_transform": liq_transform,
        },
        dpoint_explainer=(
            "Dpoint_t = P(close_{t+1} > close_t | X_t). "
            "X_t is built from OHLCV/amount/turnover data up to t only (no future leakage)."
        ),
    )
    return X, y, meta