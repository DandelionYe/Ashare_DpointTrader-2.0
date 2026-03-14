# model_builder.py
"""
模型构建与预测。
支持 LogisticRegression、SGDClassifier（均含 StandardScaler Pipeline）和 XGBoost。
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _try_import_xgboost() -> Any:
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except Exception:
        return None


def make_model(candidate: Dict[str, Any], seed: int) -> Any:
    """
    根据 candidate["model_config"] 构建未拟合的 sklearn 兼容模型（含 predict_proba）。

    支持的 model_type:
        logreg  — LogisticRegression + StandardScaler Pipeline
        sgd     — SGDClassifier(log_loss) + StandardScaler Pipeline
        xgb     — XGBClassifier（需安装 xgboost）
    """
    model_type = str(candidate["model_config"]["model_type"])

    if model_type == "logreg":
        C = float(candidate["model_config"]["C"])
        penalty = str(candidate["model_config"]["penalty"])
        solver = str(candidate["model_config"]["solver"])
        class_weight = candidate["model_config"].get("class_weight", None)
        l1_ratio = candidate["model_config"].get("l1_ratio", None)
        clf = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            l1_ratio=l1_ratio,
            class_weight=class_weight,
            max_iter=8000,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_type == "sgd":
        alpha = float(candidate["model_config"]["alpha"])
        penalty = str(candidate["model_config"]["penalty"])
        l1_ratio = float(candidate["model_config"].get("l1_ratio", 0.15))
        class_weight = candidate["model_config"].get("class_weight", None)
        clf = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            penalty=penalty,
            l1_ratio=l1_ratio if penalty == "elasticnet" else None,
            class_weight=class_weight,
            max_iter=3000,
            tol=1e-3,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_type == "xgb":
        xgb = _try_import_xgboost()
        if xgb is None:
            raise RuntimeError("xgboost_not_installed")
        clf = xgb.XGBClassifier(**candidate["model_config"]["params"])
        return clf

    raise ValueError(f"Unknown model_type: {model_type}")


def predict_dpoint(model: Any, X: pd.DataFrame) -> pd.Series:
    """
    调用 predict_proba，返回 class=1 的概率作为 Dpoint Series（保留原始 index）。
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("model has no predict_proba")
    if isinstance(model, Pipeline):
        proba = model.predict_proba(X.values)[:, 1]
    else:
        proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=X.index, name="dpoint")
