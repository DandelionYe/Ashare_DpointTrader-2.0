# splitter.py
"""
Walk-forward 时序切分。
验证集不重叠，训练集累积扩展（expanding window）。
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def walkforward_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 4,
    train_start_ratio: float = 0.5,
    min_rows: int = 80,
) -> List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]:
    """
    生成 walk-forward 时序切分。

    参数说明：
        n_folds          : 验证折数，默认 4。
        train_start_ratio: 第一折训练集占全部数据的比例，默认 0.5。
        min_rows         : 训练集或验证集的最小行数，不足时跳过该折并打印警告。

    切分示意（n_folds=4, train_start_ratio=0.5）：
        折1: train=[0%~50%]   val=[50%~62.5%]
        折2: train=[0%~62%]   val=[62.5%~75%]
        折3: train=[0%~75%]   val=[75%~87.5%]
        折4: train=[0%~87%]   val=[87.5%~100%]
        （共 n_folds = 4 个验证折，首段 0~50% 数据仅用作初始训练集）

    注意：验证集不重叠，训练集累积扩展。
    """
    n = len(X)
    cuts = [
        train_start_ratio + (1.0 - train_start_ratio) * i / n_folds
        for i in range(n_folds + 1)
    ]

    splits = []
    for k in range(len(cuts) - 1):
        train_end = int(n * cuts[k])
        val_end = int(n * cuts[k + 1])
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

        if len(X_train) < min_rows or len(X_val) < min_rows:
            print(
                f"[WARN] walkforward_splits: fold {k + 1} skipped "
                f"(train={len(X_train)}, val={len(X_val)}, min_rows={min_rows}). "
                f"Consider reducing n_folds or min_rows."
            )
            continue
        splits.append(((X_train, y_train), (X_val, y_val)))

    if not splits:
        print(
            f"[WARN] walkforward_splits: ALL {n_folds} folds skipped. "
            f"Total rows={n}, train_start_ratio={train_start_ratio}, min_rows={min_rows}."
        )
    return splits
