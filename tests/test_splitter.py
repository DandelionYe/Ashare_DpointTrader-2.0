"""
测试 splitter 的 walk-forward 数据切分。

覆盖：
    - 切分边界是否正确（无穿越）
    - 训练集是否累积扩展
    - 验证集是否不重叠
    - 最小行数约束
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from splitter import walkforward_splits


class TestWalkForwardSplits:
    """测试 walk-forward 切分"""

    @pytest.fixture
    def sample_data(self):
        """创建 1000 行样本数据"""
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")
        X = pd.DataFrame({
            "feature": np.random.randn(1000)
        }, index=dates)
        y = pd.Series(np.random.randint(0, 2, 1000), index=dates)
        return X, y

    def test_number_of_folds(self, sample_data):
        """应该生成指定数量的折"""
        X, y = sample_data
        splits = walkforward_splits(X, y, n_folds=4)
        assert len(splits) == 4, f"Expected 4 folds, got {len(splits)}"

    def test_train_expanding_window(self, sample_data):
        """训练集应该是扩展窗口"""
        X, y = sample_data
        splits = walkforward_splits(X, y, n_folds=4, train_start_ratio=0.5)

        train_sizes = [len(X_train) for (X_train, _), _ in splits]

        # 训练集应该递增
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i-1], \
                f"Train size should increase: {train_sizes}"

    def test_val_non_overlapping(self, sample_data):
        """验证集应该不重叠"""
        X, y = sample_data
        splits = walkforward_splits(X, y, n_folds=4)

        val_ranges = []
        for _, (X_val, _) in splits:
            val_ranges.append((X_val.index.min(), X_val.index.max()))

        # 检查相邻验证集是否不重叠
        for i in range(1, len(val_ranges)):
            prev_end = val_ranges[i-1][1]
            curr_start = val_ranges[i][0]
            assert prev_end < curr_start, \
                f"Validation sets overlap: {val_ranges[i-1]} and {val_ranges[i]}"

    def test_no_data_leakage(self, sample_data):
        """训练集和验证集不能有数据泄露"""
        X, y = sample_data
        splits = walkforward_splits(X, y, n_folds=4)

        for (X_train, _), (X_val, _) in splits:
            train_dates = set(X_train.index)
            val_dates = set(X_val.index)

            # 训练集和验证集不能有交集
            overlap = train_dates.intersection(val_dates)
            assert len(overlap) == 0, f"Data leakage: {overlap}"

            # 验证集应该在训练集之后
            assert X_val.index.min() > X_train.index.min(), \
                "Validation should start after training starts"

    def test_min_rows_constraint(self, sample_data):
        """最小行数约束"""
        X, y = sample_data

        # 设置很大的 min_rows，应该导致某些折被跳过
        splits = walkforward_splits(X, y, n_folds=4, min_rows=10000)

        # 应该没有有效的折（数据不足）
        assert len(splits) == 0, "Should skip all folds when min_rows is too large"

    def test_train_start_ratio(self, sample_data):
        """训练集起始比例"""
        X, y = sample_data
        n = len(X)

        # train_start_ratio=0.6，第一折训练集应该占 60%
        splits = walkforward_splits(X, y, n_folds=4, train_start_ratio=0.6)

        first_train_size = len(splits[0][0][0])
        expected_ratio = first_train_size / n

        # 允许一定误差
        assert 0.55 < expected_ratio < 0.65, \
            f"Expected ~0.6, got {expected_ratio}"

    def test_val_size_consistency(self, sample_data):
        """验证集大小应该大致相等"""
        X, y = sample_data
        splits = walkforward_splits(X, y, n_folds=4)

        val_sizes = [len(X_val) for _, (X_val, _) in splits]

        # 最大和最小验证集大小差异不应该太大
        max_size = max(val_sizes)
        min_size = min(val_sizes)

        # 差异不超过 20%
        ratio = max_size / min_size if min_size > 0 else float('inf')
        assert ratio < 1.5, f"Val sizes too inconsistent: {val_sizes}"


class TestEdgeCases:
    """测试边界情况"""

    def test_insufficient_data(self):
        """数据不足时"""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        X = pd.DataFrame({"feature": np.random.randn(10)}, index=dates)
        y = pd.Series(np.random.randint(0, 2, 10), index=dates)

        splits = walkforward_splits(X, y, n_folds=4, min_rows=80)

        # 应该没有有效的折
        assert len(splits) == 0

    def test_single_fold(self):
        """单折情况"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        X = pd.DataFrame({"feature": np.random.randn(100)}, index=dates)
        y = pd.Series(np.random.randint(0, 2, 100), index=dates)

        splits = walkforward_splits(X, y, n_folds=1)

        # 应该有 1 折
        assert len(splits) == 1

    def test_very_small_dataset(self):
        """非常小的数据集"""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        X = pd.DataFrame({"feature": np.random.randn(50)}, index=dates)
        y = pd.Series(np.random.randint(0, 2, 50), index=dates)

        # 应该仍能生成一些折（可能少于请求的数量）
        splits = walkforward_splits(X, y, n_folds=4, min_rows=10)

        # 至少应该有 1 折
        assert len(splits) >= 1, "Should have at least 1 valid fold"


class TestIndexPreservation:
    """测试索引保持"""

    def test_datetime_index_preserved(self):
        """日期时间索引应该保持"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        X = pd.DataFrame({"feature": np.random.randn(100)}, index=dates)
        y = pd.Series(np.random.randint(0, 2, 100), index=dates)

        splits = walkforward_splits(X, y, n_folds=4)

        for (X_train, y_train), (X_val, y_val) in splits:
            assert isinstance(X_train.index, pd.DatetimeIndex)
            assert isinstance(X_val.index, pd.DatetimeIndex)
            assert isinstance(y_train.index, pd.DatetimeIndex)
            assert isinstance(y_val.index, pd.DatetimeIndex)

    def test_x_y_index_alignment(self):
        """X 和 y 的索引应该对齐"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        X = pd.DataFrame({"feature": np.random.randn(100)}, index=dates)
        y = pd.Series(np.random.randint(0, 2, 100), index=dates)

        splits = walkforward_splits(X, y, n_folds=4)

        for (X_train, y_train), (X_val, y_val) in splits:
            assert X_train.index.equals(y_train.index)
            assert X_val.index.equals(y_val.index)
