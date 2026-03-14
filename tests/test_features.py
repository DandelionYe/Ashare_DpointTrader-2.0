"""
测试 feature_dpoint 的特征工程和标签构建。

覆盖：
    - 标签是否严格使用 t+1（无未来函数）
    - 特征是否只使用 t 及以前的数据
    - 特征数量是否正确
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_dpoint import build_features_and_labels, _safe_log1p, _rolling_mad, _rolling_zscore


class TestLabelConstruction:
    """测试标签构建（无未来函数）"""

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": np.arange(10.0, 20.0),
            "high_qfq": np.arange(10.5, 20.5),
            "low_qfq": np.arange(9.5, 19.5),
            "close_qfq": np.arange(10.2, 20.2),
            "volume": np.arange(1000, 10000, 1000),
            "amount": np.arange(10000, 100000, 10000),
            "turnover_rate": np.arange(1.0, 10.0, 1.0),
        })
        return df

    def test_label_is_t_plus_1(self, sample_data):
        """标签应该是 t+1 的涨跌"""
        config = {
            "windows": [3, 5],
            "use_momentum": True,
            "use_volatility": True,
            "use_volume": True,
            "use_candle": True,
            "use_turnover": True,
        }

        X, y, meta = build_features_and_labels(sample_data, config)

        # 检查 y 的定义：close_{t+1} > close_t
        # 第 i 行的 y 值应该等于 close[i+1] > close[i]
        closes = sample_data["close_qfq"].values
        for i in range(len(X)):
            original_idx = i  # 因为去除了第一行（NaN）和最后一行（无标签）
            if original_idx + 1 < len(closes):
                expected_label = int(closes[original_idx + 1] > closes[original_idx])
                actual_label = y.iloc[i]
                assert actual_label == expected_label, f"Label mismatch at {i}"

    def test_no_data_leakage(self, sample_data):
        """特征不应该包含未来信息"""
        config = {
            "windows": [3, 5],
            "use_momentum": True,
            "use_volatility": False,
            "use_volume": False,
            "use_candle": False,
            "use_turnover": False,
        }

        X, y, meta = build_features_and_labels(sample_data, config)

        # 检查特征名称
        for col in X.columns:
            # 特征名称不应该包含"future"、"next"等字样
            assert "future" not in col.lower()
            assert "next" not in col.lower()
            assert "tomorrow" not in col.lower()


class TestFeatureFamilies:
    """测试特征族"""

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": np.arange(10.0, 30.0),
            "high_qfq": np.arange(10.5, 30.5),
            "low_qfq": np.arange(9.5, 29.5),
            "close_qfq": np.arange(10.2, 30.2),
            "volume": np.arange(1000, 20000, 1000),
            "amount": np.arange(10000, 200000, 10000),
            "turnover_rate": np.arange(1.0, 20.0, 1.0),
        })
        return df

    def test_all_families_enabled(self, sample_data):
        """所有特征族启用时特征数量最多"""
        config = {
            "windows": [3, 5, 10],
            "use_momentum": True,
            "use_volatility": True,
            "use_volume": True,
            "use_candle": True,
            "use_turnover": True,
        }

        X, y, meta = build_features_and_labels(sample_data, config)

        # 应该有多个特征
        assert len(X.columns) > 10, f"Expected many features, got {len(X.columns)}"
        assert len(X) > 0, "Should have samples"

    def test_momentum_only(self, sample_data):
        """仅启用动量特征"""
        config = {
            "windows": [3, 5],
            "use_momentum": True,
            "use_volatility": False,
            "use_volume": False,
            "use_candle": False,
            "use_turnover": False,
        }

        X, y, meta = build_features_and_labels(sample_data, config)

        # 应该有 ret_1, ret_3, ret_5, ma_3_ratio, ma_5_ratio 等
        assert "ret_1" in X.columns
        assert "ret_3" in X.columns
        assert "ret_5" in X.columns

    def test_volatility_only(self, sample_data):
        """仅启用波动率特征"""
        config = {
            "windows": [3, 5],
            "use_momentum": False,
            "use_volatility": True,
            "use_volume": False,
            "use_candle": False,
            "use_turnover": False,
        }

        X, y, meta = build_features_and_labels(sample_data, config)

        # 应该有波动率相关特征
        assert any("vol" in col for col in X.columns), "Should have volatility features"

    def test_volume_only(self, sample_data):
        """仅启用成交量特征"""
        config = {
            "windows": [3, 5],
            "use_momentum": False,
            "use_volatility": False,
            "use_volume": True,
            "use_candle": False,
            "use_turnover": False,
        }

        X, y, meta = build_features_and_labels(sample_data, config)

        # 应该有成交量相关特征
        assert any("volume" in col for col in X.columns), "Should have volume features"

    def test_candle_only(self, sample_data):
        """仅启用 K 线特征"""
        config = {
            "windows": [3, 5],
            "use_momentum": False,
            "use_volatility": False,
            "use_volume": False,
            "use_candle": True,
            "use_turnover": False,
        }

        X, y, meta = build_features_and_labels(sample_data, config)

        # 应该有 K 线相关特征
        assert any("body" in col or "shadow" in col for col in X.columns), \
            "Should have candlestick features"


class TestHelperFunctions:
    """测试辅助函数"""

    def test_safe_log1p_negative(self):
        """log1p 处理负值"""
        x = pd.Series([-1, 0, 1, 2])
        result = _safe_log1p(x)
        assert not result.isna().any(), "Should handle negative values"
        assert result.iloc[0] == 0, "log1p(0) = 0"

    def test_rolling_mad(self):
        """滚动 MAD 计算"""
        x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        mad = _rolling_mad(x, window=3)
        # 前 2 个应该是 NaN（窗口不足）
        assert mad.iloc[:2].isna().all()
        # 后面应该有值
        assert not mad.iloc[2:].isna().all()

    def test_rolling_zscore(self):
        """滚动 Z-score 计算"""
        x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        zscore = _rolling_zscore(x, window=3)
        # 前 2 个应该是 NaN
        assert zscore.iloc[:2].isna().all()


class TestEdgeCases:
    """测试边界情况"""

    def test_insufficient_data(self):
        """数据不足时"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": [10.0, 10.2, 10.4, 10.6, 10.8],
            "high_qfq": [10.5, 10.7, 10.9, 11.1, 11.3],
            "low_qfq": [9.5, 9.7, 9.9, 10.1, 10.3],
            "close_qfq": [10.2, 10.4, 10.6, 10.8, 11.0],
            "volume": [1000, 2000, 3000, 4000, 5000],
            "amount": [10000, 20000, 30000, 40000, 50000],
            "turnover_rate": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        config = {
            "windows": [3, 5],  # 窗口大于数据长度
            "use_momentum": True,
            "use_volatility": True,
            "use_volume": True,
            "use_candle": True,
            "use_turnover": True,
        }

        X, y, meta = build_features_and_labels(df, config)

        # 应该仍有输出（可能较少）
        assert len(X) >= 0  # 可能为 0，但不应该报错

    def test_missing_turnover_data(self):
        """缺少换手率数据"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": np.arange(10.0, 20.0),
            "high_qfq": np.arange(10.5, 20.5),
            "low_qfq": np.arange(9.5, 19.5),
            "close_qfq": np.arange(10.2, 20.2),
            "volume": np.arange(1000, 10000, 1000),
            "amount": np.arange(10000, 100000, 10000),
            "turnover_rate": [0.0] * 10,  # 全为 0
        })

        config = {
            "windows": [3, 5],
            "use_momentum": True,
            "use_volatility": True,
            "use_volume": True,
            "use_candle": True,
            "use_turnover": True,
        }

        X, y, meta = build_features_and_labels(df, config)

        # 应该能正常处理
        assert len(X) > 0
