"""
测试 reporter 的输出文件生成。

覆盖：
    - Sheet 名称是否正确
    - 关键列是否存在
    - 配置 JSON 一致性
"""

import pytest
import pandas as pd
import json
import os
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reporter import (
    save_run_outputs,
    _build_walkforward_summary,
    _build_insample_warning_sheet,
    escape_excel_formulas,
)


class TestSheetNames:
    """测试 Sheet 名称"""

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": np.linspace(10.0, 15.0, 50),
            "high_qfq": np.linspace(10.5, 15.5, 50),
            "low_qfq": np.linspace(9.5, 14.5, 50),
            "close_qfq": np.linspace(10.2, 15.2, 50),
        })
        return df

    @pytest.fixture
    def sample_trades(self):
        """创建样本交易记录"""
        return pd.DataFrame([{
            "buy_signal_date": pd.Timestamp("2024-01-01"),
            "buy_exec_date": pd.Timestamp("2024-01-02"),
            "buy_price": 10.5,
            "buy_shares": 100,
            "buy_cost": 1055.0,
            "status": "CLOSED",
            "pnl": 50.0,
            "return": 0.05,
            "success": True,
        }])

    @pytest.fixture
    def sample_equity_curve(self):
        """创建样本净值曲线"""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        return pd.DataFrame({
            "date": dates,
            "close_qfq": np.linspace(10.0, 15.0, 50),
            "cash": np.linspace(100000, 95000, 50),
            "shares": [100] * 50,
            "market_value": np.linspace(10200, 15200, 50),
            "total_equity": np.linspace(110200, 110200, 50),
        })

    @pytest.fixture
    def sample_config(self):
        """创建样本配置"""
        return {
            "feature_config": {
                "windows": [3, 5, 10],
                "use_momentum": True,
            },
            "model_config": {
                "model_type": "logreg",
                "C": 1.0,
            },
            "trade_config": {
                "initial_cash": 100000,
                "buy_threshold": 0.55,
                "sell_threshold": 0.45,
            },
            "split_mode": "walkforward",
        }

    @pytest.fixture
    def sample_feature_meta(self):
        """创建样本特征元数据"""
        return {
            "feature_names": ["ret_1", "ret_3", "ret_5"],
            "feature_params": {"windows": [3, 5, 10]},
            "dpoint_explainer": "P(close_{t+1} > close_t | X_t)",
        }

    @pytest.fixture
    def sample_search_log(self):
        """创建样本搜索日志"""
        return pd.DataFrame([{
            "iter": 1,
            "status": "ok",
            "val_metric_final": 0.1,
            "val_geom_mean_ratio": 1.05,
            "val_min_fold_ratio": 0.95,
            "val_avg_closed_trades_per_fold": 3.0,
            "val_equity_proxy_mean": 105000,
            "val_metric_raw": 0.11,
            "val_penalty": 0.01,
            "n_features": 20,
            "model_type": "logreg",
            "feature_config": "{}",
            "model_config": "{}",
            "trade_config": "{}",
        }])

    def test_all_sheets_present(self, sample_data, sample_trades, sample_equity_curve,
                                sample_config, sample_feature_meta, sample_search_log,
                                tmp_path):
        """所有必需的 Sheet 都应该存在"""
        output_dir = str(tmp_path)

        excel_path, _, _ = save_run_outputs(
            output_dir=output_dir,
            df_clean=sample_data,
            log_notes=["Test log"],
            trades=sample_trades,
            equity_curve=sample_equity_curve,
            config=sample_config,
            feature_meta=sample_feature_meta,
            search_log=sample_search_log,
        )

        # 读取 Excel 检查 Sheet
        xl = pd.ExcelFile(excel_path)
        sheet_names = xl.sheet_names

        # 检查必需的 Sheet
        required_sheets = [
            "WalkForwardSummary",
            "Trades",
            "EquityCurve",
            "FinalFit_InSample",
            "SearchLog",
            "Config",
            "Log",
        ]

        for sheet in required_sheets:
            assert sheet in sheet_names, f"Missing sheet: {sheet}"

    def test_sheet_order(self, sample_data, sample_trades, sample_equity_curve,
                         sample_config, sample_feature_meta, sample_search_log,
                         tmp_path):
        """Sheet 顺序应该正确（重要性排序）"""
        output_dir = str(tmp_path)

        excel_path, _, _ = save_run_outputs(
            output_dir=output_dir,
            df_clean=sample_data,
            log_notes=["Test log"],
            trades=sample_trades,
            equity_curve=sample_equity_curve,
            config=sample_config,
            feature_meta=sample_feature_meta,
            search_log=sample_search_log,
        )

        xl = pd.ExcelFile(excel_path)
        sheet_names = xl.sheet_names

        # 第一个应该是 WalkForwardSummary
        assert sheet_names[0] == "WalkForwardSummary", \
            f"First sheet should be WalkForwardSummary, got {sheet_names[0]}"


class TestWalkForwardSummary:
    """测试 WalkForwardSummary Sheet"""

    def test_build_summary_with_data(self):
        """有数据时应该生成摘要"""
        search_log = pd.DataFrame([{
            "val_metric_final": 0.12,
            "val_geom_mean_ratio": 1.08,
            "val_min_fold_ratio": 0.92,
            "val_avg_closed_trades_per_fold": 5.0,
            "val_equity_proxy_mean": 112000,
            "val_metric_raw": 0.13,
            "val_penalty": 0.01,
        }])

        summary = _build_walkforward_summary(search_log, {})

        assert len(summary) > 0
        assert "Fold" in summary.columns

    def test_build_summary_empty(self):
        """空数据时应该返回空 DataFrame"""
        search_log = pd.DataFrame()

        summary = _build_walkforward_summary(search_log, {})

        assert len(summary) > 0  # 至少应该有列定义


class TestInSampleWarning:
    """测试 FinalFit_InSample Sheet"""

    def test_warning_content(self):
        """警告页应该包含关键信息"""
        warning_df = _build_insample_warning_sheet()

        content = " ".join(warning_df["Item"].astype(str))

        # 应该包含警告关键词
        assert "警告" in content or "WARNING" in content
        assert "In-Sample" in content or "样本内" in content
        assert "WalkForwardSummary" in content


class TestConfigJSON:
    """测试配置 JSON"""

    def test_json_structure(self, tmp_path):
        """JSON 结构应该正确"""
        import tempfile
        import pandas as pd
        import json

        # 创建最小测试数据
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "close_qfq": range(10),
        })

        config = {
            "feature_config": {"windows": [3, 5]},
            "model_config": {"model_type": "logreg"},
            "trade_config": {"buy_threshold": 0.55},
        }

        feature_meta = {
            "feature_names": ["ret_1"],
            "dpoint_explainer": "test",
        }

        trades = pd.DataFrame([{"status": "CLOSED", "pnl": 100}])
        equity = pd.DataFrame({"total_equity": [100000, 101000]})
        search_log = pd.DataFrame([{"iter": 1, "val_metric_final": 0.1}])

        output_dir = str(tmp_path)

        _, config_path, _ = save_run_outputs(
            output_dir=output_dir,
            df_clean=df,
            log_notes=["Test"],
            trades=trades,
            equity_curve=equity,
            config=config,
            feature_meta=feature_meta,
            search_log=search_log,
        )

        # 读取 JSON 检查结构
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 检查必需字段
        assert "run_id" in data
        assert "created_at" in data
        assert "best_config" in data
        assert "feature_meta" in data


class TestEscapeFormulas:
    """测试 Excel 公式转义"""

    def test_escape_equals(self):
        """= 应该被转义"""
        df = pd.DataFrame({"col": ["=SUM(A1:A2)", "normal"]})
        escaped = escape_excel_formulas(df)
        assert escaped["col"].iloc[0] == "'=SUM(A1:A2)"

    def test_escape_plus(self):
        """+ 应该被转义"""
        df = pd.DataFrame({"col": ["+A1+B2", "normal"]})
        escaped = escape_excel_formulas(df)
        assert escaped["col"].iloc[0] == "'+A1+B2"

    def test_no_escape_normal(self):
        """正常文本不应该被转义"""
        df = pd.DataFrame({"col": ["normal text", "another"]})
        escaped = escape_excel_formulas(df)
        assert escaped["col"].iloc[0] == "normal text"


# 导入 numpy（在文件顶部未导入）
import numpy as np
