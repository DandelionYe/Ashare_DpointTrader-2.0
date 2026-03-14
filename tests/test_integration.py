"""
最小集成测试：使用示例数据跑一次完整流程。

覆盖：
    - 数据加载
    - 特征构建
    - 模型训练（1 次迭代）
    - 回测执行
    - 报告生成
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def sample_data(self):
        """创建最小样本数据（足够跑一次）"""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": 10.0 + (dates.dayofweek * 0.1),
            "high_qfq": 10.5 + (dates.dayofweek * 0.1),
            "low_qfq": 9.5 + (dates.dayofweek * 0.1),
            "close_qfq": 10.2 + (dates.dayofweek * 0.1),
            "volume": 1000000 + dates.day * 10000,
            "amount": 10000000 + dates.day * 100000,
            "turnover_rate": 2.0 + dates.dayofweek * 0.1,
        })
        return df

    def test_data_loading(self, sample_data):
        """数据加载应该成功"""
        from data_loader import load_stock_excel
        import io

        # 将数据保存为临时 Excel 文件
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name
            sample_data.to_excel(f, index=False)

        try:
            df_clean, report = load_stock_excel(temp_path)

            assert len(df_clean) > 0, "Should load data"
            assert report.rows_raw == 100, f"Expected 100 raw rows, got {report.rows_raw}"
        finally:
            os.unlink(temp_path)

    def test_feature_engineering(self, sample_data):
        """特征工程应该成功"""
        from feature_dpoint import build_features_and_labels

        config = {
            "windows": [3, 5],
            "use_momentum": True,
            "use_volatility": True,
            "use_volume": True,
            "use_candle": True,
            "use_turnover": True,
        }

        X, y, meta = build_features_and_labels(sample_data, config)

        assert len(X) > 0, "Should have samples"
        assert len(X.columns) > 0, "Should have features"
        assert len(y) == len(X), "X and y should have same length"

    def test_model_training(self, sample_data):
        """模型训练应该成功（1 次迭代）"""
        from feature_dpoint import build_features_and_labels
        from model_builder import make_model, predict_dpoint

        config = {
            "feature_config": {
                "windows": [3, 5],
                "use_momentum": True,
                "use_volatility": False,
                "use_volume": False,
                "use_candle": False,
                "use_turnover": False,
            },
            "model_config": {
                "model_type": "logreg",
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "class_weight": None,
            },
            "trade_config": {
                "initial_cash": 100000,
                "buy_threshold": 0.55,
                "sell_threshold": 0.45,
            },
        }

        X, y, meta = build_features_and_labels(sample_data, config["feature_config"])

        # 训练模型
        model = make_model(config, seed=42)
        model.fit(X.values, y.values)

        # 预测
        dpoint = predict_dpoint(model, X)

        assert len(dpoint) == len(X)
        assert dpoint.between(0, 1).all(), "Dpoint should be probability"

    def test_backtest_execution(self, sample_data):
        """回测执行应该成功"""
        from feature_dpoint import build_features_and_labels
        from model_builder import make_model, predict_dpoint
        from backtester_engine import backtest_from_dpoint

        # 简化配置
        config = {
            "windows": [3],
            "use_momentum": True,
            "use_volatility": False,
            "use_volume": False,
            "use_candle": False,
            "use_turnover": False,
        }

        X, y, _ = build_features_and_labels(sample_data, config)

        # 训练简单模型
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, solver="lbfgs"))
        ])
        model.fit(X.values, y.values)

        dpoint = predict_dpoint(model, X)

        # 运行回测
        bt = backtest_from_dpoint(
            df=sample_data,
            dpoint=dpoint,
            initial_cash=100000,
            buy_threshold=0.55,
            sell_threshold=0.45,
            confirm_days=2,
            min_hold_days=1,
            exec_price_model="next_open",
            slippage_bps=10,
        )

        # 检查结果
        assert hasattr(bt, "trades")
        assert hasattr(bt, "equity_curve")
        assert hasattr(bt, "notes")
        assert len(bt.equity_curve) > 0

    def test_report_generation(self, sample_data, tmp_path):
        """报告生成应该成功"""
        from reporter import save_run_outputs

        # 创建最小必需数据
        trades = pd.DataFrame([{
            "buy_signal_date": pd.Timestamp("2023-01-02"),
            "buy_exec_date": pd.Timestamp("2023-01-03"),
            "buy_price": 10.5,
            "buy_shares": 100,
            "status": "CLOSED",
            "pnl": 50.0,
        }])

        equity_curve = pd.DataFrame({
            "date": sample_data["date"],
            "total_equity": [100000 + i * 100 for i in range(len(sample_data))],
        })

        config = {
            "feature_config": {"windows": [3]},
            "model_config": {"model_type": "logreg"},
            "trade_config": {"buy_threshold": 0.55},
        }

        feature_meta = {
            "feature_names": ["ret_1"],
            "dpoint_explainer": "test",
        }

        search_log = pd.DataFrame([{
            "iter": 1,
            "val_metric_final": 0.1,
            "val_geom_mean_ratio": 1.05,
        }])

        output_dir = str(tmp_path)

        excel_path, config_path, run_id = save_run_outputs(
            output_dir=output_dir,
            df_clean=sample_data,
            log_notes=["Integration test"],
            trades=trades,
            equity_curve=equity_curve,
            config=config,
            feature_meta=feature_meta,
            search_log=search_log,
        )

        # 检查文件生成
        assert os.path.exists(excel_path)
        assert os.path.exists(config_path)
        assert run_id == 1

        # 检查 Excel 内容
        xl = pd.ExcelFile(excel_path)
        assert "WalkForwardSummary" in xl.sheet_names
        assert "Trades" in xl.sheet_names

    def test_full_pipeline_mini(self, sample_data, tmp_path):
        """最小完整流程（简化版 random_search_train）"""
        from feature_dpoint import build_features_and_labels
        from model_builder import make_model, predict_dpoint
        from backtester_engine import backtest_from_dpoint
        from reporter import save_run_outputs

        # 1. 特征工程
        config = {
            "windows": [3],
            "use_momentum": True,
            "use_volatility": False,
            "use_volume": False,
            "use_candle": False,
            "use_turnover": False,
        }
        X, y, meta = build_features_and_labels(sample_data, config)

        # 2. 模型训练
        model_config = {
            "model_type": "logreg",
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
        }
        full_config = {
            "feature_config": config,
            "model_config": model_config,
            "trade_config": {
                "initial_cash": 100000,
                "buy_threshold": 0.55,
                "sell_threshold": 0.45,
                "confirm_days": 2,
                "min_hold_days": 1,
            },
        }

        model = make_model({"model_config": model_config}, seed=42)
        model.fit(X.values, y.values)

        # 3. 预测
        dpoint = predict_dpoint(model, X)

        # 4. 回测
        bt = backtest_from_dpoint(
            df=sample_data,
            dpoint=dpoint,
            initial_cash=100000,
            buy_threshold=0.55,
            sell_threshold=0.45,
            confirm_days=2,
            min_hold_days=1,
            exec_price_model="next_open",
            slippage_bps=10,
        )

        # 5. 报告生成
        trades = bt.trades if bt.trades is not None else pd.DataFrame()
        equity_curve = bt.equity_curve if bt.equity_curve is not None else pd.DataFrame()

        output_dir = str(tmp_path)

        excel_path, config_path, run_id = save_run_outputs(
            output_dir=output_dir,
            df_clean=sample_data,
            log_notes=["Mini pipeline test"],
            trades=trades,
            equity_curve=equity_curve,
            config=full_config,
            feature_meta={
                "feature_names": list(X.columns),
                "feature_params": config,
                "dpoint_explainer": meta.dpoint_explainer,
            },
            search_log=pd.DataFrame([{
                "iter": 1,
                "val_metric_final": 0.1,
                "val_geom_mean_ratio": 1.05,
                "n_features": len(X.columns),
                "model_type": "logreg",
            }]),
        )

        # 验证
        assert os.path.exists(excel_path)
        assert os.path.exists(config_path)

        # 验证 Excel 内容
        xl = pd.ExcelFile(excel_path)
        sheets = xl.sheet_names
        assert "WalkForwardSummary" in sheets
        assert "Trades" in sheets or len(trades) == 0  # 可能没有交易
