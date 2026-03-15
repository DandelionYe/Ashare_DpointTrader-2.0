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
import json
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

        # 新 API 需要三个配置对象
        best_strategy_config = {
            "feature_config": {"windows": [3]},
            "model_config": {"model_type": "logreg"},
            "trade_config": {"buy_threshold": 0.55},
            "split_mode": "walkforward",
        }

        repro_config = {
            "feature_config": {"windows": [3]},
            "model_config": {"model_type": "logreg"},
            "trade_config": {
                "buy_threshold": 0.55,
                "exec_price_model": "next_open",
                "slippage_bps": 10.0,
                "commission_rate": 0.00025,
                "commission_min": 5.0,
            },
            "search_config": {
                "runs": 100,
                "epsilon": 0.01,
                "exploit_ratio": 0.7,
                "top_k": 10,
                "max_features": 80,
                "n_jobs": -1,
            },
            "split_mode": "walkforward",
            "n_folds": 4,
            "train_start_ratio": 0.5,
            "wf_min_rows": 80,
        }

        run_context = {
            "mode": "first",
            "base_seed": 42,
            "search_seed": 42,
            "final_train_seed": 42,
            "effective_runs": 100,
            "config_source": "CLI/default",
            "git_commit": "test",
            "hostname": "test",
        }

        feature_meta = {
            "feature_names": ["ret_1"],
            "dpoint_explainer": "test",
        }

        search_log = pd.DataFrame([{
            "iter": 1,
            "val_metric_final": 0.1,
            "val_geom_mean_ratio": 1.05,
            "fold_details_json": "[]",
        }])

        output_dir = str(tmp_path)

        excel_path, config_path, run_id = save_run_outputs(
            output_dir=output_dir,
            df_clean=sample_data,
            log_notes=["Integration test"],
            trades=trades,
            equity_curve=equity_curve,
            best_strategy_config=best_strategy_config,
            repro_config=repro_config,
            run_context=run_context,
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

        # 新 API 需要三个配置对象
        best_strategy_config = {
            "feature_config": config,
            "model_config": model_config,
            "trade_config": full_config["trade_config"],
            "split_mode": "walkforward",
        }

        repro_config = {
            "feature_config": config,
            "model_config": model_config,
            "trade_config": {
                **full_config["trade_config"],
                "exec_price_model": "next_open",
                "slippage_bps": 10.0,
                "commission_rate": 0.00025,
                "commission_min": 5.0,
            },
            "search_config": {
                "runs": 100,
                "epsilon": 0.01,
                "exploit_ratio": 0.7,
                "top_k": 10,
                "max_features": 80,
                "n_jobs": -1,
            },
            "split_mode": "walkforward",
            "n_folds": 4,
            "train_start_ratio": 0.5,
            "wf_min_rows": 80,
        }

        run_context = {
            "mode": "first",
            "base_seed": 42,
            "search_seed": 42,
            "final_train_seed": 42,
            "effective_runs": 100,
            "config_source": "CLI/default",
            "git_commit": "test",
            "hostname": "test",
        }

        excel_path, config_path, run_id = save_run_outputs(
            output_dir=output_dir,
            df_clean=sample_data,
            log_notes=["Mini pipeline test"],
            trades=trades,
            equity_curve=equity_curve,
            best_strategy_config=best_strategy_config,
            repro_config=repro_config,
            run_context=run_context,
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
                "fold_details_json": "[]",
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

    def test_repro_config_roundtrip_preserves_search_split_and_exec_assumptions(self, sample_data, tmp_path):
        """
        场景：
            1. 用非默认配置运行一次（search_config.runs=17, n_folds=3, 非默认执行成本）
            2. 从生成的 config JSON 恢复 repro_config
            3. 断言第二次恢复出的 repro_config 完整一致
        """
        from reporter import save_run_outputs

        # 非默认配置
        best_strategy_config = {
            "feature_config": {
                "windows": [3, 5, 10],
                "use_momentum": True,
                "use_volatility": True,
                "use_volume": False,
                "use_candle": True,
                "use_turnover": True,
                "vol_metric": "mad",
                "liq_transform": "zscore",
            },
            "model_config": {
                "model_type": "logreg",
                "C": 0.5,
                "penalty": "l2",
                "solver": "liblinear",
                "class_weight": "balanced",
            },
            "trade_config": {
                "initial_cash": 200000.0,
                "buy_threshold": 0.58,
                "sell_threshold": 0.42,
                "confirm_days": 3,
                "min_hold_days": 2,
                "max_hold_days": 20,
            },
            "split_mode": "walkforward",
        }

        repro_config = {
            "feature_config": best_strategy_config["feature_config"],
            "model_config": best_strategy_config["model_config"],
            "trade_config": {
                **best_strategy_config["trade_config"],
                "exec_price_model": "next_close",
                "slippage_bps": 25.0,
                "commission_rate": 0.0003,
                "commission_min": 5.0,
            },
            "search_config": {
                "runs": 17,
                "epsilon": 0.015,
                "exploit_ratio": 0.65,
                "top_k": 8,
                "max_features": 70,
                "n_jobs": -1,
            },
            "split_mode": "walkforward",
            "n_folds": 3,
            "train_start_ratio": 0.6,
            "wf_min_rows": 120,
        }

        run_context = {
            "mode": "first",
            "base_seed": 42,
            "search_seed": 42,
            "final_train_seed": 42,
            "effective_runs": 17,
            "config_source": "CLI/default",
            "git_commit": "test123",
            "hostname": "test-host",
        }

        trades = pd.DataFrame()
        equity_curve = pd.DataFrame({
            "date": sample_data["date"],
            "total_equity": [100000 + i * 100 for i in range(len(sample_data))],
        })

        feature_meta = {
            "feature_names": ["ret_3", "ret_5"],
            "feature_params": best_strategy_config["feature_config"],
            "dpoint_explainer": "P(close_{t+1} > close_t | X_t)",
        }

        search_log = pd.DataFrame([{
            "iter": 1,
            "val_metric_final": 0.1,
            "val_geom_mean_ratio": 1.05,
            "fold_details_json": "[]",
        }])

        output_dir = str(tmp_path)

        # 第一次运行
        excel_path, config_path, run_id = save_run_outputs(
            output_dir=output_dir,
            df_clean=sample_data,
            log_notes=["Roundtrip test"],
            trades=trades,
            equity_curve=equity_curve,
            best_strategy_config=best_strategy_config,
            repro_config=repro_config,
            run_context=run_context,
            feature_meta=feature_meta,
            search_log=search_log,
        )

        # 从 JSON 恢复
        with open(config_path, "r", encoding="utf-8") as f:
            config_json = json.load(f)

        restored_repro = config_json["repro_config"]

        # 断言配置完整一致
        assert restored_repro["search_config"]["runs"] == 17
        assert restored_repro["n_folds"] == 3
        assert restored_repro["train_start_ratio"] == 0.6
        assert restored_repro["wf_min_rows"] == 120
        assert restored_repro["trade_config"]["exec_price_model"] == "next_close"
        assert restored_repro["trade_config"]["slippage_bps"] == 25.0
        assert restored_repro["trade_config"]["commission_rate"] == 0.0003
        assert restored_repro["feature_config"]["windows"] == [3, 5, 10]
        assert restored_repro["model_config"]["C"] == 0.5

    def test_run_config_json_contains_repro_config_not_only_best_strategy(self, sample_data, tmp_path):
        """
        断言：
            1. run_xxx_config.json 里同时有 best_strategy_config 和 repro_config
            2. repro_config.search_config.runs 存在
            3. repro_config.trade_config.exec_price_model 存在
        """
        from reporter import save_run_outputs

        best_strategy_config = {
            "feature_config": {"windows": [3, 5]},
            "model_config": {"model_type": "logreg", "C": 1.0},
            "trade_config": {"buy_threshold": 0.55},
            "split_mode": "walkforward",
        }

        repro_config = {
            "feature_config": {"windows": [3, 5]},
            "model_config": {"model_type": "logreg", "C": 1.0},
            "trade_config": {
                "buy_threshold": 0.55,
                "exec_price_model": "next_close",
                "slippage_bps": 25.0,
                "commission_rate": 0.0003,
                "commission_min": 5.0,
            },
            "search_config": {
                "runs": 17,
                "epsilon": 0.01,
                "exploit_ratio": 0.7,
            },
            "split_mode": "walkforward",
            "n_folds": 3,
            "train_start_ratio": 0.6,
            "wf_min_rows": 120,
        }

        run_context = {
            "mode": "first",
            "base_seed": 42,
            "search_seed": 42,
            "final_train_seed": 42,
            "effective_runs": 17,
            "config_source": "CLI/default",
        }

        trades = pd.DataFrame()
        equity_curve = pd.DataFrame({
            "date": sample_data["date"],
            "total_equity": [100000 + i * 100 for i in range(len(sample_data))],
        })

        feature_meta = {
            "feature_names": ["ret_3"],
            "dpoint_explainer": "test",
        }

        search_log = pd.DataFrame([{
            "iter": 1,
            "val_metric_final": 0.1,
            "fold_details_json": "[]",
        }])

        output_dir = str(tmp_path)

        excel_path, config_path, run_id = save_run_outputs(
            output_dir=output_dir,
            df_clean=sample_data,
            log_notes=["JSON structure test"],
            trades=trades,
            equity_curve=equity_curve,
            best_strategy_config=best_strategy_config,
            repro_config=repro_config,
            run_context=run_context,
            feature_meta=feature_meta,
            search_log=search_log,
        )

        # 加载 JSON
        with open(config_path, "r", encoding="utf-8") as f:
            config_json = json.load(f)

        # 断言 1: 同时有 best_strategy_config 和 repro_config
        assert "best_strategy_config" in config_json
        assert "repro_config" in config_json
        assert "run_context" in config_json

        # 断言 2: repro_config.search_config.runs 存在
        assert "search_config" in config_json["repro_config"]
        assert config_json["repro_config"]["search_config"]["runs"] == 17

        # 断言 3: repro_config.trade_config.exec_price_model 存在
        assert config_json["repro_config"]["trade_config"]["exec_price_model"] == "next_close"
        assert config_json["repro_config"]["trade_config"]["slippage_bps"] == 25.0
        assert config_json["repro_config"]["trade_config"]["commission_rate"] == 0.0003

        # 断言 4: best_strategy_config 不包含 search_config
        assert "search_config" not in config_json["best_strategy_config"]
