"""
测试 reporter 的输出文件生成。

覆盖：
    - Sheet 名称是否正确
    - 关键列是否存在
    - 配置 JSON 一致性
    - fold_details_json 解析
    - RunContext sheet
    - ExecutionAssumptions sheet
"""

import pytest
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reporter import (
    save_run_outputs,
    _build_walkforward_summary,
    _build_insample_warning_sheet,
    _build_run_context,
    _build_execution_assumptions,
    escape_excel_formulas,
)


class TestSheetNames:
    """测试 Sheet 名称"""

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        return pd.DataFrame({
            "date": dates,
            "open_qfq": np.linspace(10.0, 15.0, 50),
            "high_qfq": np.linspace(10.5, 15.5, 50),
            "low_qfq": np.linspace(9.5, 14.5, 50),
            "close_qfq": np.linspace(10.2, 15.2, 50),
        })

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
    def sample_best_strategy_config(self):
        """创建样本策略配置"""
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
    def sample_repro_config(self):
        """创建样本复现配置"""
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
                "exec_price_model": "next_open",
                "slippage_bps": 10.0,
                "commission_rate": 0.00025,
                "commission_min": 5.0,
            },
            "search_config": {
                "runs": 100,
                "epsilon": 0.01,
                "exploit_ratio": 0.7,
            },
            "split_mode": "walkforward",
            "n_folds": 4,
            "train_start_ratio": 0.5,
            "wf_min_rows": 80,
        }

    @pytest.fixture
    def sample_run_context(self):
        """创建样本运行上下文"""
        return {
            "mode": "first",
            "base_seed": 42,
            "search_seed": 42,
            "final_train_seed": 42,
            "effective_runs": 100,
            "config_source": "CLI/default",
            "git_commit": "test123",
            "hostname": "test-host",
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
            "fold_details_json": "[]",
        }])

    def test_all_sheets_present(self, sample_data, sample_trades, sample_equity_curve,
                                sample_best_strategy_config, sample_repro_config,
                                sample_run_context, sample_feature_meta, sample_search_log,
                                tmp_path):
        """所有必需的 Sheet 都应该存在"""
        output_dir = str(tmp_path)

        excel_path, _, _ = save_run_outputs(
            output_dir=output_dir,
            df_clean=sample_data,
            log_notes=["Test log"],
            trades=sample_trades,
            equity_curve=sample_equity_curve,
            best_strategy_config=sample_best_strategy_config,
            repro_config=sample_repro_config,
            run_context=sample_run_context,
            feature_meta=sample_feature_meta,
            search_log=sample_search_log,
        )

        # 读取 Excel 检查 Sheet
        xl = pd.ExcelFile(excel_path)
        sheet_names = xl.sheet_names

        # 检查必需的 Sheet
        required_sheets = [
            "WalkForwardSummary",
            "RunContext",
            "ExecutionAssumptions",
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
                         sample_best_strategy_config, sample_repro_config,
                         sample_run_context, sample_feature_meta, sample_search_log,
                         tmp_path):
        """Sheet 顺序应该正确（重要性排序）"""
        output_dir = str(tmp_path)

        excel_path, _, _ = save_run_outputs(
            output_dir=output_dir,
            df_clean=sample_data,
            log_notes=["Test log"],
            trades=sample_trades,
            equity_curve=sample_equity_curve,
            best_strategy_config=sample_best_strategy_config,
            repro_config=sample_repro_config,
            run_context=sample_run_context,
            feature_meta=sample_feature_meta,
            search_log=sample_search_log,
        )

        xl = pd.ExcelFile(excel_path)
        sheet_names = xl.sheet_names

        # 第一个应该是 WalkForwardSummary
        assert sheet_names[0] == "WalkForwardSummary", \
            f"First sheet should be WalkForwardSummary, got {sheet_names[0]}"

        # 第二个应该是 RunContext
        assert sheet_names[1] == "RunContext", \
            f"Second sheet should be RunContext, got {sheet_names[1]}"

        # 第三个应该是 ExecutionAssumptions
        assert sheet_names[2] == "ExecutionAssumptions", \
            f"Third sheet should be ExecutionAssumptions, got {sheet_names[2]}"


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
            "fold_details_json": "[]",
        }])

        summary = _build_walkforward_summary(search_log, {})

        assert len(summary) > 0
        assert "Fold" in summary.columns

    def test_build_summary_prefers_fold_details_json(self):
        """
        测试 _build_walkforward_summary 优先解析 fold_details_json。
        
        构造：
            search_log 包含 fold_details_json 列，且有有效的逐折数据
        
        断言：
            生成的摘要包含逐折明细（Fold 1, Fold 2, ...）
        """
        fold_details = [
            {"fold_id": 0, "equity_ratio": 1.05, "n_closed": 3, "train_rows": 100, "val_rows": 50},
            {"fold_id": 1, "equity_ratio": 1.03, "n_closed": 4, "train_rows": 150, "val_rows": 50},
            {"fold_id": 2, "equity_ratio": 1.07, "n_closed": 5, "train_rows": 200, "val_rows": 50},
        ]
        
        search_log = pd.DataFrame([{
            "val_metric_final": 0.12,
            "val_geom_mean_ratio": 1.05,
            "val_min_fold_ratio": 0.95,
            "val_avg_closed_trades_per_fold": 4.0,
            "val_equity_proxy_mean": 112000,
            "val_metric_raw": 0.13,
            "val_penalty": 0.01,
            "fold_details_json": json.dumps(fold_details),
        }])

        summary = _build_walkforward_summary(search_log, {})

        # 应该包含逐折明细
        assert len(summary) > 1, "Should have multiple rows for fold details"
        
        # 检查是否包含 Fold 1, Fold 2 等
        fold_labels = summary["Fold"].astype(str).tolist()
        assert any("Fold 1" in f for f in fold_labels), "Should have Fold 1"
        assert any("Fold 2" in f for f in fold_labels), "Should have Fold 2"
        
        # 检查是否包含聚合行
        assert any("Overall" in f or "Aggregated" in f for f in fold_labels), \
            "Should have overall/aggregated row"

    def test_build_summary_fallback_without_fold_details(self):
        """
        测试没有 fold_details_json 时回退到旧逻辑。
        
        构造：
            search_log 没有 fold_details_json 列
        
        断言：
            生成的摘要使用旧格式（包含警告信息）
        """
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

        # 应该有行
        assert len(summary) > 0
        assert "Fold" in summary.columns
        
        # 应该包含警告信息
        fold_labels = " ".join(summary["Fold"].astype(str))
        assert "WARNING" in fold_labels or "Derived" in fold_labels, \
            "Should have warning about derived metrics"

    def test_build_summary_empty(self):
        """空数据时应该返回空 DataFrame"""
        search_log = pd.DataFrame()

        summary = _build_walkforward_summary(search_log, {})

        assert len(summary) == 0 or "Fold" in summary.columns


class TestRunContext:
    """测试 RunContext Sheet"""

    def test_build_run_context(self):
        """测试 RunContext 构建"""
        run_context = {
            "mode": "first",
            "base_seed": 42,
            "search_seed": 43,
            "final_train_seed": 42,
            "effective_runs": 100,
            "config_source": "CLI/default",
            "git_commit": "abc123",
            "hostname": "test-host",
        }
        
        repro_config = {
            "n_folds": 4,
            "train_start_ratio": 0.5,
            "wf_min_rows": 80,
        }

        ctx_df = _build_run_context(run_context, repro_config)

        assert len(ctx_df) > 0
        assert "Parameter" in ctx_df.columns
        assert "Value" in ctx_df.columns
        
        # 检查关键参数
        params = " ".join(ctx_df["Parameter"].astype(str))
        assert "mode" in params
        assert "base_seed" in params
        assert "search_seed" in params
        assert "effective_runs" in params
        assert "n_folds" in params


class TestExecutionAssumptions:
    """测试 ExecutionAssumptions Sheet"""

    def test_build_execution_assumptions(self):
        """测试 ExecutionAssumptions 构建"""
        repro_config = {
            "trade_config": {
                "exec_price_model": "next_open",
                "slippage_bps": 10.0,
                "commission_rate": 0.00025,
                "commission_min": 5.0,
                "buy_threshold": 0.55,
                "sell_threshold": 0.45,
                "confirm_days": 2,
                "min_hold_days": 1,
                "max_hold_days": 20,
                "take_profit": None,
                "stop_loss": None,
            }
        }

        ea_df = _build_execution_assumptions(repro_config)

        assert len(ea_df) > 0
        assert "Parameter" in ea_df.columns
        assert "Value" in ea_df.columns
        
        # 检查关键参数
        params = " ".join(ea_df["Parameter"].astype(str))
        assert "exec_price_model" in params
        assert "slippage_bps" in params
        assert "commission_rate" in params


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
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "close_qfq": range(10),
        })

        best_strategy_config = {
            "feature_config": {"windows": [3, 5]},
            "model_config": {"model_type": "logreg"},
            "trade_config": {"buy_threshold": 0.55},
            "split_mode": "walkforward",
        }

        repro_config = {
            "feature_config": {"windows": [3, 5]},
            "model_config": {"model_type": "logreg"},
            "trade_config": {
                "buy_threshold": 0.55,
                "exec_price_model": "next_open",
                "slippage_bps": 10.0,
            },
            "search_config": {"runs": 100},
            "split_mode": "walkforward",
            "n_folds": 4,
        }

        run_context = {
            "mode": "first",
            "base_seed": 42,
            "effective_runs": 100,
            "config_source": "CLI/default",
        }

        feature_meta = {
            "feature_names": ["ret_1"],
            "dpoint_explainer": "test",
        }

        trades = pd.DataFrame([{"status": "CLOSED", "pnl": 100}])
        equity = pd.DataFrame({"total_equity": [100000, 101000]})
        search_log = pd.DataFrame([{"iter": 1, "val_metric_final": 0.1, "fold_details_json": "[]"}])

        output_dir = str(tmp_path)

        _, config_path, _ = save_run_outputs(
            output_dir=output_dir,
            df_clean=df,
            log_notes=["Test"],
            trades=trades,
            equity_curve=equity,
            best_strategy_config=best_strategy_config,
            repro_config=repro_config,
            run_context=run_context,
            feature_meta=feature_meta,
            search_log=search_log,
        )

        # 读取 JSON 检查结构
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 检查新结构字段
        assert "run_id" in data
        assert "created_at" in data
        assert "best_strategy_config" in data
        assert "repro_config" in data
        assert "run_context" in data
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

    def test_escape_minus(self):
        """- 应该被转义"""
        df = pd.DataFrame({"col": ["-A1+B2", "normal"]})
        escaped = escape_excel_formulas(df)
        assert escaped["col"].iloc[0] == "'-A1+B2"

    def test_escape_at(self):
        """@ 应该被转义"""
        df = pd.DataFrame({"col": ["@SUM(A1)", "normal"]})
        escaped = escape_excel_formulas(df)
        assert escaped["col"].iloc[0] == "'@SUM(A1)"

    def test_no_escape_normal(self):
        """正常文本不应该被转义"""
        df = pd.DataFrame({"col": ["normal text", "another"]})
        escaped = escape_excel_formulas(df)
        assert escaped["col"].iloc[0] == "normal text"
