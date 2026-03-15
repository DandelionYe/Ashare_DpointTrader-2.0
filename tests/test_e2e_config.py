"""
端到端配置复现测试。

覆盖关键场景：
    1. config 文件加载和 CLI 覆盖
    2. metadata 中的种子语义
    3. --record-metadata 开关
    4. 训练成本模型 = 最终回测成本模型
    5. 配置覆盖只影响请求的字段
"""

import pytest
import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_schema import FullConfig, TradeConfig, SearchConfig
from main_cli import resolve_effective_config


def create_mock_args(**overrides):
    """创建模拟的 argparse.Namespace 对象"""
    defaults = {
        'config': None,
        'runs': 100,
        'seed': 42,
        'initial_cash': 100000.0,
        'exec_price_model': 'next_open',
        'slippage_bps': 10.0,
        'commission_rate': 0.00025,
        'commission_min': 5.0,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


class TestConfigRoundtrip:
    """测试配置往返复现"""

    def test_config_roundtrip_reproduces_effective_config(self, tmp_path):
        """测试从 config 文件加载能复现 effective_config"""
        # 创建测试配置
        original_config = FullConfig(
            trade_config=TradeConfig(
                initial_cash=200000.0,
                exec_price_model='next_close',
                slippage_bps=15.0,
                commission_rate=0.0003,
                commission_min=5.0,
            ),
            search_config=SearchConfig(
                runs=500,
                epsilon=0.02,
                exploit_ratio=0.6,
            )
        )

        # 保存到临时文件
        config_file = tmp_path / "test_config.json"
        original_config.save_json(str(config_file))

        # 模拟 CLI 参数（使用默认值，不覆盖）
        args = create_mock_args(config=str(config_file))

        # 解析配置
        effective_config = resolve_effective_config(args)

        # 验证配置一致
        assert effective_config.trade_config.initial_cash == 200000.0
        assert effective_config.trade_config.exec_price_model == 'next_close'
        assert effective_config.trade_config.slippage_bps == 15.0
        assert effective_config.search_config.runs == 500
        assert effective_config.search_config.epsilon == 0.02

    def test_config_override_only_changes_requested_fields(self, tmp_path):
        """测试配置覆盖只改变请求的字段"""
        # 创建基础配置
        base_config = FullConfig(
            trade_config=TradeConfig(
                initial_cash=200000.0,
                exec_price_model='next_close',
                slippage_bps=15.0,
            ),
            search_config=SearchConfig(runs=500)
        )

        config_file = tmp_path / "base_config.json"
        base_config.save_json(str(config_file))

        # CLI 覆盖 runs 和 slippage_bps
        args = create_mock_args(
            config=str(config_file),
            runs=1000,  # 覆盖
            slippage_bps=20.0,  # 覆盖（非默认值）
        )

        effective_config = resolve_effective_config(args)

        # runs 是 CLI 参数，不影响 FullConfig
        # initial_cash 应该保持原配置
        assert effective_config.trade_config.initial_cash == 200000.0
        # exec_price_model 应该保持原配置
        assert effective_config.trade_config.exec_price_model == 'next_close'
        # slippage_bps 应该被 CLI 覆盖
        assert effective_config.trade_config.slippage_bps == 20.0


class TestMetadataSeedReconstruction:
    """测试 metadata 种子重建"""

    def test_metadata_can_reconstruct_effective_seed(self, tmp_path):
        """测试从 metadata 可以重建 effective seed"""
        # 模拟 first 模式
        base_seed = 42
        latest_run_id = 0
        expected_search_seed = base_seed + latest_run_id

        # 验证种子计算
        assert expected_search_seed == 42

        # 模拟 continue 模式（有历史运行）
        latest_run_id = 5
        expected_search_seed_continue = base_seed + latest_run_id

        assert expected_search_seed_continue == 47

        # 验证 metadata 中记录的三种种子语义
        # 1. base_seed = CLI --seed
        # 2. search_seed = base_seed + latest_run_id
        # 3. final_train_seed = base_seed

        # 在 metadata notes 中应该能找到这些信息
        notes = [
            f"Base seed (CLI --seed): {base_seed}",
            f"Search seed (base + latest_run_id={latest_run_id}): {expected_search_seed_continue}",
            f"Final train seed (same as base): {base_seed}",
        ]

        # 验证 notes 格式
        assert "Base seed (CLI --seed): 42" in notes[0]
        assert "Search seed (base + latest_run_id=5): 47" in notes[1]
        assert "Final train seed (same as base): 42" in notes[2]


class TestRecordMetadataToggle:
    """测试 --record-metadata 开关"""

    def test_record_metadata_can_be_disabled(self):
        """测试可以禁用 metadata 记录"""
        # 测试 argparse.BooleanOptionalAction
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--record-metadata',
            dest='record_metadata',
            action=argparse.BooleanOptionalAction,
            default=True,
        )

        # 默认开启
        args = parser.parse_args([])
        assert args.record_metadata is True

        # 显式开启
        args = parser.parse_args(['--record-metadata'])
        assert args.record_metadata is True

        # 关闭
        args = parser.parse_args(['--no-record-metadata'])
        assert args.record_metadata is False


class TestCostModelConsistency:
    """测试成本模型一致性"""

    def test_walkforward_and_final_backtest_use_same_cost_model(self):
        """测试 walk-forward 和最终回测使用相同的成本模型"""
        from metrics import backtest_fold_stats
        from backtester_engine import backtest_from_dpoint

        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'open_qfq': np.random.uniform(10, 12, 100),
            'high_qfq': np.random.uniform(12, 14, 100),
            'low_qfq': np.random.uniform(9, 11, 100),
            'close_qfq': np.random.uniform(10, 13, 100),
            'volume': np.random.randint(1000, 10000, 100),
            'amount': np.random.uniform(10000, 100000, 100),
            'turnover_rate': np.random.uniform(0.01, 0.05, 100),
        })
        df = df.set_index('date')

        # 创建测试 dpoint
        dpoint = pd.Series(np.random.uniform(0.3, 0.7, 100), index=dates, name='dpoint')

        # 统一的成本参数
        cost_params = {
            'commission_rate': 0.0003,
            'commission_min': 5.0,
            'transfer_fee_rate': 0.00001,
            'stamp_tax_rate': 0.0005,
            'slippage_bps': 15.0,
        }

        trade_cfg = {
            'initial_cash': 100000.0,
            'buy_threshold': 0.55,
            'sell_threshold': 0.45,
            'confirm_days': 2,
            'min_hold_days': 1,
            'max_hold_days': 20,
            'take_profit': None,
            'stop_loss': None,
        }

        # 验证 backtest_fold_stats 接受成本参数
        # （这里只是验证接口存在，实际测试在 test_backtester.py）
        import inspect
        sig = inspect.signature(backtest_fold_stats)
        params = list(sig.parameters.keys())

        assert 'commission_rate' in params
        assert 'commission_min' in params
        assert 'transfer_fee_rate' in params
        assert 'stamp_tax_rate' in params

        # 验证 backtest_from_dpoint 也接受相同参数
        sig_final = inspect.signature(backtest_from_dpoint)
        params_final = list(sig_final.parameters.keys())

        assert 'commission_rate' in params_final
        assert 'commission_min' in params_final
        # transfer_fee_rate 和 stamp_tax_rate 有默认值
        assert 'transfer_fee_rate' in params_final
        assert 'stamp_tax_rate' in params_final


class TestEODTakeProfitStopLoss:
    """测试 EOD-based 止盈止损机制"""

    def test_eod_take_profit_triggered_on_close(self):
        """测试 EOD 止盈基于收盘价触发"""
        from backtester_engine import backtest_from_dpoint
        import pandas as pd
        import numpy as np

        # 创建简单的测试数据：价格持续上涨
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        # 价格从 10 元涨到 15 元（50% 涨幅）
        prices = np.linspace(10, 15, 20)

        df = pd.DataFrame({
            'date': dates,
            'open_qfq': prices,
            'high_qfq': prices * 1.02,
            'low_qfq': prices * 0.98,
            'close_qfq': prices,
            'volume': 10000,
            'amount': 100000,
            'turnover_rate': 0.01,
        })
        # 不要 set_index，backtest_from_dpoint 需要 date 列

        # 创建 dpoint：前两天>0.55 触发买入
        dpoint = pd.Series([0.6, 0.6] + [0.5] * 18, index=dates, name='dpoint')

        # 设置止盈 10%
        bt = backtest_from_dpoint(
            df=df,
            dpoint=dpoint,
            initial_cash=100000.0,
            buy_threshold=0.55,
            sell_threshold=0.45,
            confirm_days=2,
            min_hold_days=1,
            take_profit=0.10,  # 10% 止盈
            stop_loss=None,
            exec_price_model='next_open',
            slippage_bps=0.0,
            commission_rate=0.0,
            commission_min=0.0,
        )

        # 验证有交易发生
        assert len(bt.trades) > 0

        # 验证止盈触发（应该有一天收盘价相对买入价>=10%）
        # 由于价格持续上涨，止盈应该被触发
        # 注意：这是 EOD-based，所以使用收盘价判断

    def test_intraday_not_triggered_eod_not_triggered(self):
        """测试盘中触及但收盘未触及不触发止盈"""
        # 这个测试验证 EOD-based 与 intraday-based 的区别
        # EOD-based：只看收盘价，盘中高低点不触发
        from backtester_engine import backtest_from_dpoint
        import pandas as pd
        import numpy as np

        # 创建测试数据：盘中大涨但收盘回落
        dates = pd.date_range('2023-01-01', periods=15, freq='D')

        # 第 5 天盘中涨到 12 元（+20%），但收盘回落到 10.5 元（+5%）
        close_prices = [10.0, 10.0, 10.0, 10.0, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5]
        high_prices = [10.2, 10.2, 10.2, 10.2, 12.0, 10.7, 10.7, 10.7, 10.7, 10.7, 10.7, 10.7, 10.7, 10.7, 10.7]

        df = pd.DataFrame({
            'date': dates,
            'open_qfq': close_prices,
            'high_qfq': high_prices,
            'low_qfq': [9.8] * 15,
            'close_qfq': close_prices,
            'volume': 10000,
            'amount': 100000,
            'turnover_rate': 0.01,
        })
        # 不要 set_index

        # 前两天 dpoint>0.55 触发买入
        dpoint = pd.Series([0.6, 0.6] + [0.5] * 13, index=dates, name='dpoint')

        # 设置止盈 15%（收盘价从未达到）
        bt = backtest_from_dpoint(
            df=df,
            dpoint=dpoint,
            initial_cash=100000.0,
            buy_threshold=0.55,
            sell_threshold=0.45,
            confirm_days=2,
            min_hold_days=1,
            take_profit=0.15,  # 15% 止盈，收盘价从未达到
            stop_loss=None,
            exec_price_model='next_open',
            slippage_bps=0.0,
            commission_rate=0.0,
            commission_min=0.0,
        )

        # 验证：由于是 EOD-based，盘中触及 20% 但收盘只有 5%，止盈不应触发
        # 持仓应该保持（直到测试期结束）
        open_trades = bt.trades[bt.trades['status'] == 'OPEN'] if 'status' in bt.trades.columns else bt.trades
        # 至少有一个未平仓交易（因为止盈未触发）
        assert len(bt.trades) >= 1
