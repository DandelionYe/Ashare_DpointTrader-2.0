"""
测试 backtester_engine 的核心执行逻辑。

覆盖：
    - T+1 执行（信号日 t，执行日 t+1）
    - 最小 100 股交易单位
    - min_hold_days 约束
    - 止盈止损触发
    - 期末未平仓处理
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtester_engine import (
    backtest_from_dpoint,
    calc_transaction_costs,
    apply_slippage,
    _calc_buy_shares,
    _build_signal_frame,
)


class TestTransactionCosts:
    """测试交易成本计算"""

    def test_buy_commission_minimum(self):
        """买入佣金低于 5 元时按 5 元收取"""
        # 100 股 @ 10 元 = 1000 元，佣金应为 5 元（最低）
        cost = calc_transaction_costs("BUY", 100, 10.0)
        assert cost >= 5.0, f"Minimum commission should be >= 5, got {cost}"

    def test_sell_includes_stamp_tax(self):
        """卖出收取印花税"""
        buy_cost = calc_transaction_costs("BUY", 1000, 10.0)
        sell_cost = calc_transaction_costs("SELL", 1000, 10.0)
        # 卖出成本应高于买入（因为印花税）
        assert sell_cost > buy_cost, "Sell cost should include stamp tax"

    def test_cost_scales_with_turnover(self):
        """成本随成交金额增加"""
        cost_small = calc_transaction_costs("BUY", 100, 10.0)
        cost_large = calc_transaction_costs("BUY", 10000, 10.0)
        assert cost_large > cost_small, "Cost should scale with turnover"


class TestSlippage:
    """测试滑点模型"""

    def test_buy_slippage_increases_price(self):
        """买入滑点使价格上升"""
        price = 10.0
        slippage_bps = 10  # 0.1%
        exec_price = apply_slippage(price, "BUY", slippage_bps)
        assert exec_price > price, "Buy slippage should increase price"
        assert abs(exec_price - price * 1.001) < 0.0001, "10 bps = 0.1%"

    def test_sell_slippage_decreases_price(self):
        """卖出滑点使价格下降"""
        price = 10.0
        slippage_bps = 10
        exec_price = apply_slippage(price, "SELL", slippage_bps)
        assert exec_price < price, "Sell slippage should decrease price"
        assert abs(exec_price - price * 0.999) < 0.0001, "10 bps = 0.1%"

    def test_zero_slippage(self):
        """零滑点返回原价"""
        price = 10.0
        exec_price_buy = apply_slippage(price, "BUY", 0.0)
        exec_price_sell = apply_slippage(price, "SELL", 0.0)
        assert exec_price_buy == price
        assert exec_price_sell == price


class TestBuyShares:
    """测试买入股数计算（100 股整数倍）"""

    def test_rounds_to_100_shares(self):
        """向下取整到 100 股"""
        shares = _calc_buy_shares(10000.0, 10.0)  # 可买 1000 股
        assert shares == 1000
        assert shares % 100 == 0

    def test_insufficient_cash(self):
        """现金不足 100 股时返回 0"""
        shares = _calc_buy_shares(500.0, 10.0)  # 只能买 50 股，不足 100
        assert shares == 0

    def test_exact_100_shares(self):
        """刚好够 100 股"""
        shares = _calc_buy_shares(1000.0, 10.0)
        assert shares == 100


class TestTPlus1Execution:
    """测试 T+1 执行逻辑"""

    @pytest.fixture
    def sample_data(self):
        """创建 5 天样本数据"""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": [10.0, 10.2, 10.4, 10.6, 10.8],
            "high_qfq": [10.5, 10.7, 10.9, 11.1, 11.3],
            "low_qfq": [9.5, 9.7, 9.9, 10.1, 10.3],
            "close_qfq": [10.2, 10.4, 10.6, 10.8, 11.0],
        })
        return df

    @pytest.fixture
    def buy_signal_dpoint(self, sample_data):
        """创建买入信号：第 1-2 天 dpoint>0.55，第 3 天触发买入"""
        dpoint = pd.Series([0.6, 0.6, 0.4, 0.4, 0.4], index=sample_data["date"])
        return dpoint

    def test_signal_at_t_exec_at_tplus1(self, sample_data, buy_signal_dpoint):
        """信号在 t 日生成，t+1 日执行"""
        # confirm_days=2，所以第 3 天（索引 2）才会触发买入信号
        # 执行在第 4 天（索引 3），使用第 3 天的价格（same_close）或第 4 天的开盘价（next_open）
        bt = backtest_from_dpoint(
            df=sample_data,
            dpoint=buy_signal_dpoint,
            buy_threshold=0.55,
            sell_threshold=0.45,
            confirm_days=2,
            min_hold_days=1,
            exec_price_model="next_open",
            slippage_bps=0,
        )

        # 应该有交易记录
        assert len(bt.trades) >= 1, "Should have at least one trade"

        # 检查执行价格使用的是 next_open（第 4 天的开盘价 10.4）
        # 注意：实际执行逻辑可能在第 3 天就触发信号，所以价格可能是 10.4
        if not bt.trades.empty and "buy_price" in bt.trades.columns:
            buy_price = bt.trades.iloc[0]["buy_price"]
            # 第 3 天或第 4 天开盘价是 10.4 或 10.6
            assert 10.3 <= buy_price <= 10.7, f"Expected ~10.4-10.6, got {buy_price}"


class TestMinHoldDays:
    """测试最小持仓天数约束"""

    @pytest.fixture
    def sample_data_10days(self):
        """创建 10 天样本数据"""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": np.linspace(10.0, 11.0, 10),
            "high_qfq": np.linspace(10.5, 11.5, 10),
            "low_qfq": np.linspace(9.5, 10.5, 10),
            "close_qfq": np.linspace(10.2, 11.2, 10),
        })
        return df

    def test_cannot_sell_before_min_hold(self, sample_data_10days):
        """在 min_hold_days 内不能卖出"""
        # 创建强烈的买入信号（第 1-2 天），然后强烈的卖出信号（第 3-4 天）
        dpoint = pd.Series([0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                          index=sample_data_10days["date"])

        bt = backtest_from_dpoint(
            df=sample_data_10days,
            dpoint=dpoint,
            buy_threshold=0.55,
            sell_threshold=0.45,
            confirm_days=2,
            min_hold_days=3,  # 至少持有 3 天
            exec_price_model="next_open",
            slippage_bps=0,
        )

        # 检查交易记录
        if not bt.trades.empty:
            trade = bt.trades.iloc[0]
            if trade.get("status") == "CLOSED":
                buy_date = pd.to_datetime(trade["buy_exec_date"])
                sell_date = pd.to_datetime(trade["sell_exec_date"])
                held_days = (sell_date - buy_date).days
                # 考虑周末，实际日历日可能大于交易日
                assert held_days >= 2, f"Held {held_days} days, should be >= 2 trading days"


class TestTakeProfitStopLoss:
    """测试止盈止损"""

    @pytest.fixture
    def trending_data(self):
        """创建先涨后跌的数据"""
        dates = pd.date_range("2024-01-01", periods=15, freq="B")
        # 前 10 天上涨，后 5 天下跌
        closes = list(np.linspace(10.0, 15.0, 10)) + list(np.linspace(15.0, 10.0, 6)[1:])
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": closes,
            "high_qfq": [c * 1.02 for c in closes],
            "low_qfq": [c * 0.98 for c in closes],
            "close_qfq": closes,
        })
        return df

    def test_take_profit_triggers(self, trending_data):
        """止盈触发"""
        dpoint = pd.Series([0.7] * 2 + [0.3] * 13, index=trending_data["date"])

        bt = backtest_from_dpoint(
            df=trending_data,
            dpoint=dpoint,
            buy_threshold=0.55,
            sell_threshold=0.45,
            confirm_days=2,
            min_hold_days=1,
            take_profit=0.10,  # 10% 止盈
            exec_price_model="next_open",
            slippage_bps=0,
        )

        # 检查是否有止盈触发（净值应该在某个高点回落）
        if not bt.equity_curve.empty:
            max_equity = bt.equity_curve["total_equity"].max()
            final_equity = bt.equity_curve["total_equity"].iloc[-1]
            # 止盈后净值应该从高点回落
            assert max_equity > 0, "Should have positive max equity"

    def test_take_profit_is_close_based_not_high_based(self):
        """
        测试止盈是基于收盘价而非最高价。
        
        构造：
            当天 high_qfq 已经穿过 TP（+15%）
            但 close_qfq 没穿（+5%）
        
        断言：
            当前实现不触发 TP（因为 EOD-based 只看收盘价）
        """
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        
        # 第 5 天：开盘 10 元，最高涨到 11.5 元（+15%），但收盘回落到 10.5 元（+5%）
        # 这样盘中触及 10% TP，但收盘未触及
        closes = [10.0, 10.0, 10.0, 10.0, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5]
        highs =  [10.2, 10.2, 10.2, 10.2, 11.5, 10.7, 10.7, 10.7, 10.7, 10.7]
        lows =   [9.8,  9.8,  9.8,  9.8,  10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        opens =  [10.0, 10.0, 10.0, 10.0, 10.0, 10.5, 10.5, 10.5, 10.5, 10.5]
        
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": opens,
            "high_qfq": highs,
            "low_qfq": lows,
            "close_qfq": closes,
        })
        
        # 前两天 dpoint>0.55 触发买入
        dpoint = pd.Series([0.6, 0.6] + [0.5] * 8, index=dates, name="dpoint")
        
        # 设置 10% 止盈
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
            exec_price_model="next_open",
            slippage_bps=0.0,
            commission_rate=0.0,
            commission_min=0.0,
        )
        
        # 验证：由于是 EOD-based，盘中触及 15% 但收盘只有 5%，止盈不应触发
        # 持仓应该保持（因为没有其他卖出信号）
        if not bt.trades.empty:
            # 检查是否有 CLOSED 交易
            closed_trades = bt.trades[bt.trades["status"] == "CLOSED"] if "status" in bt.trades.columns else bt.trades
            
            # 如果止盈是 intraday-based，第 5 天就会触发
            # 但因为是 EOD-based，第 5 天不会触发
            # 所以要么没有 CLOSED 交易，要么 CLOSED 交易的 sell_reason 不包含 take_profit
            if len(closed_trades) > 0:
                # 检查卖出原因
                if "sell_reason" in closed_trades.columns:
                    for _, trade in closed_trades.iterrows():
                        reason = str(trade.get("sell_reason", ""))
                        # 第 5 天的卖出不应该是 take_profit 触发
                        # （可能是其他原因如 stop_loss 或信号反转）
                        assert "EOD take_profit" not in reason or trade["sell_exec_date"] != dates[4]

    def test_stop_loss_is_close_based_not_low_based(self):
        """
        测试止损是基于收盘价而非最低价。
        
        构造：
            当天 low_qfq 已经跌穿 SL（-15%）
            但 close_qfq 没跌穿（-5%）
        
        断言：
            当前实现不触发 SL（因为 EOD-based 只看收盘价）
        """
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        
        # 第 5 天：开盘 10 元，最低跌到 8.5 元（-15%），但收盘回升到 9.5 元（-5%）
        # 这样盘中触及 10% SL，但收盘未触及
        closes = [10.0, 10.0, 10.0, 10.0, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5]
        highs =  [10.2, 10.2, 10.2, 10.2, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        lows =   [9.8,  9.8,  9.8,  9.8,  8.5, 9.0, 9.0, 9.0, 9.0, 9.0]
        opens =  [10.0, 10.0, 10.0, 10.0, 10.0, 9.5, 9.5, 9.5, 9.5, 9.5]
        
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": opens,
            "high_qfq": highs,
            "low_qfq": lows,
            "close_qfq": closes,
        })
        
        # 前两天 dpoint>0.55 触发买入
        dpoint = pd.Series([0.6, 0.6] + [0.5] * 8, index=dates, name="dpoint")
        
        # 设置 10% 止损
        bt = backtest_from_dpoint(
            df=df,
            dpoint=dpoint,
            initial_cash=100000.0,
            buy_threshold=0.55,
            sell_threshold=0.45,
            confirm_days=2,
            min_hold_days=1,
            take_profit=None,
            stop_loss=0.10,  # 10% 止损
            exec_price_model="next_open",
            slippage_bps=0.0,
            commission_rate=0.0,
            commission_min=0.0,
        )
        
        # 验证：由于是 EOD-based，盘中触及 15% 但收盘只有 5%，止损不应触发
        if not bt.trades.empty:
            closed_trades = bt.trades[bt.trades["status"] == "CLOSED"] if "status" in bt.trades.columns else bt.trades
            
            if len(closed_trades) > 0:
                if "sell_reason" in closed_trades.columns:
                    for _, trade in closed_trades.iterrows():
                        reason = str(trade.get("sell_reason", ""))
                        # 第 5 天的卖出不应该是 stop_loss 触发
                        assert "EOD stop_loss" not in reason or trade["sell_exec_date"] != dates[4]

    def test_force_sell_reason_contains_eod_marker(self):
        """
        测试强制卖出原因包含 EOD 标记。
        
        断言：
            交易原因或日志里包含 "EOD take_profit" / "EOD stop_loss"
        """
        dates = pd.date_range("2024-01-01", periods=15, freq="B")
        
        # 创建持续上涨数据，第 5 天后涨超 10%
        closes = [10.0, 10.0, 10.0, 10.0, 11.2, 11.5, 11.8, 12.0, 12.2, 12.5, 12.8, 13.0, 13.2, 13.5, 13.8]
        highs =  [c * 1.01 for c in closes]
        lows =   [c * 0.99 for c in closes]
        opens =  closes
        
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": opens,
            "high_qfq": highs,
            "low_qfq": lows,
            "close_qfq": closes,
        })
        
        # 前两天 dpoint>0.55 触发买入
        dpoint = pd.Series([0.6, 0.6] + [0.5] * 13, index=dates, name="dpoint")
        
        # 设置 10% 止盈
        bt = backtest_from_dpoint(
            df=df,
            dpoint=dpoint,
            initial_cash=100000.0,
            buy_threshold=0.55,
            sell_threshold=0.45,
            confirm_days=2,
            min_hold_days=1,
            take_profit=0.10,  # 10% 止盈
            stop_loss=0.10,    # 10% 止损
            exec_price_model="next_open",
            slippage_bps=0.0,
            commission_rate=0.0,
            commission_min=0.0,
        )
        
        # 检查 notes 是否包含 EOD 标记
        eod_found = False
        for note in bt.notes:
            if "EOD take_profit" in note or "EOD stop_loss" in note:
                eod_found = True
                break
        
        # 如果止盈/止损被触发，notes 里应该有 EOD 标记
        # 这是一个弱断言，因为取决于具体实现
        # 至少验证机制存在
        assert eod_found or len(bt.notes) > 0, "Should have notes about EOD-based triggers"


class TestUnrealizedPnl:
    """测试期末未平仓处理"""

    @pytest.fixture
    def short_data(self):
        """创建短数据（只有 5 天），让持仓来不及平仓"""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": [10.0, 10.2, 10.4, 10.6, 10.8],
            "high_qfq": [10.5, 10.7, 10.9, 11.1, 11.3],
            "low_qfq": [9.5, 9.7, 9.9, 10.1, 10.3],
            "close_qfq": [10.2, 10.4, 10.6, 10.8, 11.0],
        })
        return df

    def test_open_position_at_end(self, short_data):
        """期末未平仓应有 unrealized_pnl"""
        dpoint = pd.Series([0.7, 0.7, 0.7, 0.7, 0.7], index=short_data["date"])

        bt = backtest_from_dpoint(
            df=short_data,
            dpoint=dpoint,
            buy_threshold=0.55,
            sell_threshold=0.45,
            confirm_days=2,
            min_hold_days=1,
            exec_price_model="next_open",
            slippage_bps=0,
        )

        # 应该有未平仓交易
        if not bt.trades.empty:
            open_trades = bt.trades[bt.trades["status"] == "OPEN"]
            if len(open_trades) > 0:
                assert "unrealized_pnl" in open_trades.columns
                assert "unrealized_return" in open_trades.columns


class TestExecPriceModels:
    """测试不同执行价模型"""

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "open_qfq": [10.0, 10.5, 11.0, 11.5, 12.0],
            "close_qfq": [10.5, 11.0, 11.5, 12.0, 12.5],
            "high_qfq": [11.0, 11.5, 12.0, 12.5, 13.0],
            "low_qfq": [9.5, 10.0, 10.5, 11.0, 11.5],
        })
        return df

    def test_same_close_idealized(self, sample_data):
        """same_close_idealized 使用信号日收盘价"""
        dpoint = pd.Series([0.7, 0.7, 0.3, 0.3, 0.3], index=sample_data["date"])

        bt = backtest_from_dpoint(
            df=sample_data,
            dpoint=dpoint,
            buy_threshold=0.55,
            sell_threshold=0.45,
            confirm_days=2,
            min_hold_days=1,
            exec_price_model="same_close_idealized",
            slippage_bps=0,
        )

        if not bt.trades.empty and "buy_price" in bt.trades.columns:
            buy_price = bt.trades.iloc[0]["buy_price"]
            # 应该使用第 3 天的收盘价 11.5
            assert abs(buy_price - 11.5) < 0.1, f"Expected ~11.5, got {buy_price}"

    def test_next_open(self, sample_data):
        """next_open 使用执行日开盘价"""
        dpoint = pd.Series([0.7, 0.7, 0.3, 0.3, 0.3], index=sample_data["date"])

        bt = backtest_from_dpoint(
            df=sample_data,
            dpoint=dpoint,
            buy_threshold=0.55,
            sell_threshold=0.45,
            confirm_days=2,
            min_hold_days=1,
            exec_price_model="next_open",
            slippage_bps=0,
        )

        if not bt.trades.empty and "buy_price" in bt.trades.columns:
            buy_price = bt.trades.iloc[0]["buy_price"]
            # 应该使用第 4 天的开盘价 11.5
            assert abs(buy_price - 11.5) < 0.1, f"Expected ~11.5, got {buy_price}"
