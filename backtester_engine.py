# backtester_engine.py
"""
回测执行引擎（Realistic Execution 版本）。

关键改进：
    - 执行价模型参数化：支持 same_close_idealized、next_open、next_close
    - 真实交易成本：佣金、过户费、卖出印花税、最小费用门槛
    - 滑点模型：固定 bps 滑点，模拟成交偏差

公开 API：
    backtest_from_dpoint(df, dpoint, ...) -> BacktestResult

内部结构（信号/执行分离）：
    _build_signal_frame(df, dpoint, buy_threshold, sell_threshold)
        → 逐日的 dpoint 原始比较结果（纯向量运算，无状态，可独立测试）
    _simulate_execution(df, signal_frame, ...)
        → 含状态的执行模拟（挂单、持仓、净值快照）
    _normalize_open_trade(trade, ...)
        → 统一补全交易记录缺失字段，消除分散的 setdefault 调用

A 股约束：
    - 仅做多，不做空
    - 最小交易单位 100 股
    - T+1：信号在 t 日生成，t+1 日执行（使用可配置的执行价模型）
    - min_hold_days >= 1 强制模拟 T+1 锁定期
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd


# =========================================================
# 常量定义：交易成本参数（A 股典型值）
# =========================================================
# 佣金（券商收取，双向），默认万分之 2.5，最低 5 元
DEFAULT_COMMISSION_RATE: float = 0.00025
DEFAULT_COMMISSION_MIN: float = 5.0

# 过户费（中登公司收取，双向），默认万分之 0.1
DEFAULT_TRANSFER_FEE_RATE: float = 0.00001

# 印花税（国家收取，仅卖出），默认千分之 0.5（2023 年 8 月后调整为 0.05%）
DEFAULT_STAMP_TAX_RATE: float = 0.0005


# =========================================================
# 数据类
# =========================================================
@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    notes: List[str]


# =========================================================
# 交易成本计算
# =========================================================
def calc_transaction_costs(
    trade_type: Literal["BUY", "SELL"],
    shares: int,
    price: float,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    commission_min: float = DEFAULT_COMMISSION_MIN,
    transfer_fee_rate: float = DEFAULT_TRANSFER_FEE_RATE,
    stamp_tax_rate: float = DEFAULT_STAMP_TAX_RATE,
) -> float:
    """
    计算单笔交易的总成本。

    参数：
        trade_type       : "BUY" 或 "SELL"
        shares           : 股数
        price            : 成交价格
        commission_rate  : 佣金率（默认万分之 2.5）
        commission_min   : 最低佣金（默认 5 元）
        transfer_fee_rate: 过户费率（默认万分之 0.1）
        stamp_tax_rate   : 印花税率（默认千分之 0.5，仅卖出收取）

    返回：
        总交易成本（元）

    A 股收费规则：
        - 佣金：双向收取，成交金额的 commission_rate，最低 commission_min 元
        - 过户费：双向收取，成交金额的 transfer_fee_rate
        - 印花税：仅卖出收取，成交金额的 stamp_tax_rate
    """
    turnover = shares * price

    # 佣金（双向，有最低门槛）
    commission = max(turnover * commission_rate, commission_min)

    # 过户费（双向）
    transfer_fee = turnover * transfer_fee_rate

    # 印花税（仅卖出）
    stamp_tax = turnover * stamp_tax_rate if trade_type == "SELL" else 0.0

    return commission + transfer_fee + stamp_tax


def apply_slippage(
    price: float,
    trade_type: Literal["BUY", "SELL"],
    slippage_bps: float = 10.0,
) -> float:
    """
    应用滑点到成交价格。

    参数：
        price        : 基准价格
        trade_type   : "BUY" 或 "SELL"
        slippage_bps : 滑点（基点），默认 10 bps = 0.1%

    返回：
        滑点调整后的执行价格

    滑点逻辑：
        - 买入：价格向上调整（更贵的成交价）
        - 卖出：价格向下调整（更低的成交价）
    """
    slippage_ratio = slippage_bps / 10000.0
    if trade_type == "BUY":
        return price * (1.0 + slippage_ratio)
    else:
        return price * (1.0 - slippage_ratio)


# =========================================================
# 私有工具函数
# =========================================================
def _calc_buy_shares(cash: float, price: float) -> int:
    """按 A 股 100 股最小单位计算可买入股数。price <= 0 时返回 0。"""
    if price <= 0:
        return 0
    max_lot = int(cash // (price * 100))
    return max_lot * 100


def _normalize_open_trade(
    trade: Dict[str, object],
    buy_threshold: float,
    sell_threshold: float,
    confirm_days: int,
    min_hold_days: int,
) -> Dict[str, object]:
    """
    统一补全交易记录所有可能缺失的字段，避免 DataFrame 列不对齐。
    对 CLOSED 和 OPEN 两种状态均适用，缺失字段填 NaN / NaT。
    """
    # 卖出侧（未平仓时为空）
    trade.setdefault("sell_signal_date", pd.NaT)
    trade.setdefault("sell_exec_date", pd.NaT)
    trade.setdefault("sell_price", np.nan)
    trade.setdefault("sell_shares", np.nan)
    trade.setdefault("sell_proceeds", np.nan)
    trade.setdefault("cash_after_sell", np.nan)

    # 平仓指标（未平仓时不可用）
    trade.setdefault("pnl", np.nan)
    trade.setdefault("return", np.nan)
    trade.setdefault("success", np.nan)

    # 信号诊断字段
    trade.setdefault("buy_dpoint_signal_day", np.nan)
    trade.setdefault("sell_dpoint_signal_day", np.nan)
    trade.setdefault("buy_above_cnt_at_signal", np.nan)
    trade.setdefault("sell_below_cnt_at_signal", np.nan)

    # 策略参数快照（方便事后对账）
    trade.setdefault("buy_threshold", float(buy_threshold))
    trade.setdefault("sell_threshold", float(sell_threshold))
    trade.setdefault("confirm_days", int(confirm_days))
    trade.setdefault("min_hold_days", int(min_hold_days))

    # 交易成本字段
    trade.setdefault("buy_commission", np.nan)
    trade.setdefault("buy_transfer_fee", np.nan)
    trade.setdefault("sell_commission", np.nan)
    trade.setdefault("sell_transfer_fee", np.nan)
    trade.setdefault("sell_stamp_tax", np.nan)
    trade.setdefault("total_costs", np.nan)
    trade.setdefault("slippage_bps", np.nan)

    return trade


# =========================================================
# 第一层：信号帧构建（无状态，可独立测试）
# =========================================================
def _build_signal_frame(
    df: pd.DataFrame,
    dpoint: pd.Series,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    """
    对齐 dpoint 与行情数据，逐日计算原始阈值比较结果。

    此函数不含任何持仓状态或计数器，仅做向量化的比较运算，
    可以独立于执行模拟进行单元测试。

    返回 DataFrame，列：
        date          — 交易日
        close_qfq     — 后复权收盘价
        open_qfq      — 后复权开盘价（用于 next_open 执行价模型）
        dpoint        — 当日 Dpoint 值（NaN 表示无信号）
        dp_above_buy  — dpoint > buy_threshold（用于累计 above_cnt）
        dp_below_sell — dpoint < sell_threshold（用于累计 below_cnt）
    """
    close = df["close_qfq"].astype(float)
    open_ = df["open_qfq"].astype(float)
    dpoint_aligned = dpoint.reindex(df.index)

    signal_frame = pd.DataFrame({
        "date": df.index,
        "close_qfq": close,
        "open_qfq": open_,
        "dpoint": dpoint_aligned,
        "dp_above_buy": dpoint_aligned > buy_threshold,
        "dp_below_sell": dpoint_aligned < sell_threshold,
    })

    # NaN 的 dpoint 不触发任何方向
    signal_frame.loc[dpoint_aligned.isna(), ["dp_above_buy", "dp_below_sell"]] = False

    return signal_frame.reset_index(drop=True)


# =========================================================
# 第二层：执行模拟（有状态，按日循环）
# =========================================================
def _simulate_execution(
    signal_frame: pd.DataFrame,
    initial_cash: float,
    buy_threshold: float,
    sell_threshold: float,
    max_hold_days: int,
    take_profit: Optional[float],
    stop_loss: Optional[float],
    confirm_days: int,
    min_hold_days: int,
    exec_price_model: Literal["same_close_idealized", "next_open", "next_close"] = "next_open",
    slippage_bps: float = 10.0,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    commission_min: float = DEFAULT_COMMISSION_MIN,
    transfer_fee_rate: float = DEFAULT_TRANSFER_FEE_RATE,
    stamp_tax_rate: float = DEFAULT_STAMP_TAX_RATE,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[str]]:
    """
    有状态的逐日执行模拟。读取 _build_signal_frame 的输出，
    维护持仓状态机（挂单 → 成交 → 净值快照）。

    新增参数（真实执行）：
        exec_price_model   : 执行价模型
                             - "same_close_idealized": t 日信号，t+1 日执行，用 t 日收盘价（理想化，旧版行为）
                             - "next_open": t 日信号，t+1 日执行，用 t+1 日开盘价（推荐）
                             - "next_close": t 日信号，t+1 日执行，用 t+1 日收盘价（保守）
        slippage_bps       : 滑点（基点），默认 10 bps
        commission_rate    : 佣金率
        commission_min     : 最低佣金
        transfer_fee_rate  : 过户费率
        stamp_tax_rate     : 印花税率

    返回 (trade_rows, equity_rows, notes)，由 backtest_from_dpoint 组装为 BacktestResult。
    """
    notes: List[str] = []
    trade_rows: List[Dict[str, object]] = []
    equity_rows: List[Dict[str, object]] = []

    dates = list(signal_frame.index)

    cash: float = float(initial_cash)
    shares: int = 0
    position_entry_date: Optional[pd.Timestamp] = None
    pending_order: Optional[Dict[str, object]] = None
    open_trade: Optional[Dict[str, object]] = None

    # 连续满足条件的天数计数器（信号生成时在此维护）
    above_cnt: int = 0
    below_cnt: int = 0

    for i in range(len(dates)):
        row = signal_frame.iloc[i]
        dt: pd.Timestamp = row["date"]
        price_close_t: float = float(row["close_qfq"])
        price_open_t: float = float(row["open_qfq"])
        dp: float = float(row["dpoint"]) if pd.notna(row["dpoint"]) else float("nan")
        dp_above: bool = bool(row["dp_above_buy"])
        dp_below: bool = bool(row["dp_below_sell"])

        exec_action_today = "NONE"
        exec_price_used = np.nan

        # -----------------------------------------------------------
        # 阶段一：执行前一日挂单
        # -----------------------------------------------------------
        if pending_order is not None and pending_order.get("exec_date") == dt:
            action = str(pending_order["action"])
            signal_date = pd.to_datetime(pending_order["signal_date"])

            # --- 根据执行价模型确定基准执行价 ---
            if exec_price_model == "same_close_idealized":
                # 旧版行为：用信号日（t 日）的收盘价
                base_price = float(pending_order["price"])
            elif exec_price_model == "next_open":
                # 推荐：用执行日（t+1 日）的开盘价
                base_price = price_open_t
            elif exec_price_model == "next_close":
                # 保守：用执行日（t+1 日）的收盘价
                base_price = price_close_t
            else:
                raise ValueError(f"Unknown exec_price_model: {exec_price_model}")

            # --- 应用滑点 ---
            raw_price = base_price
            exec_price = apply_slippage(raw_price, action, slippage_bps)
            exec_price_used = exec_price

            if action == "BUY":
                if shares == 0:
                    buy_shares = _calc_buy_shares(cash, exec_price)
                    if buy_shares > 0:
                        # 计算交易成本
                        commission = calc_transaction_costs(
                            "BUY", buy_shares, exec_price,
                            commission_rate, commission_min,
                            transfer_fee_rate, stamp_tax_rate
                        )
                        cost = buy_shares * exec_price + commission
                        cash -= cost
                        shares += buy_shares
                        position_entry_date = dt
                        exec_action_today = "BUY_EXEC"
                        open_trade = {
                            "buy_signal_date": signal_date,
                            "buy_exec_date": dt,
                            "buy_price": exec_price,
                            "buy_shares": buy_shares,
                            "buy_cost": cost,
                            "cash_after_buy": cash,
                            "buy_dpoint_signal_day": float(pending_order.get("signal_dpoint", np.nan)),
                            "buy_threshold": float(buy_threshold),
                            "sell_threshold": float(sell_threshold),
                            "confirm_days": int(confirm_days),
                            "min_hold_days": int(min_hold_days),
                            "buy_above_cnt_at_signal": int(pending_order.get("above_cnt_at_signal", 0)),
                            "buy_commission": commission,
                            "buy_transfer_fee": buy_shares * exec_price * transfer_fee_rate,
                            "sell_commission": np.nan,
                            "sell_transfer_fee": np.nan,
                            "sell_stamp_tax": np.nan,
                            "total_costs": commission,
                            "slippage_bps": slippage_bps,
                            "exec_price_model": exec_price_model,
                        }
                    else:
                        notes.append(f"{dt.date()}: BUY skipped (insufficient cash for 100 shares).")
                else:
                    notes.append(f"{dt.date()}: BUY pending but already in position; skipped.")

            elif action == "SELL":
                if shares > 0:
                    held_days = (
                        (dt - position_entry_date).days
                        if position_entry_date is not None
                        else 999_999
                    )
                    if held_days >= min_hold_days:
                        # 计算交易成本
                        commission = calc_transaction_costs(
                            "SELL", shares, exec_price,
                            commission_rate, commission_min,
                            transfer_fee_rate, stamp_tax_rate
                        )
                        transfer_fee = shares * exec_price * transfer_fee_rate
                        stamp_tax = shares * exec_price * stamp_tax_rate
                        proceeds = shares * exec_price - commission
                        sell_shares = shares
                        cash += proceeds
                        shares = 0
                        position_entry_date = None
                        exec_action_today = "SELL_EXEC"

                        if open_trade is None:
                            open_trade = {}
                        open_trade.update({
                            "sell_signal_date": signal_date,
                            "sell_exec_date": dt,
                            "sell_price": exec_price,
                            "sell_shares": sell_shares,
                            "sell_proceeds": proceeds,
                            "cash_after_sell": cash,
                            "sell_dpoint_signal_day": float(pending_order.get("signal_dpoint", np.nan)),
                            "sell_below_cnt_at_signal": int(pending_order.get("below_cnt_at_signal", 0)),
                            "sell_commission": commission,
                            "sell_transfer_fee": transfer_fee,
                            "sell_stamp_tax": stamp_tax,
                            "total_costs": float(open_trade.get("total_costs", 0.0)) + commission + transfer_fee + stamp_tax,
                            "slippage_bps": slippage_bps,
                            "exec_price_model": exec_price_model,
                        })

                        buy_cost = float(open_trade.get("buy_cost", 0.0))
                        pnl = proceeds - buy_cost
                        open_trade["pnl"] = pnl
                        open_trade["return"] = pnl / buy_cost if buy_cost > 0 else np.nan
                        open_trade["success"] = bool(pnl > 0)
                        open_trade["status"] = "CLOSED"

                        open_trade = _normalize_open_trade(
                            open_trade, buy_threshold, sell_threshold,
                            confirm_days, min_hold_days,
                        )
                        trade_rows.append(open_trade)
                        open_trade = None
                    else:
                        notes.append(
                            f"{dt.date()}: SELL blocked by min_hold_days "
                            f"(held {held_days} < {min_hold_days})."
                        )
                else:
                    notes.append(f"{dt.date()}: SELL pending but no shares; skipped.")

            pending_order = None

        # -----------------------------------------------------------
        # 阶段二：更新计数器（使用预计算的 dp_above / dp_below）
        # -----------------------------------------------------------
        above_cnt = (above_cnt + 1) if dp_above else 0
        below_cnt = (below_cnt + 1) if dp_below else 0

        buy_condition_met = bool(
            (shares == 0) and (above_cnt >= confirm_days) and (pending_order is None)
        )

        # -----------------------------------------------------------
        # 阶段三：检查强制平仓条件
        # -----------------------------------------------------------
        force_sell = False
        force_reason = ""

        if shares > 0 and position_entry_date is not None and i < len(dates) - 1:
            next_dt = signal_frame.iloc[i + 1]["date"]
            held_days_exec = (next_dt - position_entry_date).days

            if held_days_exec >= max_hold_days:
                force_sell = True
                force_reason = (
                    f"max_hold_days reached ({held_days_exec}>={max_hold_days}) -> FORCE_SELL"
                )

            if open_trade is not None:
                buy_price = float(open_trade.get("buy_price", np.nan))
                if buy_price > 0:
                    pnl_ratio = (price_close_t / buy_price) - 1.0
                    if take_profit is not None and pnl_ratio >= float(take_profit):
                        force_sell = True
                        force_reason = (
                            f"take_profit reached ({pnl_ratio:.2%}>={take_profit:.2%}) -> FORCE_SELL"
                        )
                    if stop_loss is not None and pnl_ratio <= -float(stop_loss):
                        force_sell = True
                        force_reason = (
                            f"stop_loss reached ({pnl_ratio:.2%}<={-stop_loss:.2%}) -> FORCE_SELL"
                        )

        # -----------------------------------------------------------
        # 阶段四：生成今日信号，挂单至 t+1
        # -----------------------------------------------------------
        signal_today = "NONE"
        order_scheduled_for = pd.NaT
        reason = ""

        sell_condition_met = False
        if shares > 0 and (below_cnt >= confirm_days or force_sell) and (pending_order is None):
            if position_entry_date is None:
                sell_condition_met = True
            elif i < len(dates) - 1:
                next_dt = signal_frame.iloc[i + 1]["date"]
                sell_condition_met = (next_dt - position_entry_date).days >= min_hold_days

        if force_sell and shares > 0 and pending_order is None:
            sell_condition_met = True

        if i < len(dates) - 1 and pending_order is None and not np.isnan(dp):
            next_dt = signal_frame.iloc[i + 1]["date"]

            if buy_condition_met:
                signal_today = "BUY_SIGNAL"
                order_scheduled_for = next_dt
                reason = f"dpoint 连续{confirm_days}天>{buy_threshold} 且空仓 -> BUY_SIGNAL"
                pending_order = {
                    "action": "BUY",
                    "signal_date": dt,
                    "exec_date": next_dt,
                    "price": price_close_t,  # 保留 t 日收盘价用于 same_close_idealized 模型
                    "signal_dpoint": dp,
                    "above_cnt_at_signal": int(above_cnt),
                }
                above_cnt = 0
                below_cnt = 0

            elif sell_condition_met:
                signal_today = "SELL_SIGNAL"
                order_scheduled_for = next_dt
                reason = (
                    force_reason if force_sell
                    else f"dpoint 连续{confirm_days}天<{sell_threshold} "
                         f"且满足最短持有{min_hold_days}天 -> SELL_SIGNAL"
                )
                pending_order = {
                    "action": "SELL",
                    "signal_date": dt,
                    "exec_date": next_dt,
                    "price": price_close_t,  # 保留 t 日收盘价用于 same_close_idealized 模型
                    "signal_dpoint": dp,
                    "below_cnt_at_signal": int(below_cnt),
                }
                above_cnt = 0
                below_cnt = 0

        # -----------------------------------------------------------
        # 阶段五：净值快照
        # -----------------------------------------------------------
        market_value = shares * price_close_t
        equity_rows.append({
            "date": dt,
            "close_qfq": price_close_t,
            "open_qfq": price_open_t,
            "cash": cash,
            "shares": shares,
            "market_value": market_value,
            "total_equity": cash + market_value,
            "dpoint": dp if not np.isnan(dp) else np.nan,
            "above_cnt": int(above_cnt),
            "below_cnt": int(below_cnt),
            "buy_condition_met": bool(buy_condition_met),
            "sell_condition_met": bool(sell_condition_met),
            "signal_today": signal_today,
            "order_scheduled_for": order_scheduled_for,
            "exec_action_today": exec_action_today,
            "exec_price_used": exec_price_used,
            "reason": reason,
        })

    # -----------------------------------------------------------
    # 期末：处理未平仓持仓
    # -----------------------------------------------------------
    if open_trade is not None:
        last_row = signal_frame.iloc[-1]
        last_close = float(last_row["close_qfq"])
        buy_cost = float(open_trade.get("buy_cost", 0.0))
        buy_shares_held = float(open_trade.get("buy_shares", 0.0))
        mkt_value = buy_shares_held * last_close
        unreal_pnl = mkt_value - buy_cost if buy_cost > 0 else np.nan

        open_trade["status"] = "OPEN"
        open_trade["unrealized_pnl"] = unreal_pnl
        open_trade["unrealized_return"] = (unreal_pnl / buy_cost) if buy_cost > 0 else np.nan

        open_trade = _normalize_open_trade(
            open_trade, buy_threshold, sell_threshold,
            confirm_days, min_hold_days,
        )
        trade_rows.append(open_trade)

    return trade_rows, equity_rows, notes


# =========================================================
# 公开 API
# =========================================================
def backtest_from_dpoint(
    df: pd.DataFrame,
    dpoint: pd.Series,
    initial_cash: float = 100_000.0,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    max_hold_days: int = 20,
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
    confirm_days: int = 2,
    min_hold_days: int = 1,
    exec_price_model: Literal["same_close_idealized", "next_open", "next_close"] = "next_open",
    slippage_bps: float = 10.0,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    commission_min: float = DEFAULT_COMMISSION_MIN,
    transfer_fee_rate: float = DEFAULT_TRANSFER_FEE_RATE,
    stamp_tax_rate: float = DEFAULT_STAMP_TAX_RATE,
) -> BacktestResult:
    """
    将 Dpoint 序列转化为 A 股回测结果（真实执行版本）。

    流程（三步，对应三个内部函数）：
        1. 数据对齐与预处理
        2. _build_signal_frame  — 纯向量化阈值比较，无状态
        3. _simulate_execution  — 有状态执行模拟

    参数说明：
        df                 — 含 date / close_qfq / open_qfq 列的日频行情 DataFrame
        dpoint             — P(close_{t+1} > close_t | X_t)，index 为日期
        initial_cash       — 初始资金（元）
        buy_threshold      — Dpoint 连续高于此值 confirm_days 天触发买入信号
        sell_threshold     — Dpoint 连续低于此值 confirm_days 天触发卖出信号
        max_hold_days      — 最大持仓日历天数，超过则强制平仓
        take_profit        — 止盈比例（如 0.12 表示 12%），None 表示不启用
        stop_loss          — 止损比例（如 0.08 表示 8%），None 表示不启用
        confirm_days       — 连续满足条件天数，用于平滑信号
        min_hold_days      — 最短持仓天数（近似 T+1 约束）

        # === 新增：真实执行参数 ===
        exec_price_model   — 执行价模型（默认 "next_open"）
                             * "same_close_idealized": t 日信号，t+1 日执行，用 t 日收盘价（理想化，旧版行为）
                             * "next_open": t 日信号，t+1 日执行，用 t+1 日开盘价（推荐，更真实）
                             * "next_close": t 日信号，t+1 日执行，用 t+1 日收盘价（保守）
        slippage_bps       — 滑点（基点），默认 10 bps（0.1%）。买入价上浮，卖出价下浮
        commission_rate    — 佣金率，默认 0.00025（万分之 2.5）
        commission_min     — 最低佣金，默认 5 元
        transfer_fee_rate  — 过户费率，默认 0.00001（万分之 0.1）
        stamp_tax_rate     — 印花税率，默认 0.0005（千分之 0.5，仅卖出收取）
    """
    # --- 数据预处理 ---
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date", drop=False)

    dpoint = dpoint.copy()
    dpoint.index = pd.to_datetime(dpoint.index)

    # --- 第一步：构建信号帧（无状态）---
    signal_frame = _build_signal_frame(df, dpoint, buy_threshold, sell_threshold)

    # --- 第二步：执行模拟（有状态）---
    trade_rows, equity_rows, exec_notes = _simulate_execution(
        signal_frame=signal_frame,
        initial_cash=initial_cash,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_hold_days=max_hold_days,
        take_profit=take_profit,
        stop_loss=stop_loss,
        confirm_days=confirm_days,
        min_hold_days=min_hold_days,
        exec_price_model=exec_price_model,
        slippage_bps=slippage_bps,
        commission_rate=commission_rate,
        commission_min=commission_min,
        transfer_fee_rate=transfer_fee_rate,
        stamp_tax_rate=stamp_tax_rate,
    )

    # --- 第三步：组装结果 ---
    mode_note = (
        f"Execution: signal at t, execute at t+1 using {exec_price_model}. "
        f"Slippage={slippage_bps} bps, commission={commission_rate*10000:.1f}/10000 (min {commission_min}), "
        f"transfer_fee={transfer_fee_rate*10000:.1f}/10000, stamp_tax={stamp_tax_rate*1000:.2f}/1000 (sell only)."
    )
    notes = [mode_note] + exec_notes
    trades = pd.DataFrame(trade_rows)
    equity_curve = pd.DataFrame(equity_rows)

    if not equity_curve.empty:
        equity_curve = equity_curve.sort_values("date").reset_index(drop=True)
        equity_curve["cum_max_equity"] = equity_curve["total_equity"].cummax()
        equity_curve["drawdown"] = (
            equity_curve["total_equity"] / equity_curve["cum_max_equity"] - 1.0
        )

    return BacktestResult(trades=trades, equity_curve=equity_curve, notes=notes)
