import sys
sys.path.insert(0, r'J:\Ashare_DpointTrader 2.0')

from backtester_engine import backtest_from_dpoint, calc_transaction_costs, apply_slippage

print('Import OK')
print(f'Commission test (BUY 100 shares @ 10 CNY): {calc_transaction_costs("BUY", 100, 10.0):.4f} CNY')
print(f'Slippage test (10 bps on 10 CNY): {apply_slippage(10.0, "BUY", 10.0):.4f} CNY')
print(f'Stamp tax test (SELL 100 shares @ 10 CNY): {calc_transaction_costs("SELL", 100, 10.0):.4f} CNY')
