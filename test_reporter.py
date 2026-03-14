"""测试 reporter 模块的 WalkForwardSummary 功能"""
import sys
sys.path.insert(0, r'J:\Ashare_DpointTrader 2.0')

from reporter import _build_walkforward_summary, _build_insample_warning_sheet
import pandas as pd

# 测试样本内警告页
print("=== Testing In-Sample Warning Sheet ===")
insample_df = _build_insample_warning_sheet()
print(insample_df.to_string())
print()

# 测试 WalkForwardSummary
print("=== Testing WalkForwardSummary ===")
test_search_log = pd.DataFrame([
    {"iter": 1, "val_metric_final": 0.05, "val_geom_mean_ratio": 1.02, "val_min_fold_ratio": 0.98, 
     "val_avg_closed_trades_per_fold": 3.5, "val_equity_proxy_mean": 105000, "val_metric_raw": 0.06, "val_penalty": 0.01},
    {"iter": 2, "val_metric_final": 0.08, "val_geom_mean_ratio": 1.05, "val_min_fold_ratio": 0.95,
     "val_avg_closed_trades_per_fold": 4.0, "val_equity_proxy_mean": 108000, "val_metric_raw": 0.09, "val_penalty": 0.01},
    {"iter": 3, "val_metric_final": 0.12, "val_geom_mean_ratio": 1.08, "val_min_fold_ratio": 0.92,
     "val_avg_closed_trades_per_fold": 5.0, "val_equity_proxy_mean": 112000, "val_metric_raw": 0.13, "val_penalty": 0.01},
])

wf_summary = _build_walkforward_summary(test_search_log, {})
print(wf_summary.to_string())
print()

print("✅ All tests passed!")
