# evaluate_baselines.py

import os
import pandas as pd

from baselines import simulate_buy_and_hold, simulate_sma_crossover
from performance_metrics import calculate_performance_metrics

# --- Configuration ---
TICKER = "AMZN"

# Use the same combined file and slicing as your RL eval
PROCESSED_DATA_PATH = os.path.join("processed", f"{TICKER}_all_splits.csv")

RESULTS_DIR = "results"
BASELINE_METRICS_PATH = os.path.join(RESULTS_DIR, "baseline_metrics.csv")
BASIS_EQUITY_BH_PATH = os.path.join(RESULTS_DIR, f"{TICKER}_bh_equity.csv")
BASIS_EQUITY_SMA_PATH = os.path.join(RESULTS_DIR, f"{TICKER}_sma_equity.csv")

INITIAL_CAPITAL = 100000.0
TRANSACTION_COST_PCT = 0.001
SHORT_WINDOW = 20
LONG_WINDOW = 50

# --- Load Data (DD-MM-YYYY -> dayfirst=True) ---
try:
    df = pd.read_csv(PROCESSED_DATA_PATH, index_col="Date", parse_dates=True, dayfirst=True)
except FileNotFoundError:
    print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}")
    print("Make sure your processing script created *_all_splits.csv.")
    raise SystemExit(1)

# Chronological split identical to your RL workflow
train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)
test_df = df.iloc[train_size + val_size :].copy()

# --- Collect results here ---
all_results = []

# --- Buy & Hold ---
print(f"--- Running 'Buy and Hold' Baseline for {TICKER} ---")
_, bh_history = simulate_buy_and_hold(
    data=test_df,
    initial_capital=INITIAL_CAPITAL,
    transaction_cost_pct=TRANSACTION_COST_PCT,
)
metrics_bh = calculate_performance_metrics(bh_history)
metrics_bh["Strategy"] = "Buy and Hold"
all_results.append(metrics_bh)

# Save equity curve for reference
os.makedirs(RESULTS_DIR, exist_ok=True)
bh_history.to_csv(BASIS_EQUITY_BH_PATH)

# --- SMA Crossover ---
print(f"--- Running 'SMA Crossover' Baseline for {TICKER} ---")
_, sma_history = simulate_sma_crossover(
    data=test_df,
    initial_capital=INITIAL_CAPITAL,
    transaction_cost_pct=TRANSACTION_COST_PCT,
    short_window=SHORT_WINDOW,
    long_window=LONG_WINDOW,
)
metrics_sma = calculate_performance_metrics(sma_history)
metrics_sma["Strategy"] = f"SMA Crossover ({SHORT_WINDOW}/{LONG_WINDOW})"
all_results.append(metrics_sma)

# Save equity curve for reference
sma_history.to_csv(BASIS_EQUITY_SMA_PATH)

# --- Consolidate & Save ---
results_df = pd.DataFrame(all_results).set_index("Strategy")
results_df.to_csv(BASELINE_METRICS_PATH, index=True)

print("\n" + "=" * 50)
print("      CONSOLIDATED BASELINE PERFORMANCE")
print("=" * 50)
with pd.option_context("display.float_format", "{:,.2f}".format):
    print(results_df)
print("=" * 50)
print(f"\nBaseline metrics saved to: {BASELINE_METRICS_PATH}")
print(f"BH equity saved to:  {BASIS_EQUITY_BH_PATH}")
print(f"SMA equity saved to: {BASIS_EQUITY_SMA_PATH}")
