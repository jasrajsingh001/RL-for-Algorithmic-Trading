# analyze.py

import os
import pandas as pd
import numpy as np

# --- Plotting ---
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


# =========================
#   Metric Calculations
# =========================
def calculate_cumulative_return(portfolio_values: pd.Series) -> float:
    if portfolio_values.empty:
        return 0.0
    starting_value, ending_value = portfolio_values.iloc[0], portfolio_values.iloc[-1]
    if starting_value == 0:
        return 0.0
    return (ending_value / starting_value) - 1


def calculate_annualized_return(portfolio_values: pd.Series) -> float:
    if portfolio_values.empty:
        return 0.0
    starting_value, ending_value = portfolio_values.iloc[0], portfolio_values.iloc[-1]
    start_date, end_date = portfolio_values.index[0], portfolio_values.index[-1]
    num_days = (end_date - start_date).days
    if num_days == 0:
        return 0.0
    num_years = num_days / 365.25
    total_return_factor = ending_value / starting_value
    return (max(0, total_return_factor)) ** (1 / num_years) - 1


def calculate_annualized_volatility(portfolio_values: pd.Series) -> float:
    if portfolio_values.empty or len(portfolio_values) < 2:
        return 0.0
    daily_returns = portfolio_values.pct_change().dropna()
    return daily_returns.std() * np.sqrt(252)


def calculate_sharpe_ratio(portfolio_values: pd.Series, risk_free_rate: float) -> float:
    ann_ret = calculate_annualized_return(portfolio_values)
    ann_vol = calculate_annualized_volatility(portfolio_values)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    if portfolio_values.empty:
        return 0.0
    hwm = portfolio_values.cummax()
    dd = (portfolio_values - hwm) / hwm
    return dd.min()


def calculate_sortino_ratio(portfolio_values: pd.Series, risk_free_rate: float) -> float:
    ann_ret = calculate_annualized_return(portfolio_values)
    daily_returns = portfolio_values.pct_change().dropna()
    target_daily = risk_free_rate / 252.0
    downside = (target_daily - daily_returns).clip(lower=0)
    downside_var = (downside ** 2).mean()
    daily_downside_dev = np.sqrt(downside_var)
    ann_downside_dev = daily_downside_dev * np.sqrt(252)
    if ann_downside_dev == 0:
        return np.inf
    return (ann_ret - risk_free_rate) / ann_downside_dev


# =========================
#   Baseline Simulations
# =========================
# =========================
#   Baseline Simulations
# =========================
def _price_series(df: pd.DataFrame) -> pd.Series:
    """Use Close_raw if present, otherwise Close (with a warning)."""
    if "Close_raw" in df.columns:
        return df["Close_raw"]
    if "Close" in df.columns:
        print("Warning: Close_raw not found; falling back to Close for baselines.")
        return df["Close"]
    raise ValueError("Neither 'Close_raw' nor 'Close' found in processed data.")

def simulate_buy_and_hold(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    px = _price_series(df)
    first_price = float(px.iloc[0])
    if first_price <= 0:
        raise ValueError("First price is non-positive.")
    shares = initial_capital / first_price
    pv = shares * px
    pv.name = "portfolio_value"
    return pv

def simulate_sma_crossover(
    df: pd.DataFrame,
    initial_capital: float,
    short_window: int,
    long_window: int,
    transaction_cost_pct: float,
) -> pd.Series:
    sma_short_col, sma_long_col = f"SMA_{short_window}", f"SMA_{long_window}"
    if sma_short_col not in df.columns or sma_long_col not in df.columns:
        raise ValueError(f"Missing SMA columns: {sma_short_col} or {sma_long_col}")

    px = _price_series(df)
    position = (df[sma_short_col] > df[sma_long_col]).astype(float)
    signal = position.diff().fillna(0.0)

    cash, shares = float(initial_capital), 0.0
    pv_list = []
    for ts in df.index:
        price = float(px.loc[ts])
        sig = float(signal.loc[ts])

        if sig == 1.0 and cash > 0:
            notional = cash / (1 + transaction_cost_pct)
            qty = (notional / price) if price > 0 else 0.0
            shares += qty
            cash -= qty * price * (1 + transaction_cost_pct)

        elif sig == -1.0 and shares > 0:
            proceeds = shares * price * (1 - transaction_cost_pct)
            cash += proceeds
            shares = 0.0

        pv_list.append(cash + shares * price)

    return pd.Series(pv_list, index=df.index, name="portfolio_value")



# =========================
#   Visualization
# =========================
def plot_equity_curves(strategies: dict, title: str):
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x:,.0f}"))
    for name, values in strategies.items():
        ax.plot(values.index, values, label=name)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.legend(fontsize=12)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "equity_curve_comparison.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nEquity curve plot saved to: {out_path}")
    plt.show()


# =========================
#   Main
# =========================
def main():
    TICKER = "AAPL"
    RISK_FREE_RATE = 0.02
    INITIAL_CAPITAL = 100000.0
    TRANSACTION_COST_PCT = 0.001
    SHORT_WINDOW, LONG_WINDOW = 20, 50

    # --- RL results ---
    rl_original_path = os.path.join("results", "backtest_results.csv")
    rl_tuned_path = os.path.join("results", "backtest_results_tuned.csv")

    print(f"Loading ORIGINAL RL Agent backtest results from: {rl_original_path}")
    try:
        rl_original_df = pd.read_csv(rl_original_path, index_col="date", parse_dates=True)
    except FileNotFoundError:
        print("Error: Original RL results not found. Run `evaluate.py` first.")
        return

    tuned_available = True
    print(f"Loading TUNED RL Agent backtest results from: {rl_tuned_path}")
    try:
        rl_tuned_df = pd.read_csv(rl_tuned_path, index_col="date", parse_dates=True)
    except FileNotFoundError:
        tuned_available = False
        print("Warning: Tuned RL results not found. Proceeding without tuned comparison.")

    # --- Market data for baselines (use dayfirst=True for your DD-MM-YYYY files) ---
    data_path = os.path.join("processed", f"{TICKER}_all_splits.csv")
    print(f"Loading processed market data from: {data_path}")
    try:
        processed_df = pd.read_csv(data_path, index_col="Date", parse_dates=True, dayfirst=True)
    except FileNotFoundError:
        print("Error: Processed data not found. Run the data processing script from Step 2.")
        return

    # Align baselines to RL original test period
    test_start = rl_original_df.index[0]
    test_end = rl_original_df.index[-1]
    test_df = processed_df.loc[test_start:test_end].copy()

    print("\nData loaded successfully. Starting analysis...")

    # --- Baselines ---
    print("Simulating baseline strategies on the test period...")
    buy_and_hold_values = simulate_buy_and_hold(test_df, INITIAL_CAPITAL)
    sma_crossover_values = simulate_sma_crossover(
        test_df, INITIAL_CAPITAL, SHORT_WINDOW, LONG_WINDOW, TRANSACTION_COST_PCT
    )

    # --- Gather strategies & align to the SAME index ---
    # --- Gather strategies & align to the SAME index ---
    strategies = {
        "RL Agent (Original)": rl_original_df.get("portfolio_value"),
        "Buy and Hold": buy_and_hold_values,
        "SMA Crossover": sma_crossover_values,
    }
    if tuned_available:
        strategies["RL Agent (Tuned)"] = rl_tuned_df.get("portfolio_value")

    # Build intersection only over non-empty series
    series_list = [s for s in strategies.values() if isinstance(s, pd.Series) and not s.empty]
    if not series_list:
        print("Nothing to compare: all series empty.")
        return
    common_index = series_list[0].index
    for s in series_list[1:]:
        common_index = common_index.intersection(s.index)

    strategies = {k: v.reindex(common_index).dropna() for k, v in strategies.items() if v is not None}


    # # Align all to the intersection of dates (avoid NaNs)
    # common_index = None
    # for s in strategies.values():
    #     common_index = s.index if common_index is None else common_index.intersection(s.index)
    # strategies = {k: v.reindex(common_index).dropna() for k, v in strategies.items()}

    # --- KPIs ---
    performance_data = {}
    for name, values in strategies.items():
        print(f"Calculating KPIs for {name}...")
        performance_data[name] = {
            "Cumulative Return": calculate_cumulative_return(values),
            "Annualized Return (CAGR)": calculate_annualized_return(values),
            "Annualized Volatility": calculate_annualized_volatility(values),
            "Sharpe Ratio": calculate_sharpe_ratio(values, RISK_FREE_RATE),
            "Sortino Ratio": calculate_sortino_ratio(values, RISK_FREE_RATE),
            "Maximum Drawdown": calculate_max_drawdown(values),
        }

    performance_df = pd.DataFrame.from_dict(performance_data, orient="index")

    # Format for display
    performance_df["Cumulative Return"] = performance_df["Cumulative Return"].map("{:.2%}".format)
    performance_df["Annualized Return (CAGR)"] = performance_df["Annualized Return (CAGR)"].map("{:.2%}".format)
    performance_df["Annualized Volatility"] = performance_df["Annualized Volatility"].map("{:.2%}".format)
    performance_df["Maximum Drawdown"] = performance_df["Maximum Drawdown"].map("{:.2%}".format)
    performance_df["Sharpe Ratio"] = performance_df["Sharpe Ratio"].map("{:.2f}".format)
    performance_df["Sortino Ratio"] = performance_df["Sortino Ratio"].map("{:.2f}".format)

    print("\n\n--- Strategy Performance Comparison ---")
    print(performance_df)

    os.makedirs("results", exist_ok=True)
    perf_path = os.path.join("results", "performance_comparison.csv")
    performance_df.to_csv(perf_path)
    print(f"\nPerformance comparison table saved to: {perf_path}")

    # --- Plot equity curves ---
    plot_equity_curves(strategies, f"Equity Curve Comparison: {TICKER}")


if __name__ == "__main__":
    main()
