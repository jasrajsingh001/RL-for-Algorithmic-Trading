# baselines.py

import pandas as pd
import numpy as np

def _price_series(df: pd.DataFrame) -> pd.Series:
    """
    Prefer Close_raw (true price). Fall back to Close with a warning.
    """
    if "Close_raw" in df.columns:
        return df["Close_raw"]
    if "Close" in df.columns:
        print("Warning: Close_raw not found; falling back to Close.")
        return df["Close"]
    raise ValueError("Neither 'Close_raw' nor 'Close' found in data.")

def simulate_buy_and_hold(data: pd.DataFrame,
                          initial_capital: float,
                          transaction_cost_pct: float):
    """
    Buy-and-hold baseline using true prices (Close_raw) and fractional shares.
    Buys on day 1 (paying fees), holds to last day, then liquidates (paying fees).
    Returns: (final_portfolio_value, portfolio_history_series)
    """
    px = _price_series(data)
    if len(px) == 0:
        return float(initial_capital), pd.Series(dtype=float)

    cash = float(initial_capital)
    shares = 0.0
    history = []

    # --- Initial buy (fractional) ---
    p0 = float(px.iloc[0])
    if p0 > 0:
        # spend all cash accounting for fee on buy
        shares = cash / (p0 * (1.0 + transaction_cost_pct))
        spend = shares * p0 * (1.0 + transaction_cost_pct)
        cash -= spend
    # record day 1 portfolio value (no sell yet)
    history.append(cash + shares * p0)

    # --- Hold period ---
    for i in range(1, len(px)):
        p = float(px.iloc[i])
        history.append(cash + shares * p)

    # --- Liquidate on last day, including fee ---
    plast = float(px.iloc[-1])
    proceeds = shares * plast * (1.0 - transaction_cost_pct)
    final_value = cash + proceeds

    series = pd.Series(history, index=data.index, name="portfolio_value")
    return float(final_value), series


def simulate_sma_crossover(data: pd.DataFrame,
                           initial_capital: float,
                           transaction_cost_pct: float,
                           short_window: int,
                           long_window: int):
    """
    SMA crossover baseline using true prices (Close_raw) and fractional shares.
    Golden cross: go long (all-in). Death cross: exit (all-out).
    Fees applied on both entries and exits.
    Returns: (final_portfolio_value, portfolio_history_series)
    """
    sma_s = f"SMA_{short_window}"
    sma_l = f"SMA_{long_window}"
    if sma_s not in data.columns or sma_l not in data.columns:
        raise ValueError(f"Missing SMA columns: {sma_s}, {sma_l}")

    px = _price_series(data)
    if len(px) == 0:
        return float(initial_capital), pd.Series(dtype=float)

    cash = float(initial_capital)
    shares = 0.0
    long_on = False
    history = []

    # iterate over dates
    for i in range(len(px)):
        p = float(px.iloc[i])

        if i > 0:
            s_prev, l_prev = data[sma_s].iloc[i-1], data[sma_l].iloc[i-1]
            s_now,  l_now  = data[sma_s].iloc[i],   data[sma_l].iloc[i]

            # Golden cross: enter long (all-in)
            if (s_now > l_now) and (s_prev <= l_prev) and not long_on and p > 0:
                # use all cash accounting for buy fee
                qty = cash / (p * (1.0 + transaction_cost_pct))
                spend = qty * p * (1.0 + transaction_cost_pct)
                if qty > 0 and spend <= cash + 1e-9:
                    shares += qty
                    cash -= spend
                    long_on = True

            # Death cross: exit long (all-out)
            elif (s_now < l_now) and (s_prev >= l_prev) and long_on and shares > 0:
                proceeds = shares * p * (1.0 - transaction_cost_pct)
                cash += proceeds
                shares = 0.0
                long_on = False

        # mark-to-market
        history.append(cash + shares * p)

    # Liquidate at the end if still long
    if shares > 0.0:
        plast = float(px.iloc[-1])
        proceeds = shares * plast * (1.0 - transaction_cost_pct)
        cash += proceeds
        shares = 0.0

    final_value = cash
    series = pd.Series(history, index=data.index, name="portfolio_value")
    return float(final_value), series
