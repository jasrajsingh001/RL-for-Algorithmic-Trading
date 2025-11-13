# performance_metrics.py

import pandas as pd
import numpy as np

def calculate_performance_metrics(portfolio_history: pd.Series, risk_free_rate: float = 0.02) -> dict:
    """
    Calculate common performance metrics from a portfolio equity curve.

    Args:
        portfolio_history (pd.Series): Series of portfolio values indexed by datetime.
        risk_free_rate (float): Annualized risk-free rate (e.g., 0.02 for 2%).

    Returns:
        dict: Performance metrics (raw floats, not formatted).
    """
    # --- Basic guards ---
    if portfolio_history is None or len(portfolio_history) < 2:
        return {
            'Cumulative Return (%)': 0.0,
            'Annualized Return (%)': 0.0,
            'Annualized Volatility (%)': 0.0,
            'Sharpe Ratio': 0.0,
            'Sortino Ratio': 0.0,
            'Maximum Drawdown (%)': 0.0,
            'Final Portfolio Value': float(portfolio_history.iloc[-1]) if len(portfolio_history) else 0.0,
        }

    # Ensure monotonic datetime index (helps if upstream left it unsorted)
    if not portfolio_history.index.is_monotonic_increasing:
        portfolio_history = portfolio_history.sort_index()

    # Convert to float to avoid object dtype surprises
    pv = pd.to_numeric(portfolio_history, errors='coerce').dropna()

    if len(pv) < 2:
        return {
            'Cumulative Return (%)': 0.0,
            'Annualized Return (%)': 0.0,
            'Annualized Volatility (%)': 0.0,
            'Sharpe Ratio': 0.0,
            'Sortino Ratio': 0.0,
            'Maximum Drawdown (%)': 0.0,
            'Final Portfolio Value': float(pv.iloc[-1]) if len(pv) else 0.0,
        }

    # --- Daily returns ---
    daily_returns = pv.pct_change().dropna()

    # --- Cumulative return ---
    start_val, end_val = pv.iloc[0], pv.iloc[-1]
    cumulative_return = (end_val / start_val) - 1.0 if start_val != 0 else 0.0

    # --- Annualized return (CAGR) using calendar time, consistent with analyze.py ---
    start_date, end_date = pv.index[0], pv.index[-1]
    num_days = (end_date - start_date).days
    if num_days > 0 and start_val > 0:
        years = num_days / 365.25
        total_ret_factor = end_val / start_val
        annualized_return = (max(0.0, total_ret_factor)) ** (1.0 / years) - 1.0
    else:
        annualized_return = 0.0

    # --- Annualized volatility ---
    annualized_volatility = float(daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 1 else 0.0

    # --- Sharpe ratio (annual) ---
    sharpe_ratio = 0.0
    if annualized_volatility > 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    # --- Sortino ratio (annual) ---
    # Downside deviation (annualized) with target = risk_free_rate/252 per day
    if len(daily_returns) > 1:
        target_daily = risk_free_rate / 252.0
        downside = (target_daily - daily_returns).clip(lower=0.0)
        downside_var = (downside ** 2).mean()
        daily_down_dev = np.sqrt(downside_var) if downside_var >= 0 else 0.0
        ann_down_dev = float(daily_down_dev * np.sqrt(252))
        sortino_ratio = (annualized_return - risk_free_rate) / ann_down_dev if ann_down_dev > 0 else np.inf
    else:
        sortino_ratio = 0.0

    # --- Max drawdown ---
    running_max = pv.cummax()
    drawdown = (pv - running_max) / running_max.replace(0, np.nan)
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    return {
        'Cumulative Return (%)': cumulative_return * 100.0,
        'Annualized Return (%)': annualized_return * 100.0,
        'Annualized Volatility (%)': annualized_volatility * 100.0,
        'Sharpe Ratio': float(sharpe_ratio) if np.isfinite(sharpe_ratio) else 0.0,
        'Sortino Ratio': float(sortino_ratio) if np.isfinite(sortino_ratio) else 0.0,
        'Maximum Drawdown (%)': max_drawdown * 100.0,
        'Final Portfolio Value': float(end_val),
    }
