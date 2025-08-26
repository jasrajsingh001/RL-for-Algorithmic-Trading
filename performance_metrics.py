# performance_metrics.py

import pandas as pd
import numpy as np

def calculate_performance_metrics(portfolio_history):
    """
    Calculates key performance indicators (KPIs) from a portfolio's value history.

    Args:
        portfolio_history (pd.Series): A Series of portfolio values, indexed by date.

    Returns:
        dict: A dictionary containing the calculated performance metrics.
    """
    # --- 1. Calculate Daily Returns ---
    # The foundation for most financial metrics. pct_change() calculates the
    # percentage change between the current and a prior element.
    daily_returns = portfolio_history.pct_change().dropna()

    # --- 2. Cumulative Return ---
    # The total return over the entire period.
    # Calculated as (final_value / initial_value) - 1.
    cumulative_return = (portfolio_history.iloc[-1] / portfolio_history.iloc[0]) - 1

    # --- 3. Annualized Return ---
    # This metric standardizes the return to a one-year period, making it
    # comparable across strategies that run for different lengths of time.
    # There are approximately 252 trading days in a year.
    num_trading_days = len(portfolio_history)
    annualized_return = ((1 + cumulative_return) ** (252 / num_trading_days)) - 1

    # --- 4. Annualized Volatility (Risk) ---
    # This measures the "risk" of the strategy. It's the standard deviation
    # of the daily returns, scaled to a yearly figure by multiplying by the
    # square root of the number of trading days in a year.
    annualized_volatility = daily_returns.std() * np.sqrt(252)

    # --- 5. Sharpe Ratio ---
    # The gold standard for risk-adjusted return. It measures the excess
    # return per unit of risk (volatility). A higher Sharpe Ratio is better.
    # We assume a risk-free rate of 0 for simplicity.
    risk_free_rate = 0.0
    # We add a small epsilon to the denominator to avoid division by zero
    # in case the volatility is zero (e.g., a pure cash portfolio).
    sharpe_ratio = (annualized_return - risk_free_rate) / (annualized_volatility + 1e-9)

    # --- 6. Maximum Drawdown ---
    # This measures the largest peak-to-trough decline in the portfolio's value,
    # representing the worst-case loss from a single peak.
    # First, we calculate the running maximum of the portfolio value.
    running_max = portfolio_history.cummax()
    # Then, we calculate the drawdown at each point in time.
    drawdown = (portfolio_history - running_max) / running_max
    max_drawdown = drawdown.min() # The maximum drawdown is the minimum value of the drawdown series.

    metrics = {
        'Cumulative Return (%)': cumulative_return * 100,
        'Annualized Return (%)': annualized_return * 100,
        'Annualized Volatility (%)': annualized_volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown (%)': max_drawdown * 100,
        'Final Portfolio Value': portfolio_history.iloc[-1]
    }
    
    return metrics
