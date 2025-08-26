# baselines.py

import pandas as pd
import numpy as np

def simulate_buy_and_hold(data, initial_capital, transaction_cost_pct):
    """
    Simulates a Buy and Hold strategy, including transaction costs.

    This function buys the maximum possible number of shares on the first day
    and holds them until the end, calculating the portfolio value at each step.

    Args:
        data (pd.DataFrame): The preprocessed test dataset. Must contain an 'Adj Close' column.
                             The DataFrame's index should be the timestamps.
        initial_capital (float): The starting amount of cash.
        transaction_cost_pct (float): The percentage cost for each transaction (buy or sell).

    Returns:
        tuple: A tuple containing:
            - final_portfolio_value (float): The final liquidated value of the portfolio.
            - portfolio_history (pd.Series): A Series with the portfolio's value at each time step,
                                             indexed by date.
    """
    # Initialize our portfolio state variables
    cash = initial_capital
    shares_held = 0
    # This list will store the value of our portfolio at the end of each day
    portfolio_history = []

    # --- Day 1: The Initial Purchase ---
    # Get the price on the very first day of the simulation period
    first_day_price = data['Close'].iloc[0]
    
    # Calculate the cost of one share, including the transaction fee
    # We multiply by (1 + cost) because the fee is added to the price
    cost_per_share = first_day_price * (1 + transaction_cost_pct)
    
    # Calculate the maximum number of shares we can afford
    if cost_per_share > 0:
        # We use floor division (//) because we can only buy whole shares
        shares_to_buy = cash // cost_per_share
    else:
        shares_to_buy = 0
        
    # Update our portfolio state after the purchase
    if shares_to_buy > 0:
        # Deduct the total cost of the purchase from our cash
        cash -= shares_to_buy * cost_per_share
        shares_held = shares_to_buy

    # Record the initial value of the portfolio on day 1
    # This value is our remaining cash plus the market value of the shares we just bought.
    initial_portfolio_value = cash + (shares_held * first_day_price)
    portfolio_history.append(initial_portfolio_value)
    
    # --- Days 2 to N: The Holding Period ---
    # We loop through the rest of the data, starting from the second day (index 1)
    for i in range(1, len(data)):
        # Get the closing price for the current day
        current_price = data['Close'].iloc[i]
        
        # In a hold period, our cash doesn't change.
        # The portfolio's value changes only because the value of our held shares changes.
        current_portfolio_value = cash + (shares_held * current_price)
        portfolio_history.append(current_portfolio_value)
        
    # --- Final Value Calculation (Liquidation) ---
    # To get a final, concrete value, we simulate selling all shares on the last day.
    last_day_price = data['Close'].iloc[-1]
    
    # Calculate the proceeds from selling all our shares, accounting for the transaction fee.
    # We multiply by (1 - cost) because the fee is deducted from the sale price.
    sell_proceeds = (shares_held * last_day_price) * (1 - transaction_cost_pct)
    
    # The final value is our leftover cash plus the proceeds from this final sale.
    final_portfolio_value = cash + sell_proceeds

    # Create a pandas Series from our history list, using the original DataFrame's index.
    # This is crucial for plotting later, as it aligns our values with the correct dates.
    portfolio_history_series = pd.Series(portfolio_history, index=data.index)
    
    return final_portfolio_value, portfolio_history_series

# highlight-start
def simulate_sma_crossover(data, initial_capital, transaction_cost_pct, short_window, long_window):
    """
    Simulates an SMA Crossover strategy.

    Buys when the short-term SMA crosses above the long-term SMA (Golden Cross).
    Sells when the short-term SMA crosses below the long-term SMA (Death Cross).

    Args:
        data (pd.DataFrame): The test dataset. Must contain 'Adj Close',
                             and SMA columns named f'SMA_{short_window}' and f'SMA_{long_window}'.
        initial_capital (float): The starting amount of cash.
        transaction_cost_pct (float): The percentage cost for each transaction.
        short_window (int): The lookback period for the short-term SMA.
        long_window (int): The lookback period for the long-term SMA.

    Returns:
        tuple: A tuple containing:
            - final_portfolio_value (float): The final liquidated value of the portfolio.
            - portfolio_history (pd.Series): A Series with the portfolio's value at each time step.
    """
    # Define the names of the SMA columns based on the window sizes
    short_sma_col = f'SMA_{short_window}'
    long_sma_col = f'SMA_{long_window}'
    
    # Ensure the required SMA columns exist in the data
    if short_sma_col not in data.columns or long_sma_col not in data.columns:
        raise ValueError(f"Data must contain '{short_sma_col}' and '{long_sma_col}' columns.")

    # Initialize portfolio state
    cash = initial_capital
    shares_held = 0
    position = 'neutral'  # Can be 'neutral' or 'long'
    portfolio_history = []

    # Loop through the data starting from the first day
    for i in range(len(data)):
        current_price = data['Close'].iloc[i]
        
        # We need at least one previous day to check for a crossover
        if i > 0:
            # Get SMA values for the current and previous day
            short_sma_current = data[short_sma_col].iloc[i]
            long_sma_current = data[long_sma_col].iloc[i]
            short_sma_previous = data[short_sma_col].iloc[i-1]
            long_sma_previous = data[long_sma_col].iloc[i-1]

            # --- Check for Trading Signals ---
            # Golden Cross: Buy signal
            if short_sma_current > long_sma_current and short_sma_previous < long_sma_previous and position == 'neutral':
                # Calculate cost and number of shares to buy with all available cash
                cost_per_share = current_price * (1 + transaction_cost_pct)
                if cost_per_share > 0:
                    shares_to_buy = cash // cost_per_share
                    if shares_to_buy > 0:
                        cash -= shares_to_buy * cost_per_share
                        shares_held = shares_to_buy
                        position = 'long' # Update position state

            # Death Cross: Sell signal
            elif short_sma_current < long_sma_current and short_sma_previous > long_sma_previous and position == 'long':
                # Sell all held shares
                sell_proceeds = (shares_held * current_price) * (1 - transaction_cost_pct)
                cash += sell_proceeds
                shares_held = 0
                position = 'neutral' # Update position state

        # Calculate portfolio value for the current day regardless of trading
        current_portfolio_value = cash + (shares_held * current_price)
        portfolio_history.append(current_portfolio_value)

    # --- Final Value Calculation (Liquidation) ---
    # To ensure a fair comparison, we liquidate any final holdings
    last_day_price = data['Close'].iloc[-1]
    if shares_held > 0:
        sell_proceeds = (shares_held * last_day_price) * (1 - transaction_cost_pct)
        cash += sell_proceeds
    
    final_portfolio_value = cash
    
    # Create the pandas Series for the portfolio history
    portfolio_history_series = pd.Series(portfolio_history, index=data.index)
    
    return final_portfolio_value, portfolio_history_series
# highlight-end