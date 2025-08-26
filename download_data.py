# Import necessary libraries
import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# --- 1. Define Tickers and Time Period ---

# Define the list of stock tickers for our portfolio.
# We're using a diversified list of well-known, high-liquidity stocks from different sectors.
# AAPL (Tech), MSFT (Tech), GOOGL (Comm. Services), AMZN (Consumer Disc.), JPM (Financials), JNJ (Healthcare)
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ']

# Define the time range for our historical data.
# A period of 5-10 years is generally a good starting point for training a trading model.
# It's long enough to capture different market cycles (bull, bear, sideways).
START_DATE = '2014-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d') # Use today's date as the end date

# --- 2. Create a Directory to Store Data ---

# Define the path for the directory where we will save our data files.
# It's a best practice to keep raw data in a separate folder.
DATA_PATH = 'data'

# Create the directory if it doesn't already exist.
# The `exist_ok=True` argument prevents an error from being raised if the directory is already there.
os.makedirs(DATA_PATH, exist_ok=True)

# --- 3. Download and Save Data for Each Ticker ---

print("Starting data download...")

# Loop through each ticker in our list
for ticker in TICKERS:
    print(f"Downloading data for {ticker}...")
    try:
        # Use yfinance's download function.
        # It returns a pandas DataFrame with the historical data.
        # We specify the ticker, start date, end date, and the interval (daily).
        data = yf.download(ticker, start=START_DATE, end=END_DATE, interval='1d')

        # Check if the download was successful and the DataFrame is not empty.
        if data.empty:
            print(f"  - No data found for {ticker}. It might be delisted or the ticker symbol is incorrect.")
            # The 'continue' keyword skips the rest of the loop for this ticker and moves to the next one.
            continue
        
        # Define the full path for the CSV file.
        # e.g., 'data/AAPL.csv'
        file_path = os.path.join(DATA_PATH, f"{ticker}.csv")
        
        # Save the DataFrame to a CSV file.
        # The `index=True` argument ensures that the date index is saved as the first column in the file.
        data.to_csv(file_path)
        
        print(f"  - Data for {ticker} saved successfully to {file_path}")
        
    except Exception as e:
        # Catch any other potential errors during download or saving.
        print(f"  - Failed to download or save data for {ticker}. Error: {e}")

print("\nData download process complete.")