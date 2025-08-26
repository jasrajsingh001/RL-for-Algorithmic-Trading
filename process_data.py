# process_data.py

# Import necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# --- 1. Define Constants and Load Data ---

DATA_PATH = 'data'
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ']

dataframes = {}

print("Loading data into DataFrames...")

for ticker in TICKERS:
    file_path = os.path.join(DATA_PATH, f"{ticker}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        # IMPORTANT: Preserve the true price for portfolio math before any transforms
        df["Close_raw"] = df["Close"]
        dataframes[ticker] = df
        print(f"  - Successfully loaded {ticker}. Shape: {df.shape}")
    else:
        print(f"  - Warning: Could not find data file for {ticker} at {file_path}")

print("\nData loading complete.")

# --- 2. Initial Inspection (optional prints) ---
if 'AAPL' in dataframes:
    print("\n--- Inspecting AAPL DataFrame ---")
    print("First 5 rows:")
    print(dataframes['AAPL'].head())
    print("\nDataFrame Info:")
    dataframes['AAPL'].info()

# --- 3. Verify Data Integrity: Check for Missing Values (NaNs) ---
print("\n--- Verifying Data Integrity ---")
for ticker, df in dataframes.items():
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    if total_missing > 0:
        print(f"\nMissing values found in {ticker}:")
        print(missing_values[missing_values > 0])
    else:
        print(f"  - No missing values found in {ticker} DataFrame. It's clean!")
print("\nData integrity check complete.")

# --- 4. Handle Missing Data with Forward-Fill ---
print("\n--- Handling Missing Data ---")
for ticker, df in dataframes.items():
    initial_missing = df.isnull().sum().sum()
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    final_missing = df.isnull().sum().sum()
    if initial_missing > 0:
        print(f"  - {ticker}: Filled {initial_missing} missing value(s).")
    if final_missing == 0:
        print(f"  - {ticker}: Data is now complete. No missing values remain.")
    else:
        print(f"  - WARNING: {ticker} still has {final_missing} missing values after filling.")
print("\nMissing data handling complete.")

# --- 5. Feature Engineering: Simple Moving Averages (SMAs) ---
print("\n--- Engineering Features: Simple Moving Averages (SMAs) ---")
SMA_WINDOWS = [20, 50]
for ticker, df in dataframes.items():
    print(f"  - Calculating SMAs for {ticker}...")
    for window in SMA_WINDOWS:
        df[f"SMA_{window}"] = df['Close'].rolling(window=window).mean()
print("\nSMA feature engineering complete.")

# (Optional inspection)
if 'AAPL' in dataframes:
    print("\n--- Inspecting AAPL DataFrame After Adding SMAs ---")
    print("Last 5 rows with new SMA features:")
    print(dataframes['AAPL'].tail())
    print("\nFirst 25 rows to show initial NaN values in SMA columns:")
    print(dataframes['AAPL'].head(25))
print("\nSMA feature engineering complete.")

# --- 6. Feature Engineering: Exponential Moving Averages (EMAs) ---
print("\n--- Engineering Features: Exponential Moving Averages (EMAs) ---")
EMA_SPANS = [20, 50]
for ticker, df in dataframes.items():
    print(f"  - Calculating EMAs for {ticker}...")
    for span in EMA_SPANS:
        df[f"EMA_{span}"] = df['Close'].ewm(span=span, adjust=False).mean()
print("\nEMA feature engineering complete.")

# (Optional inspection)
if 'AAPL' in dataframes:
    print("\n--- Inspecting AAPL DataFrame After Adding EMAs ---")
    print("Last 5 rows with new SMA and EMA features:")
    print(dataframes['AAPL'].tail())
    print("\nFirst 25 rows to show initial EMA values:")
    print(dataframes['AAPL'].head(25))

# --- 7. Feature Engineering: Relative Strength Index (RSI) ---
print("\n--- Engineering Features: Relative Strength Index (RSI) ---")
RSI_WINDOW = 14
for ticker, df in dataframes.items():
    print(f"  - Calculating RSI for {ticker}...")
    delta = df['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(com=RSI_WINDOW - 1, min_periods=RSI_WINDOW).mean()
    avg_loss = loss.ewm(com=RSI_WINDOW - 1, min_periods=RSI_WINDOW).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
print("\nRSI feature engineering complete.")

if 'AAPL' in dataframes:
    print("\n--- Inspecting AAPL DataFrame After Adding RSI ---")
    print(dataframes['AAPL'][['Close', 'SMA_50', 'EMA_50', 'RSI']].tail())
    print("\n")
    print(dataframes['AAPL'][['Close', 'RSI']].head(20))

# --- 8. Feature Engineering: MACD ---
print("\n--- Engineering Features: Moving Average Convergence Divergence (MACD) ---")
MACD_SHORT_WINDOW = 12
MACD_LONG_WINDOW = 26
MACD_SIGNAL_WINDOW = 9
for ticker, df in dataframes.items():
    print(f"  - Calculating MACD for {ticker}...")
    ema_short = df['Close'].ewm(span=MACD_SHORT_WINDOW, adjust=False).mean()
    ema_long  = df['Close'].ewm(span=MACD_LONG_WINDOW, adjust=False).mean()
    df['MACD'] = ema_short - ema_long
    df['MACD_Signal'] = df['MACD'].ewm(span=MACD_SIGNAL_WINDOW, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
print("\nMACD feature engineering complete.")

if 'AAPL' in dataframes:
    print("\n--- Inspecting AAPL DataFrame After Adding MACD ---")
    print(dataframes['AAPL'][['Close', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']].tail(10))

# --- 9. Handle Initial NaN values (drop warm-up rows from indicators) ---
print("\n--- Handling Initial NaN Values ---")
for ticker, df in dataframes.items():
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)
    print(f"  - {ticker}: Dropped {rows_before - rows_after} initial rows with NaN indicator values.")

# --- 10. Chronological Data Splitting (Train, Validation, Test) ---
print("\n--- Splitting Data Chronologically ---")
TRAIN_PCT = 0.70
VALIDATION_PCT = 0.15
train_data, validation_data, test_data = {}, {}, {}

for ticker, df in dataframes.items():
    train_end_idx = int(len(df) * TRAIN_PCT)
    validation_end_idx = int(len(df) * (TRAIN_PCT + VALIDATION_PCT))
    train_data[ticker] = df.iloc[:train_end_idx].copy()
    validation_data[ticker] = df.iloc[train_end_idx:validation_end_idx].copy()
    test_data[ticker] = df.iloc[validation_end_idx:].copy()
    print(f"  - {ticker}:")
    print(f"    - Training set shape:   {train_data[ticker].shape}")
    print(f"    - Validation set shape: {validation_data[ticker].shape}")
    print(f"    - Test set shape:       {test_data[ticker].shape}")

# --- 11. Normalize Data (Correctly; no leakage; preserve Close_raw) ---
print("\n--- Normalizing Data Sets ---")

# Features the AGENT will use (do NOT include Close_raw here)
COLUMNS_TO_NORMALIZE = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist'
]

# (A) Log-transform Volume (stabilizes scale) BEFORE fitting scaler
for ticker in TICKERS:
    for split_dict in (train_data, validation_data, test_data):
        if ticker in split_dict and 'Volume' in split_dict[ticker].columns:
            split_dict[ticker]['Volume'] = np.log1p(split_dict[ticker]['Volume'].clip(lower=0))

# (B) Fit MinMax on TRAIN only; transform all splits (per ticker)
for ticker in TICKERS:
    if ticker not in train_data:
        continue
    scaler = MinMaxScaler()
    scaler.fit(train_data[ticker][COLUMNS_TO_NORMALIZE])

    train_data[ticker][COLUMNS_TO_NORMALIZE]      = scaler.transform(train_data[ticker][COLUMNS_TO_NORMALIZE])
    validation_data[ticker][COLUMNS_TO_NORMALIZE] = scaler.transform(validation_data[ticker][COLUMNS_TO_NORMALIZE])
    test_data[ticker][COLUMNS_TO_NORMALIZE]       = scaler.transform(test_data[ticker][COLUMNS_TO_NORMALIZE])

    # (C) Create Close_norm from the scaled Close; keep Close_raw untouched
    for split_dict in (train_data, validation_data, test_data):
        split = split_dict[ticker]
        split['Close_norm'] = split['Close']  # agent-friendly normalized close
        # Keep 'Close' as-is (scaled) to avoid breaking any downstream expectations
        # Critical: 'Close_raw' remains the true price for the env's accounting

    print(f"  - {ticker}: Train/Val/Test normalized (Close_raw preserved, Close_norm created).")

# --- 12. Final Inspection ---
print("\n--- Final Inspection of Normalized Data (AAPL test) ---")
if 'AAPL' in test_data:
    print(test_data['AAPL'][COLUMNS_TO_NORMALIZE + ['Close_norm', 'Close_raw']].describe())

# --- 13. Save per-split CSVs (keep Close_raw) ---
os.makedirs("processed", exist_ok=True)
for ticker in TICKERS:
    if ticker in train_data:
        train_data[ticker].to_csv(f"processed/{ticker}_train.csv")
        validation_data[ticker].to_csv(f"processed/{ticker}_validation.csv")
        test_data[ticker].to_csv(f"processed/{ticker}_test.csv")
        print(f"Saved: {ticker}_train.csv, {ticker}_validation.csv, {ticker}_test.csv")

# --- 14. Save combined *_all_splits.csv with ISO dates and Close_raw ---
print("\n--- Saving combined *_all_splits.csv with ISO dates ---")

splits = {"train": train_data, "validation": validation_data, "test": test_data}

for ticker in TICKERS:
    parts = []
    for split_name in ("train", "validation", "test"):
        if ticker not in splits[split_name]:
            continue
        df = splits[split_name][ticker].copy()
        # Move index to Date column
        df = df.reset_index().rename(columns={df.index.name or "index": "Date"})
        # Ensure ISO date strings
        df["Date"] = df["Date"].astype(str)
        # Put Date first
        cols = ["Date"] + [c for c in df.columns if c != "Date"]
        df = df[cols]
        parts.append(df)

    combined = pd.concat(parts, ignore_index=True)

    # Preferred order: Close_raw for accounting, features for agent, Close_norm explicit
    prefer_order = [
        "Date", "Close_raw", "Open", "High", "Low", "Volume",
        "SMA_20", "SMA_50", "EMA_20", "EMA_50",
        "RSI", "MACD", "MACD_Signal", "MACD_Hist", "Close", "Close_norm"
    ]
    cols = [c for c in prefer_order if c in combined.columns] + \
           [c for c in combined.columns if c not in prefer_order]
    combined = combined[cols]

    combined.to_csv(f"processed/{ticker}_all_splits.csv", index=False)
    print(f"✅ Saved: processed/{ticker}_all_splits.csv")

print("\nAll done.")
