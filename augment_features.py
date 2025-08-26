# augment_features.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = "processed"
# edit list if you want fewer/more tickers
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ"]

# the new features we will compute + min-max scale (fit on train only)
NEW_FEATS = ["RET_1D", "VOL_20", "VOL_SHOCK", "TREND_20_50"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # use raw close where available (fallback to Close)
    px = out["Close_raw"] if "Close_raw" in out.columns else out["Close"]

    # 1) 1-day return
    out["RET_1D"] = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 2) 20-day vol of returns
    out["VOL_20"] = out["RET_1D"].rolling(20).std().fillna(0.0)

    # 3) volume shock: z-score over 20d → clamp → map to [0,1]
    if "Volume" in out.columns:
        ma = out["Volume"].rolling(20).mean()
        sd = out["Volume"].rolling(20).std()
        z = (out["Volume"] - ma) / sd.replace(0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3, 3)
        out["VOL_SHOCK"] = (z + 3) / 6.0
    else:
        out["VOL_SHOCK"] = 0.0

    # 4) trend: EMA20/EMA50 ratio (safe fallback = 1.0)
    if "EMA_20" in out.columns and "EMA_50" in out.columns:
        ratio = out["EMA_20"] / out["EMA_50"].replace(0, np.nan)
        out["TREND_20_50"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    else:
        out["TREND_20_50"] = 1.0

    return out

def scale_new_feats(train_df, val_df, test_df):
    scaler = MinMaxScaler()
    scaler.fit(train_df[NEW_FEATS])

    train_df[NEW_FEATS] = scaler.transform(train_df[NEW_FEATS])
    val_df[NEW_FEATS]   = scaler.transform(val_df[NEW_FEATS])
    test_df[NEW_FEATS]  = scaler.transform(test_df[NEW_FEATS])
    return train_df, val_df, test_df

def run_for_ticker(ticker: str):
    path = os.path.join(DATA_DIR, f"{ticker}_all_splits.csv")
    if not os.path.exists(path):
        print(f"skip {ticker}: {path} not found")
        return

    # keep your existing DD-MM-YYYY handling (dayfirst=True)
    df = pd.read_csv(path, parse_dates=["Date"], dayfirst=True).sort_values("Date").set_index("Date")

    df = add_features(df)

    # same splits you use elsewhere: 70/15/15
    n = len(df)
    i_train = int(n * 0.70)
    i_val   = int(n * 0.85)

    train_df = df.iloc[:i_train].copy()
    val_df   = df.iloc[i_train:i_val].copy()
    test_df  = df.iloc[i_val:].copy()

    # fill any initial NaNs from rolling windows in NEW features only
    for c in NEW_FEATS:
        train_df[c] = train_df[c].fillna(method="bfill").fillna(0.0)
        val_df[c]   = val_df[c].fillna(method="bfill").fillna(0.0)
        test_df[c]  = test_df[c].fillna(method="bfill").fillna(0.0)

    # scale NEW features using train only (no leakage)
    train_df, val_df, test_df = scale_new_feats(train_df, val_df, test_df)

    # stitch back and save (keep your date format)
    df_out = pd.concat([train_df, val_df, test_df])
    df_out.reset_index().to_csv(path, index=False, date_format="%d-%m-%Y")
    print(f"✅ augmented & saved: {path}  (+{NEW_FEATS})")

def main():
    for t in TICKERS:
        run_for_ticker(t)

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
