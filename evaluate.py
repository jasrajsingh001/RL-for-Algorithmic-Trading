# evaluate.py

import os
import pandas as pd
import numpy as np

from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation

from trading_env import TradingEnv


def main():
    # --- 1) Config ---
    TICKER = "AAPL"
    DATA_PATH = os.path.join("processed", f"{TICKER}_all_splits.csv")

    # <<< non-tuned model + default results file >>>
    MODEL_PATH = os.path.join("models", "ppo_trading_agent.zip")
    RESULTS_SAVE_PATH = os.path.join("results", "backtest_results.csv")

    INITIAL_CAPITAL = 100000.0
    TRANSACTION_COST_PCT = 0.001
    LOOKBACK_WINDOW = 20
    SHARPE_REWARD_ETA = 0.1

    # --- 2) Load model ---
    print(f"Step 2: Loading PPO model from {MODEL_PATH}...")
    try:
        model = PPO.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    print("Model loaded successfully.")

    # --- 3) Load data (DD-MM-YYYY -> parse with dayfirst) ---
    print("Step 3: Loading and preparing the test data...")
    try:
        df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True, dayfirst=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # Chronological split
    train_size = int(len(df) * 0.70)
    val_size = int(len(df) * 0.15)
    test_df = df.iloc[train_size + val_size :]

    print(f"Data loaded. Test period: {test_df.index.min()} to {test_df.index.max()}")
    print(f"Number of days in test set: {len(test_df)}")

    # --- 4) Build env ---
    print("Step 4: Instantiating the test environment...")
    base_env = TradingEnv(
        df=test_df,
        lookback_window=LOOKBACK_WINDOW,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        sharpe_reward_eta=SHARPE_REWARD_ETA,
    )

    # Try dict-obs; if model expects flattened obs, auto-wrap.
    env = FlattenObservation(base_env)
    obs, _ = env.reset()
    try:
        _ = model.predict(obs, deterministic=True)
    except Exception as e:
        print(f"Predict failed with dict obs ({type(e).__name__}: {e}). "
              f"Retrying with FlattenObservation...")
        env = FlattenObservation(base_env)
        obs, _ = env.reset()
        _ = model.predict(obs, deterministic=True)
        print("Predict succeeded with flattened observation.")

    # --- 5) Backtest loop ---
    print("Step 5: Running the evaluation loop (backtest)...")
    max_steps = len(test_df)
    action_counts = {0: 0, 1: 0, 2: 0}

    # make sure action is an int for our Discrete(3) env
    def to_int(a):
        return int(np.asarray(a).item())

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = to_int(action)
        obs, reward, done, truncated, info = env.step(action)
        if action in action_counts:
            action_counts[action] += 1
        if done or truncated:
            print("Evaluation finished (episode end).")
            break
    else:
        print("Evaluation finished (reached max test steps).")

    # --- Force liquidation at end so PnL is realized ---
    base = getattr(env, "env", env)  # if flattened, unwrap once
    if hasattr(base, "shares_held") and base.shares_held > 0:
        # simulate a sell-all at the last available price
        last_idx = base.current_step if base.current_step < base.n_steps else base.n_steps - 1
        last_price = float(base.df.loc[base.df.index[last_idx], "Close_raw"])
        proceeds = base.shares_held * last_price * (1 - base.transaction_cost_pct)
        base.cash += proceeds
        base.shares_held = 0.0
        base.portfolio_value = base.cash

        # log the synthetic liquidation step
        base.history.append({
            "date": base.df.index[last_idx],
            "close": last_price,
            "cash": float(base.cash),
            "shares_held": float(base.shares_held),
            "portfolio_value": float(base.portfolio_value),
            "action": 1,           # mark as Sell
            "reward": 0.0,
        })

    # --- 6) Save results ---
    print("Step 6: Storing and analyzing results...")
    os.makedirs("results", exist_ok=True)

    # history is on the base env
    history_env = base
    if not hasattr(history_env, "history") or len(history_env.history) == 0:
        print("Warning: no history recorded; nothing to save.")
        return

    backtest_results_df = pd.DataFrame(history_env.history).set_index("date")

    print("\n--- Backtest Results (First 5 Days) ---")
    print(backtest_results_df.head())
    print("\n--- Backtest Results (Last 5 Days) ---")
    print(backtest_results_df.tail())

    backtest_results_df.to_csv(RESULTS_SAVE_PATH)
    print(f"\nBacktest results saved to: {RESULTS_SAVE_PATH}")

    # --- 7) Quick KPIs ---
    pv = backtest_results_df["portfolio_value"]
    if len(pv) >= 2:
        cum_ret = pv.iloc[-1] / pv.iloc[0] - 1.0
        dd = (pv / pv.cummax() - 1.0).min()
        print("\n--- Quick KPIs ---")
        print(f"Cumulative Return: {cum_ret:.2%}")
        print(f"Max Drawdown:      {dd:.2%}")

    print("\n--- Final State ---")
    print(f"Cash: ${history_env.cash:,.2f} | Shares: {history_env.shares_held:.4f} | "
          f"Portfolio: ${history_env.portfolio_value:,.2f}")

    print("\n--- Action Counts ---")
    print({0: "Buy", 1: "Sell", 2: "Hold"})
    print(action_counts)


    print("\nEvaluation step complete.")


if __name__ == "__main__":
    main()
