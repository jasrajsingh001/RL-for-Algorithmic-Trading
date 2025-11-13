# evaluate_tuned.py

import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation

from trading_env import TradingEnv


def _unwrap_to_base(env):
    """Walk through wrappers until we find the env that carries .history."""
    base = env
    depth_guard = 0
    while hasattr(base, "env") and not hasattr(base, "history") and depth_guard < 10:
        base = base.env
        depth_guard += 1
    return base


def _force_liquidation(base_env):
    """Sell any remaining position at the last known price and log it."""
    if hasattr(base_env, "shares_held") and base_env.shares_held > 0:
        last_idx = min(base_env.current_step, base_env.n_steps - 1)
        last_price = float(base_env.df.loc[base_env.df.index[last_idx], "Close_raw"])
        proceeds = base_env.shares_held * last_price * (1 - base_env.transaction_cost_pct)
        base_env.cash += proceeds
        base_env.shares_held = 0.0
        base_env.portfolio_value = float(base_env.cash)
        base_env.history.append({
            "date": base_env.df.index[last_idx],
            "close": last_price,
            "cash": float(base_env.cash),
            "shares_held": float(base_env.shares_held),
            "portfolio_value": float(base_env.portfolio_value),
            "action": 1,  # mark as Sell
            "reward": 0.0,
        })


def main():
    # --- 1) Config ---
    TICKER = "AMZN"
    DATA_PATH = os.path.join("processed", f"{TICKER}_all_splits.csv")

    MODEL_PATH = os.path.join("models", "ppo_trading_agent_tuned.zip")  # TUNED
    RESULTS_SAVE_PATH = os.path.join("results", "backtest_results_tuned.csv")

    INITIAL_CAPITAL = 100000.0
    TRANSACTION_COST_PCT = 0.001
    LOOKBACK_WINDOW = 20
    SHARPE_REWARD_ETA = 0.1

    # --- 2) Load model ---
    print(f"Step 2: Loading PPO model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    print("Model loaded successfully.")

    # --- 3) Load data ---
    print("Step 3: Loading and preparing the test data...")
    df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True, dayfirst=True)

    # Chronological split
    train_size = int(len(df) * 0.70)
    val_size = int(len(df) * 0.15)
    test_df = df.iloc[train_size + val_size:]

    print(f"Data loaded. Test period: {test_df.index.min()} to {test_df.index.max()}")
    print(f"Number of days in test set: {len(test_df)}")

    # --- 4) Build env (Dict observation expected for tuned model) ---
    base_env = TradingEnv(
        df=test_df,
        lookback_window=LOOKBACK_WINDOW,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        sharpe_reward_eta=SHARPE_REWARD_ETA,
        # If you trained with realism knobs, add them here:
        # trade_deadband=0.05, trade_cooldown=3, slippage_bps=1.0,
    )

    # Try Dict observation first (preferred for tuned MultiInputPolicy).
    env = base_env
    obs, _ = env.reset()
    try:
        # Quick probe to ensure compatibility
        _ = model.predict(obs, deterministic=True)
        print("Detected model expects Dict observations (MultiInputPolicy).")
    except Exception as e:
        # Fallback: some tuned models may still have been saved expecting a flat Box
        print(f"Predict failed with Dict obs ({type(e).__name__}: {e}). "
              f"Retrying with FlattenObservation...")
        env = FlattenObservation(base_env)
        obs, _ = env.reset()
        _ = model.predict(obs, deterministic=True)
        print("Predict succeeded with flattened observation.")

    # --- 5) Backtest loop ---
    print("Step 5: Running the evaluation loop (backtest)...")
    max_steps = len(test_df)
    action_counts = {0: 0, 1: 0, 2: 0}
    ACTION_NAMES = {0: "Flat (0%)", 1: "Half (50%)", 2: "Full (100%)"}

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

    # --- 6) Force liquidation and save ---
    base = _unwrap_to_base(env)
    _force_liquidation(base)

    print("Step 6: Storing and analyzing results...")
    os.makedirs("results", exist_ok=True)

    if not hasattr(base, "history") or len(base.history) == 0:
        print("Warning: no history recorded; nothing to save.")
        return

    backtest_results_df = pd.DataFrame(base.history).set_index("date")

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

    # --- 8) Final state & action counts ---
    print("\n--- Final State ---")
    print(f"Cash: ${base.cash:,.2f} | Shares: {base.shares_held:.4f} | "
          f"Portfolio: ${base.portfolio_value:,.2f}")

    print("\n--- Action Counts ---")
    print(ACTION_NAMES)
    print(action_counts)

    print("\nEvaluation step complete.")


if __name__ == "__main__":
    main()
