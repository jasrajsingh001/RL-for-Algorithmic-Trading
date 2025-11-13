# train.py

import os
import pandas as pd

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation

from trading_env import TradingEnv


def main():
    # --- 1. Config ---
    TICKER = "AMZN"
    DATA_PATH = os.path.join("processed", f"{TICKER}_all_splits.csv")

    LOG_DIR = os.path.join("logs", "tensorboard_logs")
    MODEL_DIR = os.path.join("models")

    INITIAL_CAPITAL = 100000.0
    TRANSACTION_COST_PCT = 0.001
    LOOKBACK_WINDOW = 20

    TRAINING_TIMESTEPS = 25_000  # bump up as needed
    SEED = 42

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- 2. Load data (DD-MM-YYYY -> dayfirst=True) ---
    print("Step 2: Loading and preparing data...")
    try:
        df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True, dayfirst=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    train_size = int(len(df) * 0.70)
    train_df = df.iloc[:train_size]
    print(f"Data loaded. Training period: {train_df.index.min()} to {train_df.index.max()}")

    if "Close_raw" not in train_df.columns:
        raise ValueError("Expected 'Close_raw' in data for pricing. Make sure your processing step added it.")

    # --- 3. Build env (Dict obs -> we will flatten for MlpPolicy) ---
    print("Step 3: Instantiating the trading environment...")
    base_env = TradingEnv(
        df=train_df,
        lookback_window=LOOKBACK_WINDOW,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        # return_dict_obs defaults to True (fine here because we will flatten below)
    )

    # Optional sanity check on the dict-obs env
    print("Step 3a: Checking the environment's compatibility...")
    try:
        check_env(base_env)
        print("Environment check passed! Your custom environment is compliant.")
    except Exception as e:
        print(f"Environment check failed: {e}")
        return

    # PPO/MlpPolicy expects a Box; flatten the Dict obs and wrap with Monitor
    env = FlattenObservation(base_env)
    env = Monitor(env)

    # --- 4. PPO model ---
    print("Step 4: Instantiating the PPO model with TensorBoard logging...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=LOG_DIR,
        verbose=1,
        seed=SEED,
        # exploration-friendly defaults
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,           # encourage exploration so it doesn't collapse to a single action
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    # --- 5. Train ---
    print(f"Step 5: Training the model for {TRAINING_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TRAINING_TIMESTEPS)

    # --- 6. Save ---
    model_path = os.path.join(MODEL_DIR, "ppo_trading_agent.zip")
    print(f"Step 6: Saving the trained model to: {model_path}")
    model.save(model_path)

    print("\nTraining complete and model saved!")


if __name__ == "__main__":
    main()
