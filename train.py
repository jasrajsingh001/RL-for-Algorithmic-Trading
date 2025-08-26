# train.py

import os
import pandas as pd

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from trading_env import TradingEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor  # optional but helpful

def main():
    # --- 1. Config ---
    TICKER = 'AAPL'
    DATA_PATH = os.path.join('processed', f'{TICKER}_all_splits.csv')

    TENSORBOARD_LOG_DIR = os.path.join('logs', 'tensorboard_logs')
    MODEL_SAVE_DIR = os.path.join('models')

    INITIAL_CAPITAL = 100000.0
    TRANSACTION_COST_PCT = 0.001
    LOOKBACK_WINDOW = 20
    SHARPE_REWARD_ETA = 0.1

    TRAINING_TIMESTEPS = 100_000
    SEED = 42  # <- optional: reproducibility

    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # --- 2. Load data (DD-MM-YYYY -> dayfirst=True) ---
    print("Step 2: Loading and preparing data...")
    try:
        df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True, dayfirst=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure you have run the data processing scripts from Step 2.")
        return

    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size]

    print(f"Data loaded. Training period: {train_df.index.min()} to {train_df.index.max()}")
    if "Close_raw" not in train_df.columns:
        raise ValueError("Expected 'Close_raw' in data for pricing. Make sure process_data added it.")

    # --- 3. Build env ---
    print("Step 3: Instantiating the trading environment...")
    base_env = TradingEnv(
        df=train_df,
        lookback_window=LOOKBACK_WINDOW,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        sharpe_reward_eta=SHARPE_REWARD_ETA,
    )

    # 3a. Env check on the base Dict-obs env (good sanity check)
    print("Step 3a: Checking the environment's compatibility...")
    try:
        check_env(base_env)
        print("Environment check passed! Your custom environment is compliant.")
    except Exception as e:
        print(f"Environment check failed: {e}")
        return

    # Flatten Dict -> Box for PPO, and add Monitor for logging
    env = FlattenObservation(base_env)
    env = Monitor(env)

    # --- 4. PPO model ---
    print("Step 4: Instantiating the PPO model with TensorBoard logging...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        seed=SEED,  # optional
        # You can tune these later if learning is sluggish:
        # learning_rate=3e-4, n_steps=2048, batch_size=256, ent_coef=0.01,
    )

    # --- 5. Train ---
    print(f"Step 5: Training the model for {TRAINING_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TRAINING_TIMESTEPS)

    # --- 6. Save ---
    model_save_path = os.path.join(MODEL_SAVE_DIR, 'ppo_trading_agent.zip')
    print(f"Step 6: Saving the trained model to: {model_save_path}")
    model.save(model_save_path)

    print("\nTraining complete and model saved!")

if __name__ == '__main__':
    main()
