# retrain_best_model.py

import os
import pandas as pd
from stable_baselines3 import PPO

# We must import our custom environment
from trading_env import TradingEnv

def main():
    """
    Main function to retrain the final model with the best hyperparameters
    found during the tuning phase.
    """
    # --- 1. Configuration & The Winning Formula ---
    TICKER = 'AAPL'
    INITIAL_CAPITAL = 100000.0
    TRANSACTION_COST_PCT = 0.001
    LOOKBACK_WINDOW = 20
    SHARPE_REWARD_ETA = 0.1
    TRAIN_TIMESTEPS = 1_000_000 # Use the full training duration

    # Define the path for the final, tuned model
    # We name it differently to distinguish it from the first model.
    TUNED_MODEL_PATH = os.path.join('models', 'ppo_trading_agent_tuned.zip')
    TENSORBOARD_LOG_PATH = "./tensorboard_logs_tuned/"

    # ==> The Best Hyperparameters Found by Optuna <==
    # These are the results from the `study.best_params` in tune.py
    # In a real-world scenario, you might load these from a file (e.g., JSON).
    # For this project, we'll hardcode them based on our hypothetical study results.
    best_params = {
        'learning_rate': 0.0001234, # A sample value from a hypothetical log-uniform search
        'n_steps': 1024,
        'gamma': 0.99,
        'clip_range': 0.2,
        'net_arch': 'medium' # The string we used in Optuna
    }
    
    # We need to map the net_arch string back to the dictionary structure SB3 expects
    net_arch_dict = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[128, 128], vf=[128, 128])]
    }
    # Create the final model parameters dictionary, substituting the net_arch
    model_params = best_params.copy()
    model_params['policy_kwargs'] = {"net_arch": net_arch_dict[best_params['net_arch']]}
    # We no longer need the string version of net_arch in the main dictionary
    del model_params['net_arch']

    # --- 2. Load and Prepare the Combined Dataset ---
    DATA_PATH = os.path.join('processed', f'{TICKER}_all_splits.csv')
    print("Loading and preparing the combined train+validation dataset...")
    try:
        df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # This is the key step: combine the original 70% train and 15% validation sets.
    # The final 15% test set remains completely separate and untouched.
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_val_df = df.iloc[:train_size + val_size]
    test_df = df.iloc[train_size + val_size:] # For reference, we are not using this here.
    
    print(f"Combined training data size: {len(train_val_df)} days")
    print(f"Held-out test data size: {len(test_df)} days")

    # --- 3. Instantiate Environment and Model ---
    print("\nInstantiating the environment with the combined dataset...")
    
    # Create the environment using our combined training and validation data
    env = TradingEnv(
        df=train_val_df,
        lookback_window=LOOKBACK_WINDOW,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        sharpe_reward_eta=SHARPE_REWARD_ETA
    )

    print("Instantiating the PPO model with the best hyperparameters...")
    # Instantiate the PPO model, unpacking the `model_params` dictionary.
    # The ** operator elegantly passes all key-value pairs as arguments.
    model = PPO(
        "MultiInputPolicy",
        env,
        tensorboard_log=TENSORBOARD_LOG_PATH,
        verbose=1,
        **model_params
    )

    # --- 4. Train and Save the Champion Model ---
    print(f"\nStarting final training for {TRAIN_TIMESTEPS} timesteps...")
    print("Monitor progress with: tensorboard --logdir ./tensorboard_logs_tuned/")
    
    model.learn(total_timesteps=TRAIN_TIMESTEPS)

    print("\nTraining complete. Saving the final tuned model...")
    model.save(TUNED_MODEL_PATH)
    print(f"Tuned model saved to: {TUNED_MODEL_PATH}")
    print("\nRetraining process finished successfully.")

if __name__ == '__main__':
    main()
