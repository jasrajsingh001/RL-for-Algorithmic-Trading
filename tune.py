# tune.py

import os
import tempfile
import numpy as np
import pandas as pd
import optuna

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit, FlattenObservation

from trading_env import TradingEnv
from analyze import calculate_sharpe_ratio


# ----------------------------
# Global config for tuning
# ----------------------------
TICKER = "AMZN"
DATA_PATH = os.path.join("processed", f"{TICKER}_all_splits.csv")

INITIAL_CAPITAL = 100000.0
TRANSACTION_COST_PCT = 0.001
LOOKBACK_WINDOW = 20
SHARPE_REWARD_ETA = 0.1
RISK_FREE_RATE = 0.02
EPISODE_STEPS = 252          # ~1 trading year
TRAIN_TIMESTEPS = 150_000      # per trial
SEED = 42


def make_wrapped_env(df_slice: pd.DataFrame) -> Monitor:
    """
    Build an env wrapped exactly like train.py: TimeLimit -> FlattenObservation -> Monitor.
    """
    env = TradingEnv(
        df=df_slice,
        lookback_window=LOOKBACK_WINDOW,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        sharpe_reward_eta=SHARPE_REWARD_ETA,
    )
    env = TimeLimit(env, max_episode_steps=EPISODE_STEPS)
    env = FlattenObservation(env)
    env = Monitor(env)
    return env


# --- 1. Objective function for Optuna ---
def objective(trial: optuna.Trial) -> float:
    print(f"\n--- Starting Trial {trial.number} ---")

    # ----- Suggest hyperparameters -----
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps       = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size    = trial.suggest_categorical("batch_size", [256, 512, 1024])
    gamma         = trial.suggest_float("gamma", 0.95, 0.9999)
    gae_lambda    = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range    = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    ent_coef      = trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True)
    net_arch_opt  = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    net_arch = {"small": [128, 128], "medium": [256, 256], "large": [256, 256, 128]}[net_arch_opt]

    print(
        f"Hyperparams: lr={learning_rate:.6f}, n_steps={n_steps}, batch={batch_size}, "
        f"gamma={gamma:.4f}, gae={gae_lambda:.3f}, clip={clip_range}, ent={ent_coef:.5f}, arch={net_arch_opt}"
    )

    # ----- Build train/val envs -----
    train_env = make_wrapped_env(train_df)
    val_env   = make_wrapped_env(val_df)

    # ----- Build model (match your train.py style) -----
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=net_arch),
        verbose=0,
        seed=SEED,
        tensorboard_log=None,
    )

    # ----- Train -----
    try:
        model.learn(total_timesteps=TRAIN_TIMESTEPS, progress_bar=False)
    except Exception as e:
        print(f"Training error in trial {trial.number}: {e}")
        return -np.inf

    # ----- Validation rollout (deterministic) -----
    obs, _ = val_env.reset()
    for _ in range(len(val_df)):  # cap to validation set length
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = val_env.step(action)
        if done or truncated:
            break

    # ----- Score by Sharpe on validation equity curve -----
    base_env = val_env.unwrapped  # unwrap FlattenObservation to get .history
    hist = getattr(base_env, "history", None)
    if not hist:
        print("Validation history is empty. Returning worst score.")
        return -np.inf

    val_hist_df = pd.DataFrame(hist)
    if val_hist_df.empty or "portfolio_value" not in val_hist_df.columns:
        print("Validation history invalid. Returning worst score.")
        return -np.inf

    val_hist_df.set_index("date", inplace=True)
    sharpe = calculate_sharpe_ratio(val_hist_df["portfolio_value"], RISK_FREE_RATE)

    if not np.isfinite(sharpe):
        return -np.inf

    print(f"Trial {trial.number} Sharpe: {sharpe:.4f}")
    return float(sharpe)


# --- 2. Main execution ---
if __name__ == "__main__":
    print("Loading and splitting data for tuning...")
    try:
        df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True, dayfirst=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        raise SystemExit(1)

    # Chronological split
    train_size = int(len(df) * 0.70)
    val_size   = int(len(df) * 0.15)

    train_df = df.iloc[:train_size].copy()
    val_df   = df.iloc[train_size : train_size + val_size].copy()

    print(f"Data ready. Train: {len(train_df)} days, Validation: {len(val_df)} days.")

    # --- Create and run the Optuna study ---
    print("\nStarting Optuna hyperparameter optimization study...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    print("\nOptimization study finished.")
    best_trial = study.best_trial
    print(f"Best Sharpe: {best_trial.value:.4f}")
    print("Best Params:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")
