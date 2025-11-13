# debug_model_diff.py
import os, hashlib, numpy as np, torch
from stable_baselines3 import PPO
from trading_env import TradingEnv

MODELS = [
    ("original", "models/ppo_trading_agent.zip"),
    ("tuned",    "models/ppo_trading_agent_tuned.zip"),
]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def flat_params(model):
    arrs = []
    for p in model.policy.parameters():
        arrs.append(p.detach().cpu().numpy().ravel())
    return np.concatenate(arrs)

print("== File hashes ==")
for name, p in MODELS:
    print(f"{name:9s}: {p} | exists={os.path.exists(p)}")
    if os.path.exists(p):
        print(f"  sha256: {sha256(p)}")

# Load models
m1 = PPO.load(MODELS[0][1])
m2 = PPO.load(MODELS[1][1])

print("\n== Policy classes ==")
print("original:", m1.policy.__class__.__name__)
print("tuned   :", m2.policy.__class__.__name__)

# Compare parameter vectors
v1 = flat_params(m1); v2 = flat_params(m2)
print("\n== Param vector stats ==")
print("len(v1), len(v2):", len(v1), len(v2))
print("allclose:", np.allclose(v1, v2))
print("max |diff|:", float(np.max(np.abs(v1 - v2))))

# Compare actions on the same observation
import pandas as pd
TICKER = "AMZN"
DATA_PATH = os.path.join("processed", f"{TICKER}_all_splits.csv")
df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True, dayfirst=True)
train, val = int(len(df)*0.7), int(len(df)*0.15)
test_df = df.iloc[train+val:]

# build dict-obs env (matches your current TradingEnv)
env = TradingEnv(df=test_df, lookback_window=20, initial_capital=100000.0,
                 transaction_cost_pct=0.001, sharpe_reward_eta=0.1)
obs, _ = env.reset()

def to_int(a): return int(np.asarray(a).item())

acts1 = []
acts2 = []
for _ in range(50):  # first 50 steps
    a1, _ = m1.predict(obs, deterministic=True); a1 = to_int(a1)
    a2, _ = m2.predict(obs, deterministic=True); a2 = to_int(a2)
    acts1.append(a1); acts2.append(a2)
    obs, _, done, truncated, _ = env.step(a1)  # step with original action to move obs
    if done or truncated: break

print("\n== First 50 actions ==")
print("original:", acts1)
print("tuned   :", acts2)
