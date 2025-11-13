# trading_env.py (simplified for exploration)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLS = [
    "Open", "High", "Low", "Volume",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50",
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "Close_norm",
    "RET_1D", "VOL_20", "VOL_SHOCK", "TREND_20_50",
]


class TradingEnv(gym.Env):
    """
    Trading environment (simplified):
      - Obs: dict(market_data, portfolio_status)
      - Actions: Discrete(3) -> 0=Flat, 1=Half, 2=Full
      - Reward: log-return of portfolio (optionally + DSR later)
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        df: pd.DataFrame,
        lookback_window: int = 10,
        initial_capital: float = 100000.0,
        transaction_cost_pct: float = 0.001,
        feature_cols=None,
        return_dict_obs: bool = True,
    ):
        super().__init__()

        self.df = df
        self.lookback_window = int(lookback_window)
        self.initial_capital = float(initial_capital)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.return_dict_obs = bool(return_dict_obs)

        if "Close_raw" not in self.df.columns:
            raise ValueError("DataFrame must contain 'Close_raw' column")

        wanted = DEFAULT_FEATURE_COLS if feature_cols is None else feature_cols
        self.feature_cols = [c for c in wanted if c in self.df.columns]
        if len(self.feature_cols) == 0:
            raise ValueError("No valid feature columns in df")

        self.n_steps = len(self.df)
        self.n_features = len(self.feature_cols)

        # --- State ---
        self.current_step = 0
        self.cash = 0.0
        self.shares_held = 0.0
        self.portfolio_value = 0.0
        self.history = []

        # --- Observation space ---
        if self.return_dict_obs:
            self.observation_space = spaces.Dict({
                "market_data": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self.lookback_window, self.n_features),
                    dtype=np.float32
                ),
                "portfolio_status": spaces.Box(
                    low=0.0, high=np.inf, shape=(3,), dtype=np.float32
                ),
            })
        else:
            flat_dim = self.lookback_window * self.n_features + 3
            self.observation_space = spaces.Box(
                low=0.0, high=np.inf, shape=(flat_dim,), dtype=np.float32
            )

        # --- Action space ---
        self.action_space = spaces.Discrete(3)  # 0=Flat, 1=Half, 2=Full
        self._action_to_string = {0: "Flat", 1: "Half", 2: "Full"}

    # ===== Helpers =====
    def _build_obs(self):
        start_idx = self.current_step - self.lookback_window + 1
        end_idx = self.current_step + 1

        market = self.df.iloc[start_idx:end_idx][self.feature_cols].values.astype(np.float32)
        status = np.array([
            self.cash / self.initial_capital,
            self.shares_held / 1e6,
            self.portfolio_value / self.initial_capital,
        ], dtype=np.float32)
        status = np.maximum(status, 0.0)

        if self.return_dict_obs:
            return {"market_data": market, "portfolio_status": status}
        else:
            return np.concatenate([market.flatten(), status]).astype(np.float32)

    # ===== Gym API =====
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window - 1
        self.cash = float(self.initial_capital)
        self.shares_held = 0.0
        self.portfolio_value = float(self.initial_capital)
        self.history = []

        idx = self.df.index[self.current_step]
        price = float(self.df.loc[idx, "Close_raw"])
        self.history.append({
            "date": idx,
            "close": price,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "portfolio_value": self.portfolio_value,
            "action": 0,
            "reward": 0.0,
        })

        return self._build_obs(), {}

    def step(self, action):
        prev_val = float(self.portfolio_value)
        price = float(self.df.loc[self.df.index[self.current_step], "Close_raw"])

        targets = {0: 0.0, 1: 0.5, 2: 1.0}
        w_tgt = targets[int(action)]

        # Rebalance instantly (no cooldown/deadband/slippage)
        desired_val = prev_val * w_tgt
        current_val = self.shares_held * price
        diff = desired_val - current_val

        if diff > 0:  # buy
            gross = min(diff, self.cash)
            shares = gross / price
            fee = gross * self.transaction_cost_pct
            self.shares_held += shares
            self.cash -= (gross + fee)
        elif diff < 0:  # sell
            gross = min(-diff, current_val)
            shares = gross / price
            proceeds = shares * price
            fee = proceeds * self.transaction_cost_pct
            self.shares_held -= shares
            self.cash += (proceeds - fee)

        # Update portfolio value
        self.portfolio_value = self.cash + self.shares_held * price

        # --- Reward = log return ---
        reward = np.log((self.portfolio_value + 1e-9) / (prev_val + 1e-9))

        # Advance step
        self.current_step += 1
        terminated = bool(self.current_step >= self.n_steps - 1)

        if terminated:
            # force liquidation
            self.cash += self.shares_held * price * (1 - self.transaction_cost_pct)
            self.shares_held = 0.0
            self.portfolio_value = self.cash

        obs = self._build_obs() if not terminated else self.observation_space.sample()
        info = {"portfolio_value": float(self.portfolio_value)}

        # Log
        idx = self.df.index[min(self.current_step, self.n_steps - 1)]
        self.history.append({
            "date": idx,
            "close": price,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "portfolio_value": self.portfolio_value,
            "action": int(action),
            "reward": reward,
        })

        return obs, float(reward), terminated, False, info

    def render(self):
        idx = self.df.index[self.current_step]
        print(f"Step {self.current_step} | Date {idx} | "
              f"Action {self._action_to_string[self.history[-1]['action']]} | "
              f"PV {self.portfolio_value:.2f}")

    def close(self):
        pass
