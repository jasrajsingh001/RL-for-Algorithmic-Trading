# trading_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# Agent-facing features (normalized/bounded in [0,1])
DEFAULT_FEATURE_COLS = [
    "Open", "High", "Low", "Volume",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50",
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "Close_norm",
]


class TradingEnv(gym.Env):
    """
    Custom trading environment with:
      - Observations: dict(
            market_data: (lookback_window, n_features) normalized to [0,1],
            portfolio_status: [cash/I0, shares_scaled, pv/I0]
        )
      - Actions: Discrete(3) -> target weights in asset: 0% / 50% / 100%
      - Pricing & accounting: uses Close_raw (UNSCALED real prices)
      - Reward: guarded Differential Sharpe Ratio (DSR)
    """
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        df: pd.DataFrame,
        lookback_window: int = 10,
        initial_capital: float = 100000.0,
        transaction_cost_pct: float = 0.001,
        sharpe_reward_eta: float = 0.1,
        feature_cols=None,
    ):
        super().__init__()

        # --- 1) Data & params ---
        self.df = df
        self.lookback_window = int(lookback_window)
        self.initial_capital = float(initial_capital)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.sharpe_reward_eta = float(sharpe_reward_eta)

        if "Close_raw" not in self.df.columns:
            raise ValueError("DataFrame must contain 'Close_raw' (unscaled) for pricing.")

        wanted = DEFAULT_FEATURE_COLS if feature_cols is None else feature_cols
        self.feature_cols = [c for c in wanted if c in self.df.columns]
        if len(self.feature_cols) == 0:
            raise ValueError("No valid feature columns found in df for observations.")

        # --- 2) Dimensions ---
        self.n_steps = len(self.df)
        self.n_features = len(self.feature_cols)

        # --- 3) State ---
        self.current_step = 0
        self.cash = 0.0
        self.shares_held = 0.0
        self.portfolio_value = 0.0
        self.A = 0.0  # EWMA of returns for DSR
        self.B = 0.0  # EWMA of squared returns for DSR

        self._last_action = 0
        self._last_reward = 0.0
        self.history = []

        # --- 4) Observation space ---
        # market_data: normalized features in [0,1]
        self.observation_space = spaces.Dict({
            "market_data": spaces.Box(
                low=0.0, high=1.0, shape=(self.lookback_window, self.n_features), dtype=np.float32
            ),
            # portfolio_status: ratios >= 0
            "portfolio_status": spaces.Box(
                low=0.0, high=np.inf, shape=(3,), dtype=np.float32
            ),
        })

        # --- 5) Action space (target weights) ---
        # 0 -> 0% (flat), 1 -> 50% (half), 2 -> 100% (fully invested)
        self.action_space = spaces.Discrete(3)
        self._action_to_string = {0: "Flat (0%)", 1: "Half (50%)", 2: "Full (100%)"}

    # ========= Helpers =========
    def _get_observation(self):
        start_idx = self.current_step - self.lookback_window + 1
        end_idx = self.current_step + 1

        market_features = (
            self.df.iloc[start_idx:end_idx][self.feature_cols].values.astype(np.float32)
        )

        # Safe portfolio_status (clip tiny negatives due to fp error)
        portfolio_status = np.array([
            self.cash / self.initial_capital,
            self.shares_held / 1e6,
            self.portfolio_value / self.initial_capital,
        ], dtype=np.float32)
        portfolio_status = np.maximum(portfolio_status, 0.0)

        return {
            "market_data": market_features,
            "portfolio_status": portfolio_status,
        }

    # ========= Gym API =========
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start at first index that has a full lookback window
        self.current_step = self.lookback_window - 1

        self.cash = float(self.initial_capital)
        self.shares_held = 0.0
        self.portfolio_value = float(self.initial_capital)
        self.A = 0.0
        self.B = 0.0

        self._last_action = 0
        self._last_reward = 0.0
        self.history = []

        idx = self.df.index[self.current_step]
        price = float(self.df.loc[idx, "Close_raw"])
        self.history.append({
            "date": idx,
            "close": price,
            "cash": float(self.cash),
            "shares_held": float(self.shares_held),
            "portfolio_value": float(self.portfolio_value),
            "action": 0,  # Flat
            "reward": 0.0,
        })

        return self._get_observation(), {}

    def step(self, action):
        # --- Snapshot for reward ---
        previous_portfolio_value = float(self.portfolio_value)

        # --- Current price & state ---
        idx = self.df.index[self.current_step]
        price = float(self.df.loc[idx, "Close_raw"])

        # Fallback pv in case of first step numerical edge
        prev_value = (
            previous_portfolio_value
            if previous_portfolio_value > 0
            else (self.cash + self.shares_held * price)
        )

        # Current weight (0..1) in asset; guard division
        w_now = 0.0 if prev_value <= 1e-9 else (self.shares_held * price) / prev_value

        # Target weights for Discrete(3)
        targets = {0: 0.0, 1: 0.5, 2: 1.0}
        w_tgt = float(targets[int(action)])

        # Notional trade to move to target (one-shot rebalance)
        trade_val = (w_tgt - w_now) * prev_value

        # --- Execute trade with costs ---
        if trade_val > 0:  # Buy
            gross = min(trade_val, self.cash)
            shares = gross / price if price > 0 else 0.0
            fee = shares * price * self.transaction_cost_pct
            # Adjust if fees would exceed cash
            if gross + fee > self.cash and price > 0:
                shares = self.cash / (price * (1.0 + self.transaction_cost_pct))
                gross = shares * price
                fee = gross * self.transaction_cost_pct
            self.shares_held += shares
            self.cash -= (gross + fee)

        elif trade_val < 0:  # Sell
            gross = min(abs(trade_val), self.shares_held * price)
            shares = gross / price if price > 0 else 0.0
            proceeds = shares * price
            fee = proceeds * self.transaction_cost_pct
            self.shares_held -= shares
            self.cash += (proceeds - fee)

        # Mark-to-market
        self.portfolio_value = float(self.cash + self.shares_held * price)
        current_price = price

        # --- DSR reward (guarded) ---
        daily_return = float((self.portfolio_value / (previous_portfolio_value + 1e-9)) - 1.0)
        prev_A, prev_B = float(self.A), float(self.B)
        eta = float(self.sharpe_reward_eta)

        self.A = (1.0 - eta) * self.A + eta * daily_return
        self.B = (1.0 - eta) * self.B + eta * (daily_return ** 2)

        var_proxy = float(prev_B - prev_A * prev_A)  # can be tiny negative from fp
        if not np.isfinite(var_proxy) or var_proxy <= 1e-12:
            reward = 0.0
        else:
            denom = var_proxy ** 1.5
            numer = prev_B * daily_return - prev_A * (daily_return ** 2)
            reward = float(numer / (denom + 1e-12))

        # >>> ADD HERE (after DSR reward, before storing it) <<<
        trade_penalty = 0.001 * abs(w_tgt - w_now)       # discourage big reallocations
        hold_bonus    = 0.0002 if action == 2 else 0     # slight bias to hold
        reward = reward - trade_penalty + hold_bonus
        # <<< END ADD >>>

        self._last_action = int(action)
        self._last_reward = float(reward)

        # --- Advance time ---
        self.current_step += 1
        terminated = bool(self.current_step >= self.n_steps - 1)

        # Liquidate at the end so PnL is realized
        if terminated and self.shares_held > 0.0:
            proceeds = self.shares_held * price * (1.0 - self.transaction_cost_pct)
            self.cash += proceeds
            self.shares_held = 0.0
            self.portfolio_value = float(self.cash)

        # Next observation
        if terminated:
            observation = {
                "market_data": np.zeros((self.lookback_window, self.n_features), dtype=np.float32),
                "portfolio_status": np.zeros(3, dtype=np.float32),
            }
        else:
            observation = self._get_observation()

        info = {"portfolio_value": float(self.portfolio_value)}

        # Log
        log_step = min(self.current_step, self.n_steps - 1)
        log_date = self.df.index[log_step]
        self.history.append({
            "date": log_date,
            "close": float(current_price),
            "cash": float(self.cash),
            "shares_held": float(self.shares_held),
            "portfolio_value": float(self.portfolio_value),
            "action": int(action),
            "reward": float(reward),
        })

        return observation, float(reward), terminated, False, info

    def render(self):
        idx = self.df.index[self.current_step]
        print(
            f"--- Step: {self.current_step} | Date: {idx.strftime('%Y-%m-%d')} ---\n"
            f"  Action Taken:      {self._action_to_string.get(self._last_action, self._last_action)}\n"
            f"  Reward Received:   {self._last_reward:.6f}\n"
            f"-----------------------------------------\n"
            f"  Portfolio Value:   ${self.portfolio_value:,.2f}\n"
            f"  Cash Balance:      ${self.cash:,.2f}\n"
            f"  Shares Held:       {self.shares_held}\n"
            f"-----------------------------------------\n"
        )

    def close(self):
        pass
