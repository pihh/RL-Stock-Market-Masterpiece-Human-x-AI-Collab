import gymnasium as gym
import numpy as np

import gymnasium as gym
import numpy as np
import pandas as pd
from datetime import timedelta

class PositionTradingEnv(gym.Env):
    def __init__(
        self,
        full_df: pd.DataFrame,
        ticker: str,
        n_timesteps: int = 60,
        lookback: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.full_df = full_df.copy()
        self.ticker = ticker
        self.n_timesteps = n_timesteps
        self.lookback = lookback
        self.random_state = np.random.RandomState(seed)
        self.action_space = gym.spaces.Discrete(2)  # 0 = Flat, 1 = Long
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        self.episode_df = None
        self.step_idx = 0
        self._prepare_ticker_df()
        self._resample_episode()

    def _prepare_ticker_df(self):
        self.df = self.full_df[self.full_df['symbol'] == self.ticker].copy()
        self.df = self.df.sort_values("date")
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.reset_index(drop=True)

    def _resample_episode(self):
        mondays = self.df[self.df["date"].dt.weekday == 0].copy()
        valid_starts = []

        for date in mondays["date"]:
            start_idx = self.df.index[self.df["date"] == date][0]
            end_idx = start_idx + self.n_timesteps - 1
            if end_idx >= len(self.df):
                continue

            ep_slice = self.df.iloc[start_idx:end_idx + 1]
            if (ep_slice["symbol"].nunique() == 1) and (ep_slice["date"].is_monotonic_increasing):
                valid_starts.append(start_idx)

        if not valid_starts:
            raise ValueError("No valid episodes found with the current constraints.")

        self.start_idx = self.random_state.choice(valid_starts)
        self.end_idx = self.start_idx + self.n_timesteps - 1
        self.lookback_idx = max(0, self.start_idx - self.lookback)
        self.episode_df = self.df.iloc[self.lookback_idx:self.end_idx + 1].reset_index(drop=True)

        # Set prices used for reward logic
        self.prices = self.episode_df["close"].values
        self._precompute_step_weights()

    def _precompute_step_weights(self):
        raw_weights = [abs(self.prices[i + 1] - self.prices[i]) for i in range(len(self.prices) - 1)]
        total = sum(raw_weights)
        self.step_weights = [w / total if total > 0 else 1 / (len(raw_weights)) for w in raw_weights]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.random_state.seed(seed)
        self._resample_episode()
        self.step_idx = self.lookback
        self.position = 0
        self.total_reward = 0.0
        self.rewards = []
        self.actions = []
        self.values = []
        obs = np.array([self.prices[self.step_idx]], dtype=np.float32)
        return obs, {}

    def step(self, action):
        curr_idx = self.step_idx
        next_idx = min(curr_idx + 1, len(self.prices) - 1)
        curr_price = self.prices[curr_idx]
        next_price = self.prices[next_idx]
        price_diff = next_price - curr_price

        self.position = action
        agent_reward = price_diff if self.position == 1 else -price_diff
        oracle_reward = abs(price_diff)
        anti_reward = -oracle_reward

        if oracle_reward == anti_reward:
            step_score = 0.5
        else:
            step_score = (agent_reward - anti_reward) / (oracle_reward - anti_reward)

        step_score = float(np.clip(step_score, 0, 1))
        weight = self.step_weights[curr_idx - self.lookback] if curr_idx - self.lookback < len(self.step_weights) else 0
        scaled_reward = step_score * weight * 100

        self.total_reward += scaled_reward
        self.rewards.append(self.total_reward)
        self.actions.append(self.position)
        self.values.append(curr_price)

        self.step_idx += 1
        terminated = self.step_idx >= self.lookback + self.n_timesteps - 1
        truncated = False
        obs = np.array([self.prices[min(self.step_idx, len(self.prices) - 1)]], dtype=np.float32)

        return obs, scaled_reward, terminated, truncated, {}





def score_episode(agent_ret, oracle_ret, anti_ret):
    if oracle_ret == anti_ret:
        return 50
    return float(np.clip(100 * (agent_ret - anti_ret) / (oracle_ret - anti_ret), 0, 100))
