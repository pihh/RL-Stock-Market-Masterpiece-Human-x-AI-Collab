import numpy as np
import pandas as pd
import gymnasium as gym
from datetime import timedelta



class PositionTradingEnv(gym.Env):
    
    __version__ = 0
    
    def __init__(
        self,
        full_df: pd.DataFrame,
        ticker: str,
        market_features:list = ['close'],
        n_timesteps: int = 60,
        lookback: int = 0,
        seed: int = 42,
        start_idx = None
    ):
        super().__init__()
     
        self.full_df = full_df.copy()
        self.ticker = ticker
        self.n_timesteps = n_timesteps
        self.lookback = lookback
        self.market_features = market_features
        self.random_state = np.random.RandomState(seed)
        self.action_space = gym.spaces.Discrete(2)  # 0 = Flat, 1 = Long
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(len(market_features),), dtype=np.float32)
        self.episode_df = None
        self.step_idx = 0
        self.fixed_start_idx = start_idx  # store the user-specified start index

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

       

        if self.fixed_start_idx is not None:
            # Use the specified start index if it's valid
            chosen_idx = self.fixed_start_idx
            # Optionally, verify that chosen_idx is a valid start (e.g., in valid_starts)
            
        else:
            if not valid_starts :
                raise ValueError("No valid episodes found with the current constraints.")
            # Randomly select a start index as before
            chosen_idx = self.random_state.choice(valid_starts)
            
        self.start_idx = chosen_idx
        self.end_idx = self.start_idx + self.n_timesteps - 1
        if self.end_idx >= len(self.df):
            raise ValueError("Episode end index out of range (start_idx too close to end).")
        self.lookback_idx = max(0, self.start_idx - self.lookback)
        self.episode_df = self.df.iloc[self.lookback_idx : self.end_idx + 1].reset_index(drop=True)

        # Set prices used for reward logic
        self.prices = self.episode_df["close"].values
        self.episode_values = self.episode_df[self.market_features].values 
        self._precompute_step_weights()

    def _precompute_step_weights(self):
        raw_weights = [abs(self.prices[i + 1] - self.prices[i]) for i in range(len(self.prices) - 1)]
        total = sum(raw_weights)
        self.step_weights = [w / total if total > 0 else 1 / (len(raw_weights)) for w in raw_weights]

    def reset(self, *, seed=None, options=None):
        if self.fixed_start_idx is not None:
            
            # Maybe just reuse the current episode if already set, or call _resample_episode()
            # which in turn will use the fixed_start_idx (since we added that logic).
            self._resample_episode()
        else:
            self._resample_episode()
        #self._resample_episode()
        self.step_idx = self.lookback
        self.position = 0
        self.total_reward = 0.0
        self.rewards = []
        self.actions = []
        self.values = []
        obs = np.array([self.episode_values[self.step_idx]], dtype=np.float32)
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
        obs = np.array([self.episode_values[min(self.step_idx, len(self.prices) - 1)]], dtype=np.float32)

        return obs, scaled_reward, terminated, truncated, {}


class PositionTradingEnvV1(PositionTradingEnv):
    """
    V1 introduces some kind of memory to the agent and his    
    observation now contains:
    * last 3 actions as one hot encode -> 2 possible actions * 3 last steps
    * entry price ( the price he did buy ) 
    * holding time ( how long he is in the same position ) 
    * pnl 
    * price ratio
    * rolling returns
    
    observation is now shaped like this:
    * current position,
    * how long he is holding his decision (holding_time),
    * profit and loss - pnl,
    * price ratio - price_ratio,
    * rolling return - rolling_ret,
    * day one hot encode - since it's stock market focues we will use 5 days - day_one_hot ,
    * action one hot encode - 2 possible actions ( positions ) - grab and hold a stock or wait to grab * 3 last days - 6 slots
    * market features.
            
    Args:
        PositionTradingEnv (_type_): _description_
    """
    __version__ = 1
    def __init__(self ,*args, **kwargs):
        super().__init__(*args, **kwargs)
       
        self.last_actions = [0, 0, 0]  # history of last 3 actions
        self.entry_price = 0
        self.holding_time = 0
        #self.market_features = market_features
        obs_dim =len(self.market_features)+ 5 + 5 + 6 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.last_actions = [0, 0, 0]
        self.entry_price = self.prices[self.step_idx]
        self.holding_time = 0
        return self._get_observation(), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        """
        if action == 1:
            if self.position == 0:  # Enter long
                
                self.entry_price = self.prices[self.step_idx]
                self.holding_time = 0
            else:  # Exit
                self.entry_price = 0
                self.holding_time = 0
        elif self.position == 1:
            self.holding_time += 1

        """
        price_now = self.prices[self.step_idx]
        if self.position == 0:
            if action == 0:
                # Still flat – just waiting
                self.holding_time += 1
            elif action == 1:
                # Enter long
                self.entry_price = price_now
                self.holding_time = 0
        else:  # self.position == 1
            if action == 1:
                # Hold position
                self.holding_time += 1
            elif action == 0:
                # Exit position
                # (optional: track realized pnl here if not already in reward)
                self.entry_price = 0
                self.holding_time = 0
        self.last_actions = self.last_actions[1:] + [action]
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        price_now = self.prices[self.step_idx]
        step_values = self.episode_values[self.step_idx]
        
        # If in position, use true entry price; otherwise, use current price as neutral baseline
        entry_price = self.entry_price if self.position else price_now

        # Safe PNL and ratio computation
        if entry_price > 0:
            pnl = (price_now - entry_price) / entry_price
            price_ratio = price_now / entry_price
        else:
            pnl = 0.0
            price_ratio = 1.0

        # Rolling return
        window_start = max(0, self.step_idx - 5)
        if self.step_idx > window_start:
            price_slice = self.prices[window_start:self.step_idx + 1]
            rolling_ret = np.mean(np.diff(price_slice) / price_slice[:-1])
        else:
            rolling_ret = 0.0

        # One-hot encode day of week
        day = int(self.episode_df.iloc[self.step_idx]["day_of_week"])
        day_one_hot = np.zeros(5)
        day_one_hot[day] = 1

        # One-hot encode last 3 actions
        action_onehots = []
        for a in self.last_actions:
            onehot = np.zeros(2)
            onehot[a] = 1
            action_onehots.extend(onehot)

        obs = np.array([
            
            self.position,
            self.holding_time,
            pnl,
            price_ratio,
            rolling_ret,
            *day_one_hot,
            *action_onehots,
            *step_values
        ], dtype=np.float32)
        return obs





def score_episode(agent_ret, oracle_ret, anti_ret):
    if oracle_ret == anti_ret:
        return 50
    return float(np.clip(100 * (agent_ret - anti_ret) / (oracle_ret - anti_ret), 0, 100))


import numpy as np
import pandas as pd
import gymnasium as gym
from datetime import timedelta


class PositionTradingEnv(gym.Env):

    __version__ = 0

    def __init__(
        self,
        full_df: pd.DataFrame,
        ticker: str,
        market_features: list = ['close'],
        n_timesteps: int = 60,
        lookback: int = 0,
        seed: int = 42,
        start_idx=None
    ):
        super().__init__()
        self.full_df = full_df.copy()
        self.ticker = ticker
        self.n_timesteps = n_timesteps
        self.lookback = lookback
        self.market_features = market_features
        self.random_state = np.random.RandomState(seed)
        self.action_space = gym.spaces.Discrete(2)  # 0 = Flat, 1 = Long
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(len(market_features),), dtype=np.float32)
        self.fixed_start_idx = start_idx
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

        chosen_idx = self.fixed_start_idx if self.fixed_start_idx is not None else self.random_state.choice(valid_starts)

        self.start_idx = chosen_idx
        self.end_idx = self.start_idx + self.n_timesteps - 1
        self.lookback_idx = max(0, self.start_idx - self.lookback)
        self.episode_df = self.df.iloc[self.lookback_idx: self.end_idx + 1].reset_index(drop=True)

        self.prices = self.episode_df["close"].values
        self.episode_values = self.episode_df[self.market_features].values
        self._precompute_step_weights()

    def _precompute_step_weights(self):
        raw_weights = [abs(self.prices[i + 1] - self.prices[i]) for i in range(len(self.prices) - 1)]
        total = sum(raw_weights)
        self.step_weights = [w / total if total > 0 else 1 / len(raw_weights) for w in raw_weights]

    def reset(self, *, seed=None, options=None):
        self._resample_episode()
        self.step_idx = self.lookback
        self.position = 0
        self.total_reward = 0.0
        self.rewards = []
        self.actions = []
        self.values = []
        obs = np.array(self.episode_values[self.step_idx], dtype=np.float32)
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

        step_score = (agent_reward - anti_reward) / (oracle_reward - anti_reward) if oracle_reward != anti_reward else 0.5
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
        obs = np.array(self.episode_values[min(self.step_idx, len(self.prices) - 1)], dtype=np.float32)

        return obs, scaled_reward, terminated, truncated, {}


class PositionTradingEnvV1(PositionTradingEnv):

    __version__ = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_actions = [0, 0, 0]
        self.entry_price = 0
        self.holding_time = 0
        obs_dim = len(self.market_features) + 5 + 5 + 6
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.last_actions = [0, 0, 0]
        self.entry_price = self.prices[self.step_idx]
        self.holding_time = 0
        return self._get_observation(), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        price_now = self.prices[self.step_idx]
        if self.position == 0:
            if action == 0:
                self.holding_time += 1
            elif action == 1:
                self.entry_price = price_now
                self.holding_time = 0
        else:
            if action == 1:
                self.holding_time += 1
            elif action == 0:
                self.entry_price = 0
                self.holding_time = 0

        self.last_actions = self.last_actions[1:] + [action]
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        price_now = self.prices[self.step_idx]
        step_values = self.episode_values[self.step_idx]
        entry_price = self.entry_price if self.position else price_now

        pnl = (price_now - entry_price) / entry_price if entry_price > 0 else 0.0
        price_ratio = price_now / entry_price if entry_price > 0 else 1.0

        window_start = max(0, self.step_idx - 5)
        rolling_ret = np.mean(np.diff(self.prices[window_start:self.step_idx + 1]) / self.prices[window_start:self.step_idx + 1][:-1]) if self.step_idx > window_start else 0.0

        day = int(self.episode_df.iloc[self.step_idx]["day_of_week"])
        day_one_hot = np.zeros(5)
        day_one_hot[day] = 1

        action_onehots = []
        for a in self.last_actions:
            onehot = np.zeros(2)
            onehot[a] = 1
            action_onehots.extend(onehot)

        obs = np.array([
            self.position,
            self.holding_time,
            pnl,
            price_ratio,
            rolling_ret,
            *day_one_hot,
            *action_onehots,
            *step_values
        ], dtype=np.float32)
        return obs


def score_episode(agent_ret, oracle_ret, anti_ret):
    if oracle_ret == anti_ret:
        return 50
    return float(np.clip(100 * (agent_ret - anti_ret) / (oracle_ret - anti_ret), 0, 100))
