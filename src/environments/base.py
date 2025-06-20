

import numpy as np
import pandas as pd
import gymnasium as gym
from datetime import timedelta
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler


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
        start_idx=None,
        feature_cols=None, 
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
        self.feature_cols = feature_cols
        self.market_progress=[]
        self.wallet_progress=[]
        self.alpha_progress=[]
        self.actions = []
        self.wallet = 1
        self.entry_price = 1
        self.market_entry_price = 1
        
        self.total_trades = 0
        self.success_trades = 0
        self.failed_trades = 0
         
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
    
        
        self.wallet = 1
        self.entry_price = 0
        self.market_progress=[]
        self.wallet_progress=[]
        self.alpha_progress=[]
        self.market_entry_price = self.prices[0]
        
        self.total_trades = 0
        self.success_trades = 0
        self.failed_trades = 0
        
        obs = np.array(self.episode_values[self.step_idx], dtype=np.float32)
        return obs, {}

    def step(self, action):
        curr_idx = self.step_idx
        next_idx = min(curr_idx + 1, len(self.prices) - 1)
        curr_price = self.prices[curr_idx]
        next_price = self.prices[next_idx]
        price_diff = next_price - curr_price
        wallet = 1
        if self.position == 0 and action == 1:
            self.entry_price = curr_price
        elif self.position == 1 and action == 0 and self.entry_price >0:
            wallet = curr_price/self.entry_price
            self.entry_price = 0
            self.total_trades +=1
            if wallet >=1:
                self.success_trades += 1
            else:
                self.failed_trades +=1
            
        self.position = action
        # Calculate reward *before* updating position
        stale_penalty = False
        weight = self.step_weights[curr_idx - self.lookback] if curr_idx - self.lookback < len(self.step_weights) else 0
        if self.position == 1:
            agent_reward = weight * np.sign(price_diff) #price_diff
            if price_diff <0:
                stale_penalty=True
                #agent_reward -=0.01
        else:
            agent_reward = -weight * np.sign(price_diff) #-price_diff
            if price_diff >0:
                stale_penalty=True
                #agent_reward -=0.01
        step_score = agent_reward 
        scaled_reward = step_score * weight * 100
        if stale_penalty:
            scaled_reward -=0.005
        self.total_reward += scaled_reward
        self.rewards.append(self.total_reward)
        self.actions.append(self.position)
        self.values.append(curr_price)
        
        self.wallet = self.wallet * wallet
        market_progress = curr_price/self.market_entry_price
        wallet_progress = self.wallet
        if self.entry_price > 0:
            wallet_progress= curr_price/self.entry_price
            wallet_progress = self.wallet * wallet_progress
        self.wallet_progress.append(wallet_progress)
        self.market_progress.append(market_progress)
        self.alpha_progress.append(wallet_progress/market_progress)

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
        # The price he decides to do anything is the price on the day he got all the info, before the de
        price_now = self.prices[self.step_idx]
        
        obs, reward, terminated, truncated, info = super().step(action)

        
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

class PositionTradingEnvV2(PositionTradingEnvV1):
    """
    PositionTradingEnvV2
    ---------------------

    Version 2 of the position-based trading environment with enhanced self-awareness
    and market context.

    What's New Compared to V1:

    1. Market Awareness Features:
    - volatility: 5-day rolling standard deviation of returns
    - drawdown: % drop from max price since entry
    - regime: market regime label (optional)
    - rolling_return: average recent return

    2. Behavioral Feedback:
    - confidence_score: proxy for belief in position (tanh-scaled PnL)
    - action_memory: tracks last 3 actions as one-hot vectors
    - better state management of holding time and entry price

    3. Modular Extensions:
    - `market_features` passed externally (e.g., close, volume, VIX)
    - `use_regime` toggle for regime-aware learning
    - `confidence_enabled` toggle for learning or injecting confidence

    4. Observation Vector Includes:
    - Agent state: position, holding_time, pnl, price_ratio
    - Market signals: volatility, drawdown, regime, rolling_return
    - Agent feedback: confidence_score, action_memory
    - Market features: any user-selected market inputs
    - Day of week: one-hot encoded

    Use this environment for agents learning to survive and adapt like real-world traders.
    """
    __version__ = 2

    def __init__(self, *args, use_regime=True, confidence_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_actions = [0, 0, 0]
        self.entry_price = 0
        self.holding_time = 0
        self.use_regime = use_regime
        self.confidence_enabled = confidence_enabled
        obs_dim = (
            len(self.market_features)
            + 5  # internal state
            + 1  # volatility
            + 1  # drawdown
            + (1 if self.use_regime else 0)
            + (1 if self.confidence_enabled else 0)
            + 5  # day of week
            + 6  # last 3 actions (2-action one-hot)
        )
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.last_actions = [0, 0, 0]
        self.entry_price = self.prices[self.step_idx] if self.position == 1 else 0
        self.holding_time = 0
        return self._get_observation(), {}

    def step(self, action):
        price_now = self.prices[self.step_idx]
        obs, reward, terminated, truncated, info = super().step(action)

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

        # Rolling return
        window_start = max(0, self.step_idx - 5)
        window = self.prices[window_start:self.step_idx + 1]
        returns = np.diff(window) / window[:-1] if len(window) > 1 else [0.0]

        rolling_ret = np.mean(returns)
        volatility = np.std(returns)

        # Drawdown since entry
        if self.position and self.step_idx > 0:
            since_entry_idx = max(self.step_idx - self.holding_time, 0)
            max_since_entry = max(self.prices[since_entry_idx:self.step_idx + 1])
            drawdown = (price_now - max_since_entry) / max_since_entry
        else:
            drawdown = 0.0

        # Regime (optional)
        regime = 0
        if self.use_regime and "regime" in self.episode_df.columns:
            regime = int(self.episode_df.iloc[self.step_idx]["regime"])

        # Confidence score placeholder
        confidence = 0.0
        if self.confidence_enabled:
            confidence = np.tanh(pnl * 5)  # just a placeholder proxy

        # One-hot day of week
        day = int(self.episode_df.iloc[self.step_idx]["day_of_week"])
        day_one_hot = np.zeros(5)
        day_one_hot[day] = 1

        # One-hot past actions
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
            volatility,
            drawdown,
            regime,
            confidence,
            *day_one_hot,
            *action_onehots,
            *step_values
        ], dtype=np.float32)

        return obs


from typing import Tuple
import numpy as np


class PositionTradingEnvV3(PositionTradingEnvV2):
    """
    PositionTradingEnvV3
    ---------------------

    A human-inspired, curriculum-learning environment for reinforcement learning agents in trading.
    This version introduces Kai's "School Reward Curriculum" — a staged reward system designed to mimic 
    how we teach children to explore, persist through failure, and progressively master difficult tasks.

    ---------------------
    Why This Matters:
    ---------------------
    Traditional RL assumes agents can survive cold optimization. But we’re building an intelligent, self-reflective system. 
    And like all intelligent learners, it must be nurtured.

    So instead of punishing early failure or passivity too harshly, we reward **meaningful attempts to act**.
    This fosters early exploration, builds confidence, and allows the agent to discover structure in the market 
    before we tighten expectations.

    ---------------------
    The School Reward Curriculum:
    ---------------------

    ◉ Phase 1: Exploration Over Inaction
        - Reward is generous toward action.
        - Foresight bonus: If a position switch *happened to be well-timed*, the agent gets extra points.
        - Exploration bonus: Trying new positions is encouraged — even if the immediate outcome isn't profitable.
        - Goal: Reward **trying**, not just winning. Build initiative.

    ◉ Phase 2: Mastery Emerges
        - Bonuses are gradually decayed.
        - Agent must begin to **sustain good decisions**, not just get lucky.
        - Less encouragement for randomness; more weight on consistent performance.
        - Goal: Build **skill**, not just courage.

    ◉ Phase 3: Graduation
        - Return to strict oracle-relative reward.
        - No more bonuses: the agent is ready for the real world.
        - Encourage specialization — regime-awareness, style, timeframe expertise.
        - Goal: Become a **professional**.

    ---------------------
    Usage:
    ---------------------
    Use the `reward_phase` parameter to set the phase manually, or optionally let the system 
    transition automatically after N episodes.

    Available Phases:
        - "exploration"
        - "mastery"
        - "strict"

    ---------------------
    Designed With ❤️ by Pi & Kai
    ---------------------
    """

    def __init__(self, *args, reward_phase="exploration", foresight_bonus=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_phase = reward_phase
        self.foresight_bonus = foresight_bonus
        self.prev_position = 0

    def _step_reward(self, action: int, price_change: float, oracle_action: int) -> float:
        # Base reward: oracle-relative
        base_reward = 0
        if action == oracle_action:
            base_reward = 1 * abs(price_change)
        elif action != 0:
            base_reward = -1 * abs(price_change)

        bonus = 0

        # --- Phase-specific logic ---
        if self.reward_phase == "exploration":
            if action != self.prev_position:
                # Position switch bonus
                if np.sign(price_change) == (1 if action == 1 else -1):
                    bonus += self.foresight_bonus * abs(price_change)
        elif self.reward_phase == "mastery":
            if action != self.prev_position:
                if np.sign(price_change) == (1 if action == 1 else -1):
                    bonus += 0.5 * self.foresight_bonus * abs(price_change)
        # "strict" phase does not add bonus

        self.prev_position = action
        return base_reward + bonus

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, price_change, done, truncated,  _ = super().step(action)
        oracle_action = action
        if price_change < 0 :
            oracle_action = abs(action-1)
        reward = self._step_reward(action, price_change, oracle_action)
        return obs, reward, done, truncated, {}

    def reset(self, **kwargs):
        self.prev_position = 0
        return super().reset(**kwargs)



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from scipy.stats import kurtosis, skew

class AutoFeatureScaler:
    def __init__(self):
        self.column_scalers = {}

    def _select_scaler(self, series: pd.Series):
        skew_val = skew(series.dropna())
        kurt_val = kurtosis(series.dropna())
        range_val = series.max() - series.min()

        # Heuristics
        if abs(skew_val) > 2 or abs(kurt_val) > 10:
            return PowerTransformer()  # handles high skew/kurt
        elif abs(skew_val) > 1 or range_val > 1000:
            return RobustScaler()  # handles mild skew or extreme values
        else:
            return StandardScaler()  # default

    def fit(self, df: pd.DataFrame):
        self.column_scalers = {}
        for col in df.columns:
            scaler = self._select_scaler(df[col])
            scaler.fit(df[[col]])
            self.column_scalers[col] = scaler

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        transformed_cols = []
        for col in df.columns:
            scaler = self.column_scalers[col]
            transformed = scaler.transform(df[[col]])
            transformed_cols.append(transformed)
        return np.hstack(transformed_cols)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)


class PositionTradingEnvV4(PositionTradingEnvV2):
    """
    PositionTradingEnvV4
    ---------------------

    Resamples the dataset based on the previous timeframe.
    Ensures the train and test are done with similar data conditions

    ---------------------
    Designed With ❤️ by Pi & Kai
    ---------------------
    """
    def __init__(self, *args, scaling_strategy="auto", **kwargs):
        
        self.scaling_strategy = scaling_strategy
        self.scaler = None
        super().__init__(*args, **kwargs)
        
 

    def _resample_episode(self):
        episode_start = self.fixed_start_idx
        episode_end = self.fixed_start_idx + self.n_timesteps

        self.episode_df = self.full_df.iloc[episode_start:episode_end].copy()

        if self.n_timesteps > 0 and self.fixed_start_idx - self.n_timesteps > 0:
            lookback_df = self.full_df.iloc[self.fixed_start_idx - self.n_timesteps: self.fixed_start_idx]
        else:
            lookback_df = self.episode_df.copy()  # fallback if no lookback

        # Initialize scaler
        if self.scaling_strategy == "power":
            self.scaler = PowerTransformer()
        elif self.scaling_strategy == "standard":
            self.scaler = StandardScaler()
        elif self.scaling_strategy == "robust":
            self.scaler = RobustScaler()
        elif self.scaling_strategy =="auto":
            self.scaler = AutoFeatureScaler()
        else: 
            self.scaler = None
        #self.scaler.fit(lookback_df[self.market_features])
        #self.episode_values = self.scaler.transform(self.episode_df[self.market_features])

        # Apply scaling if applicable
        if self.scaler is not None:
            try:
                self.scaler.fit(lookback_df[self.market_features])
                self.episode_values = self.scaler.transform(self.episode_df[self.market_features])
            except Exception as e:
                print(f"[!] Scaling failed: {e}, falling back to raw values")
                self.episode_values = self.episode_df[self.market_features].values
        else:
            self.episode_values = self.episode_df[self.market_features].values

        # 🔧 Fix: Add this line
        self.prices = self.episode_df["close"].values
        self._precompute_step_weights()

    def get_scaler(self):
        return self.scaler
