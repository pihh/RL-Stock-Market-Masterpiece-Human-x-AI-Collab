import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from gym import Env


def sample_valid_episodes(df, ticker, n_timesteps=60, lookback=0, episodes=30, seed=42):
    df = df[df['symbol'] == ticker].copy()
    df = df.sort_values('date')
    df['date'] = pd.to_datetime(df['date'])

    mondays = df[df['date'].dt.weekday == 0]
    valid_starts = []

    for date in mondays['date']:
        start_idx = df.index[df['date'] == date][0]
        end_idx = start_idx + n_timesteps - 1
        if end_idx >= len(df):
            continue

        episode = df.iloc[start_idx - lookback if start_idx - lookback >= 0 else 0 : end_idx + 1]
        if episode['symbol'].nunique() == 1 and episode['date'].is_monotonic_increasing:
            valid_starts.append(start_idx)

    rng = np.random.default_rng(seed)
    print(valid_starts,episodes)
    sampled_starts = rng.choice(valid_starts, size=episodes, replace=False)
    return sampled_starts


import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from hurst import compute_Hc
from statsmodels.tsa.stattools import adfuller

def extract_meta_features(df):
    features = {}

    close = df["close"].values
    returns = np.diff(close) / close[:-1]

    # Core statistical properties
    features["mean_return"] = np.mean(returns)
    features["median_return"] = np.median(returns)
    features["std_return"] = np.std(returns)
    features["skew_return"] = skew(returns)
    features["kurtosis_return"] = kurtosis(returns)
    features["return_trend"] = np.polyfit(np.arange(len(returns)), returns, 1)[0]
    try:
        ewm_mean = pd.Series(returns).ewm(span=10).mean().iloc[-1]
        features["ewm_mean_return"] = ewm_mean
    except Exception:
        features["ewm_mean_return"] = np.nan
    # Hurst exponent
    try:
        H, _, _ = compute_Hc(close, kind='price', simplified=True)
        features["hurst"] = H
    except Exception:
        features["hurst"] = np.nan

    # ADF test (stationarity)
    try:
        adf_stat, pval, *_ = adfuller(close)
        features["adf_stat"] = adf_stat
        features["adf_pval"] = pval
    except Exception:
        features["adf_stat"] = np.nan
        features["adf_pval"] = np.nan

    # Entropy (binned returns)
    hist, _ = np.histogram(returns, bins=10, density=True)
    hist = hist[hist > 0]
    features["entropy"] = -np.sum(hist * np.log(hist))

    return features


# def run_learning_evaluation(df, ticker="AAPL", timesteps=10_000, eval_episodes=30, n_timesteps=60, lookback=0, seed=42):
#     np.random.seed(seed)

#     # Sample episode start points
#     sampled_starts = sample_valid_episodes(df, ticker, n_timesteps, lookback, eval_episodes, seed)

#     # Train on the environment normally
#     env = Monitor(PositionTradingEnv(df, ticker, n_timesteps, lookback, seed=seed))
#     model = PPO("MlpPolicy", env, verbose=1, seed=seed)
#     model.learn(total_timesteps=timesteps)

#     # Evaluate PPO and Random with same episodes
#     ppo_scores = []
#     random_scores = []

#     for start_idx in sampled_starts:
#         # PPO agent evaluation
#         env_ppo = PositionTradingEnv(df, ticker, n_timesteps, lookback, seed=seed)
#         env_ppo.start_idx = start_idx  # override sampling
#         env_ppo.end_idx = start_idx + n_timesteps - 1
#         env_ppo.lookback_idx = max(0, start_idx - lookback)
#         env_ppo.episode_df = env_ppo.df.iloc[env_ppo.lookback_idx : env_ppo.end_idx + 1].reset_index(drop=True)
#         env_ppo.prices = env_ppo.episode_df["close"].values
#         env_ppo._precompute_step_weights()
#         obs, _ = env_ppo.reset()
#         done = False
#         while not done:
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, _ = env_ppo.step(action)
#             done = terminated or truncated
#         ppo_scores.append(env_ppo.total_reward)

#         # Random agent evaluation
#         env_rand = PositionTradingEnv(df, ticker, n_timesteps, lookback, seed=seed)
#         env_rand.start_idx = start_idx
#         env_rand.end_idx = start_idx + n_timesteps - 1
#         env_rand.lookback_idx = max(0, start_idx - lookback)
#         env_rand.episode_df = env_rand.df.iloc[env_rand.lookback_idx : env_rand.end_idx + 1].reset_index(drop=True)
#         env_rand.prices = env_rand.episode_df["close"].values
#         env_rand._precompute_step_weights()
#         obs, _ = env_rand.reset()
#         done = False
#         while not done:
#             action = env_rand.action_space.sample()
#             obs, reward, terminated, truncated, _ = env_rand.step(action)
#             done = terminated or truncated
#         random_scores.append(env_rand.total_reward)

#     t_stat, p_val = ttest_ind(ppo_scores, random_scores, equal_var=False)

#     return {
#         "ppo_mean": np.mean(ppo_scores),
#         "random_mean": np.mean(random_scores),
#         "t_stat": t_stat,
#         "p_val": p_val,
#         "ppo_scores": ppo_scores,
#         "random_scores": random_scores
#     }, model, env

# # --- Simulated test series ---
# def plot_evaluation_results(result_summary, title="Agent vs Random Performance"):
#     ppo_scores = result_summary["ppo_scores"]
#     random_scores = result_summary["random_scores"]
    
#     plt.figure(figsize=(10, 6))
#     sns.histplot(ppo_scores, color="green", label="PPO Agent", kde=True, stat="density", bins=10)
#     sns.histplot(random_scores, color="red", label="Random Policy", kde=True, stat="density", bins=10)

#     plt.axvline(np.mean(ppo_scores), color="green", linestyle="--", label=f"PPO Mean: {np.mean(ppo_scores):.2f}")
#     plt.axvline(np.mean(random_scores), color="red", linestyle="--", label=f"Random Mean: {np.mean(random_scores):.2f}")

#     plt.title(title)
#     plt.xlabel("Episode Score (0–100)")
#     plt.ylabel("Density")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
    
# result_summary = run_learning_evaluation(
#     df_raw[df_raw['symbol']=="AAPL"].reset_index(),
#     ticker='AAPL', 
#     timesteps=5000, 
#     eval_episodes=5, 
#     n_timesteps=30, 
#     lookback=0
# )