
import os
import sys

sys.path.append(os.path.abspath(".."))  

import numpy as np
import pandas as pd

from tqdm import tqdm

# Configuration ======================

from src.utils.system import boot
from notebooks.episodes import EpisodeTracker
from notebooks.tracker import OHLCV_DF, EpisodeTracker, AgentTracker, EnvironmentTracker


EXCLUDED_TICKERS = sorted(["CEG", "GEHC", "GEV", "KVUE", "SOLV"])

CONFIG = {
    "regressor": "RandomForestRegressor",
    "n_estimators": 300,
    "random_state": 314,
    "transaction_cost": 0,
}
LOOKBACK = 0
EPISODE_LENGTH = 50

RUN_SETTINGS = {
    "excluded_tickers": EXCLUDED_TICKERS,
    "cv_folds": 3,
    "lags": 5,
    "seed": 314,
    'total_timesteps':50_000,
    "episode": {
        "episode_length": EPISODE_LENGTH,
        "lookback": LOOKBACK,
    },
    "environment": {
        "market_features": ["close", "price_change", "volume_change"],
        "version": "v2",
        "lookback": LOOKBACK,
        "episode_length": EPISODE_LENGTH,
        "transaction_cost": 0,
    },
    "agent": {
        "model_class": "PPO",
        "policy_class": "MlpPolicy",
        "config": {
            "verbose": 1,
            "policy_kwargs": 
                {
                    "net_arch": [64, 64]
                    }
                },
    },
}


class EpisodeBenchmark:
    def __init__(
        self,
        tickers=["AAPL"],
        config=CONFIG,
        run_settings=RUN_SETTINGS,
        start_date="2023-01-01",
    ):
        self.ohlcv_df = OHLCV_DF.copy()
        self.tickers = tickers  # Force test with AAPL
        self.start_date = start_date

        self.config = CONFIG
        self.run_settings = RUN_SETTINGS

        self.ep_tracker = EpisodeTracker()
        self.env_tracker = EnvironmentTracker()
        self.agent_tracker = AgentTracker()

    def extract_agent_diagnostics(self, env, model, mode="train"):
        rewards = []
        obs = env.reset()[0]
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)

        # Residual diagnostics
        oracle = env.oracle_progress
        agent = env.wallet_progress
        r = np.array(agent) - np.array(oracle)

        returns = env.wallet_progress
        market_returns = env.market_progress

        diagnostics = {
            f"{mode}_total_rewards": env.total_rewards,
            f"{mode}_resid_ctd_diff": central_tendency_difference(
                np.mean(r), np.median(r), np.std(r)
            ),
            f"{mode}_resid_std": np.std(r),
            f"{mode}_resid_skew": skew(r),
            f"{mode}_resid_kurtosis": kurtosis(r),
            f"{mode}_resid_acf1": pd.Series(r).autocorr(lag=1),
            f"{mode}_ljung_pval": (
                acorr_ljungbox(r, lags=[min(10, len(r) - 1)], return_df=True).iloc[0][
                    "lb_pvalue"
                ]
                if len(r) > 10
                else np.nan
            ),
            f"{mode}_resid_mean": np.mean(r),
            f"{mode}_resid_median": np.median(r),
            f"{mode}_resid_max": np.max(r),
            f"{mode}_resid_min": np.min(r),
            f"{mode}_sharpe": sharpe_ratio(returns),
            f"{mode}_sortino": sortino_ratio(returns),
            f"{mode}_calmar": calmar_ratio(agent),
            f"{mode}_market_sharpe": sharpe_ratio(market_returns),
            f"{mode}_market_sortino": sortino_ratio(market_returns),
            f"{mode}_market_calmar": calmar_ratio(env.market_progress),
        }

        return diagnostics

    def run(self, tickers=None):
        # Configurations =============================
        config = self.config
        run_settings = self.run_settings

        # Feature Extraction Loop ====================
        features, targets, metadata, runs = [], [], [], []
        ohlcv_df = self.ohlcv_df.copy()

        if tickers == None:
            tickers = [self.tickers]
        
        seed = 314
        boot(seed)
        
        for symbol in tqdm(tickers):
            df = ohlcv_df[ohlcv_df["symbol"] == symbol].sort_values("date").copy()
            df = df[df["date"] > self.start_date]
            df = df.iloc[: -self.run_settings["episode"]["episode_length"]]
            months = df["month"].unique()

            for i , n in range(len(months)):
                try:

                    target_date = str(months[i]) + "-01"
                    episodes = self.ep_tracker.findEpisode(
                        target_date,
                        symbol,
                        episode_length=self.run_settings["episode"]["episode_length"],
                        lookback=self.run_settings["episode"]["lookback"],
                        mode="both",
                    )

                    train_episode = episodes["train"]
                    test_episode = episodes["test"]

                    env_tracker = EnvironmentTracker()

                    train_env_config = {
                        "ticker": symbol,
                        "n_timesteps": self.run_settings["episode"]["episode_length"],
                        "lookback": self.run_settings["episode"]["lookback"],
                        "market_features":self.run_settings['environment']['market_features'],
                        "seed": seed,
                        "start_idx": ep["train"]["df_start_iloc"],  # type: ignore
                    }
                    test_env_config = train_env_config.copy()
                    test_env_config["start_idx"] = ep["test"]["df_start_iloc"] # type: ignore

                    env_info = env_tracker.findEnvironment(
                        version="v2", config=train_env_config
                    )
                    train_env = env_info["environment"]
                    env_config["start_idx"] = ep["test"]["df_start_iloc"]
                    
                    test_env = env_tracker.findEnvironment(
                        version="v2", config=train_env_config
                    )
                    test_env = test_env["environment"]

                    tracker = AgentTracker()
                    
                    agent = tracker.findAgent(
                        **self.run_settings['agent']
                        #model_class="PPO",
                        #policy_class="MlpPolicy",
                        #config={"verbose": 1, "policy_kwargs": {"net_arch": [64, 64]}},
                        # name="ppo_mlp_baseline"
                    )
                    _model = agent["model"].boot(train_env)
                    _model.learn(total_timesteps=self.run_settings['total_timesteps'])

                    runs.append(
                        {
                            "train_episode_id": episodes["train"]["id"],
                            "test_episode_id": episodes["test"]["id"],
                            "total_timesteps": self.run_settings['total_timesteps'],
                            "ticker": symbol,
                            "target_date": target_date,
                            "environment_id": env_info["id"],
                            "agent_id": agent["id"],
                            "model": _model,
                            "train_env": train_env,
                            "test_env": test_env,
                        }
                    )
                    # features.append(feat)
                    # targets.append(cv_r2)
                    # metadata.append({
                    #    "symbol": symbol,
                    #    "target_date": target_date,
                    #    "train_episode_id": episodes['train']['episode_id'],
                    #    "test_episode_id": episodes['test']['episode_id']
                    # })

                except Exception as e:
                    print(f"Skipping {symbol} {months[i]} due to error: {e}")
                except:
                    print(months[i])
