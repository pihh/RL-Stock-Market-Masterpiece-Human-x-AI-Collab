import os
import json
import logging
import pandas as pd
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any

def deep_hash(config: Dict) -> int:
    """
    Generates a stable hash from a config dict using JSON serialization.
    """
    return hash(json.dumps(config, sort_keys=True))

# Configurable paths ========================
EXPERIMENT_DB = os.getenv("EXPERIMENT_DB", "rl_trading.db")
EXPERIMENT_CSV = os.getenv("EXPERIMENT_CSV", "../data/experiments/research-and-development.csv")

conn = sqlite3.connect(EXPERIMENT_DB)

class ExperimentTracker:
    """
    Tracks, logs, and manages experiment runs for RL research.
    - Deduplicates runs by config+target_date+phase.
    - Stores run configs and results as CSV and (optionally) SQLite.
    - Manages consistent episode sequences across experiments.
    """

    def __init__(
        self,
        project: str,
        phase: str = "r&d",
        train_episode_sequences: Optional[List[Any]] = None,
        test_episode_sequences: Optional[List[Any]] = None
    ):
        self.project = project
        self.phase = phase
        self.csv_path = EXPERIMENT_CSV
        self.train_episode_sequences = train_episode_sequences or []
        self.test_episode_sequences = test_episode_sequences or []
        self.run_hash = ""

    def set_hash(self,config):
        self.run_hash = deep_hash(config)
        
    def find_or_create_sequences(self, config: Dict, env) -> bool:
        """
        Loads or generates episode sequences for this config.
        Returns True if this is the first run (fresh split).
        """
        pdf = self.load_runs()
        run_hash = deep_hash(config)
        first_run = True
        try:
            pdf = pdf[pdf['hash'] == run_hash]
            if len(pdf) > 0:
                self.train_episode_sequences = json.loads(pdf.iloc[0]['train_episode_sequences'])
                self.test_episode_sequences = json.loads(pdf.iloc[0]['test_episode_sequences'])
                if self.train_episode_sequences and self.test_episode_sequences:
                    logging.info('Loaded previous episode sequences.')
                    first_run = False
                else:
                    self.train_episode_sequences = env.generate_episode_sequences(config['total_timesteps'])
                    self.test_episode_sequences = [("AAPL", 0)]
            else:
                self.train_episode_sequences = env.generate_episode_sequences(config['total_timesteps'])
                self.test_episode_sequences = [("AAPL", 0)]
        except Exception as e:
            logging.warning(f"Error loading sequences: {e}")
            self.train_episode_sequences = env.generate_episode_sequences(config['total_timesteps'])
            self.test_episode_sequences = [("AAPL", 0)]
        return first_run

    def save_run(
        self,
        config: Dict,
        results: Dict,
        target_date: str,
        run_settings: Dict,
        files: Optional[Dict] = None
    ) -> None:
        """
        Saves experiment config and results to CSV.
        Avoids duplicate runs.
        """
        run_hash = deep_hash(config)
        files = files or {}
        mode = run_settings.get('mode', 'in-sample')
        experiment_sequence = self.train_episode_sequences if mode == "in-sample" else self.test_episode_sequences

        row = {
            "project": self.project,
            "target_date": target_date,
            "hash": run_hash,
            "date": datetime.now().isoformat(),
            "config_json": json.dumps(config, sort_keys=True),
            "results_json": json.dumps(results, sort_keys=True),
            "run_settings_json": json.dumps(run_settings, sort_keys=True),
            "experiment_sequence": json.dumps(experiment_sequence),
            "train_episode_sequences": json.dumps(self.train_episode_sequences),
            "test_episode_sequences": json.dumps(self.test_episode_sequences),
            "phase": self.phase,
            "files": json.dumps(files)
        }

        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            is_duplicate = (
                (df["hash"] == run_hash) &
                (df["target_date"] == target_date) &
                (df["project"] == self.project) &
                (df["run_settings_json"] == json.dumps(run_settings, sort_keys=True)) &
                (df["phase"] == self.phase)
            ).any()
            if is_duplicate:
                logging.info(f"Experiment [{self.project} | {target_date} | {run_hash}] already exists—skipping.")
                return
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(self.csv_path, index=False)
        logging.info(f"Saved experiment [{self.project} | {target_date} | {run_hash}].")

    def load_runs(
        self,
        config: Dict = {},
        target_date: Optional[str] = None,
        by_hash: bool = False,
        csv_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Loads all runs for this project+phase. Optional filters by config or target_date.
        """
        run_hash = deep_hash(config)
        if csv_path is None:
            csv_path = self.csv_path

        if not os.path.exists(csv_path):
            return pd.DataFrame([])

        df = pd.read_csv(csv_path)
        df = df[df['project'] == self.project]
        df = df[df['phase'] == self.phase]
        if target_date is not None:
            df = df[df['target_date'] == target_date]
        if by_hash:
            df = df[df['hash'] == run_hash]
        return df

    def did_run(
        self,
        config: Dict,
        target_date: str,
        run_settings: Dict,
        csv_path: Optional[str] = None
    ) -> bool:
        """
        Checks if a run with given config and settings already exists.
        """
        run_hash = deep_hash(config)
        if csv_path is None:
            csv_path = self.csv_path

        if not os.path.exists(csv_path):
            return False
        df = pd.read_csv(csv_path)
        return (
            (df["hash"] == run_hash) &
            (df["target_date"] == target_date) &
            (df["project"] == self.project) &
            (df["run_settings_json"] == json.dumps(run_settings, sort_keys=True)) &
            (df["phase"] == self.phase)
        ).any()

    def expand_dataframe(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Flattens out config/results JSON columns for quick metrics analysis.
        """
        # Parse JSON columns
        json_cols = ['config_json', 'results_json', 'run_settings_json', 'files']
        for col in json_cols:
            results_df[col] = results_df[col].apply(json.loads)
        # Extract metrics for analysis (assumes 'test' and 'train' keys)
        def extract_metrics(row, section):
            return pd.Series(row['results_json'].get(section, {}))
        test_metrics = results_df.apply(lambda row: extract_metrics(row, 'test'), axis=1)
        train_metrics = results_df.apply(lambda row: extract_metrics(row, 'train'), axis=1)
        df_flat = pd.concat([
            results_df[["project", "target_date", "date"]],
            results_df["config_json"].apply(lambda x: x.get("reward", None)).rename("reward_fn"),
            test_metrics.add_prefix("test_"),
            train_metrics.add_prefix("train_")
        ], axis=1)
        return df_flat

