
import os
import sys

sys.path.append(os.path.abspath(".."))  

import json
import hashlib
import sqlite3

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

from src.data.feature_pipeline import load_base_dataframe

# Optional import block if you're using SB3
from stable_baselines3 import PPO, A2C

from sb3_contrib import RecurrentPPO


# SYSTEM BOOT ==============================================
OHLCV_DF = load_base_dataframe()

# ENVIRONMENT TRACKER ======================================
class EnvironmentTracker:
    def __init__(self, db_path="../rl_trading.db", df=OHLCV_DF):
        self.db_path = Path(db_path)
        self.table = "environments"
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()
        #if type(df) != type(None):
        #    df = OHLCV_DF.copy()
        self.df = df.copy()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS environments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                version TEXT,
                config_hash TEXT,
                config_json TEXT,
                timestamp TEXT
            )
        """)

    def _sanitize_config(self, config):
        def convert(o):
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
            return o
        return {k: convert(v) for k, v in config.items()}

    def _resolve_class(self, version):
        from environments import (
            PositionTradingEnv,
            PositionTradingEnvV1,
            PositionTradingEnvV2,
            PositionTradingEnvV3,
            PositionTradingEnvV4,
        )
        version_map = {
            "v0": PositionTradingEnv,
            "v1": PositionTradingEnvV1,
            "v2": PositionTradingEnvV2,
            "v3": PositionTradingEnvV3,
            "v4": PositionTradingEnvV4,
        }
        if version not in version_map:
            raise ValueError(f"Unknown environment version: {version}")
        return version_map[version]
    
    def _filtered_config(self, config):
        exclude_keys = {"full_df", "seed", "start_idx", "feature_cols",'ticker'}
        return {k: v for k, v in config.items() if k not in exclude_keys}

    def _hash_config(self, version, config):
        config_str = json.dumps({
        
            "config": self._sanitize_config(self._filtered_config(config))
        }, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def insert(self, name, version, config):
        config = self._sanitize_config(config)
        config_json = json.dumps(config, sort_keys=True)
        config_hash = self._hash_config(version, config)
        now = datetime.now().isoformat()

        cursor = self.conn.cursor()
        cursor.execute(
            f"INSERT INTO {self.table} (name, version, config_hash, config_json, timestamp) VALUES (?, ?, ?, ?, ?)",
            (name, version, config_hash, config_json, now)
        )
        self.conn.commit()
        return cursor.lastrowid

    def find_by_hash(self, version, config):
        config_hash = self._hash_config(version, config)
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT id FROM {self.table} WHERE config_hash = ? AND version= ?", (config_hash,version) )
        row = cursor.fetchone()
        return row[0] if row else None

    def load_instance(self, id):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT version, config_json FROM {self.table} WHERE id = ?", (id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"No environment found with id={id}")

        version, config_json = row
        config = json.loads(config_json)
        version, config_json = row
        config = json.loads(config_json)
        env_class = self._resolve_class(version)
        #return env_class, config
        return {
            "env_id": id,
            "id": id,
            "version": version,
            "config": config,
            "environment": env_class
        }

    def findEnvironment(self, version, config, ticker=None, seed=None, start_idx=None, df=OHLCV_DF, name=None ):
        params = {"ticker":ticker,"seed":seed, "start_idx":start_idx}
        if config['market_features']:
            config['market_features'].sort()
            
        existing_id = self.find_by_hash(version, config)
        
        
        
        if existing_id:
            instance= self.load_instance(existing_id)
            instance['config'].update(config)
            for k,v in params.items():
                if params[k] != None:
                    instance['config'][k]=v
    
            instance["environment"]= instance["environment"](full_df=df.copy(),**instance['config'])
            return instance
        if name is None:
            name = f"env_{version}"
        
        new_id = self.insert(name, version, config)
        instance = self.load_instance(new_id)
        instance['config'].update(config)
        print(instance['config'],config)
        for k,v in params.items():
                #print(k,v)
            if params[k] != None:
                instance['config'][k]=v
        
        instance["environment"]= instance["environment"](full_df=df.copy(),**instance['config'])
        
        return instance


# EPISODE TRACKER ==========================================

class EpisodeTracker:
    def __init__(self, db_path="../rl_trading.db", df=OHLCV_DF):
        self.db_path = Path(db_path)
        self.table = "episodes"
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()
       
        self.df = df.copy()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                config_hash TEXT,
                config_json TEXT,
                timestamp TEXT
            )
        """)

    def _sanitize_config(self, config):
        def convert(o):
            if isinstance(o, (pd.Timestamp, pd.Timedelta)):
                return str(o)
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
            if isinstance(o, (pd.Series, pd.DataFrame)):
                return o.to_dict()
            return o

        return {k: convert(v) for k, v in config.items()}

    def _hash_config(self, config):
        if not isinstance(config, str):
            config = json.dumps(self._sanitize_config(config), sort_keys=True)
        return hashlib.sha256(config.encode()).hexdigest()

    def _get_df_by_instance(self,instance):
        df = self.df.copy()
        df = df[df['symbol']==instance['ticker']]
        df = df.iloc[:instance['df_end_iloc']]
        return df 
    
    def insert(self, name, config):
        config = self._sanitize_config(config)
        config_json = json.dumps(config, sort_keys=True)
        config_hash = self._hash_config(config_json)
        now = datetime.now().isoformat()

        cursor = self.conn.cursor()
        cursor.execute(
            f"INSERT INTO {self.table} (name, config_hash, config_json, timestamp) VALUES (?, ?, ?, ?)",
            (name, config_hash, config_json, now)
        )
        self.conn.commit()
        return cursor.lastrowid

    def find_by_hash(self, config):
        config = self._sanitize_config(config)
        config_hash = self._hash_config(json.dumps(config, sort_keys=True))
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT id FROM {self.table} WHERE config_hash = ?", (config_hash,))
        row = cursor.fetchone()
        return row[0] if row else None

    def findById(self, id):
        return self.load_instance(id)

    def load_instance(self, id):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT config_json FROM {self.table} WHERE id = ?", (id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"No episode found with id={id}")
        instance = json.loads(row[0])
        instance['df']= self._get_df_by_instance(instance)
        #return self._get_df_by_instance(instance)
        return instance# json.loads(row[0])

    def findEpisode(self, target_date, ticker, episode_length, lookback=0, mode="train", df=OHLCV_DF):
        if df is None and self.df is None:
            raise ValueError("DataFrame (df) must be provided.")

        
            #df = self.df.copy()
            
        symbol_df = df[df['symbol'] == ticker].reset_index(drop=True).copy()

        result = {"ticker": ticker, "episode_length": episode_length, "target_date": target_date, "train": None, "test": None}

        def create_episode(mode, start_iloc, end_iloc):
            episode_config = {
                "mode": mode,
                "ticker": ticker,
                "target_date": target_date,
                "episode_length": episode_length,
                "df_start_iloc": start_iloc,
                "df_end_iloc": end_iloc,
                "df_target_date_iloc": end_iloc if mode == "train" else start_iloc,
                "start_date": str(symbol_df.iloc[start_iloc]['date']),
                "end_date": str(symbol_df.iloc[end_iloc]['date'])
            }
            existing_id = self.find_by_hash(episode_config)
            if existing_id is not None:
                episode_config['df']= symbol_df.iloc[:episode_config['df_end_iloc']].copy()
                episode_config['id'] = existing_id
                return episode_config | {"episode_id": existing_id}

            name = f"{ticker}_{target_date}_{mode}"
            new_id = self.insert(name, episode_config)
            
            episode_config['df']= symbol_df.iloc[:end_iloc].copy() 
            episode_config['id'] = new_id
            return episode_config | {"episode_id": new_id}
        

        if mode in ["train", "both"]:
            train_end_iloc = symbol_df[symbol_df['date'] <= target_date].shape[0] - 1
            train_start_iloc = train_end_iloc - episode_length
            result["train"] = create_episode("train", train_start_iloc, train_end_iloc)

        if mode in ["test", "both"]:
            test_start_iloc = symbol_df[symbol_df['date'] > target_date].index.min()
            test_end_iloc = test_start_iloc + episode_length
            result["test"] = create_episode("test", test_start_iloc, test_end_iloc)

        return result


# AGENT TRACKER =============================================
class Agent:
    def __init__(self, model_class, policy_class, config):
        self.model_class = model_class
        self.policy_class = policy_class
        self.config = config
        self.model = None

    def boot(self, environment):
        self.model = self.model_class(self.policy_class, environment, **self.config)
        return self.model


class AgentTracker:
    def __init__(self, db_path="../rl_trading.db"):
        self.db_path = Path(db_path)
        self.table = "agents"
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                model_class TEXT,
                policy_class TEXT,
                config_hash TEXT,
                config_json TEXT,
                timestamp TEXT
            )
        """)

    def _filtered_config(self, config):
        exclude_keys = {"verbose", "seed"}
        return {k: v for k, v in config.items() if k not in exclude_keys}
    
    def _sanitize_config(self, config):
        def convert(o):
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
            return o
        return {k: convert(v) for k, v in config.items()}

    def _hash_config(self, model_class, policy_class, config):
        config = self._filtered_config(config)
        config_str = json.dumps({
            "model_class": model_class,
            "policy_class": policy_class,
            "config": self._sanitize_config(config)
        }, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def insert(self, name, model_class, policy_class, config):
        config = self._sanitize_config(config)
        config_json = json.dumps(config, sort_keys=True)
        config_hash = self._hash_config(model_class, policy_class, config)
        now = datetime.now().isoformat()

        cursor = self.conn.cursor()
        cursor.execute(
            f"INSERT INTO {self.table} (name, model_class, policy_class, config_hash, config_json, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (name, model_class, policy_class, config_hash, config_json, now)
        )
        self.conn.commit()
        return cursor.lastrowid

    def find_by_hash(self, model_class, policy_class, config):
        config_hash = self._hash_config(model_class, policy_class, config)
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT id FROM {self.table} WHERE config_hash = ?", (config_hash,))
        row = cursor.fetchone()
        return row[0] if row else None

    def load_instance(self, id):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT model_class, policy_class, config_json FROM {self.table} WHERE id = ?", (id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"No agent found with id={id}")

        model_class, policy_class, config_json = row
        config = json.loads(config_json)
        agent_cls = self._resolve_model(model_class)
        policy_cls = self._resolve_policy(policy_class)
        return Agent(agent_cls, policy_cls, config)

    def findAgent(self, model_class, policy_class, config, name=None):
        if "ent_coef" not in config.keys():
            config["ent_coef"]=0.1
        if "verbose" not in config.keys():
            config["verbose"]=1
            
        existing_id = self.find_by_hash(model_class, policy_class, config)
        if existing_id:
            instance=  self.load_instance(existing_id)
            return {"agent_id":existing_id,"id":existing_id,"model":instance}
        if name is None:
            name = f"{model_class}_{policy_class}_agent"
        new_id = self.insert(name, model_class, policy_class, config)
        
        instance = self.load_instance(new_id)
        return {"agent_id":new_id,"id":new_id,"model":instance}
    
    def _resolve_model(self, model_class):
        if model_class == "PPO": return PPO
        if model_class == "A2C": return A2C
        if model_class == "RecurrentPPO": return RecurrentPPO
        raise ValueError(f"Unknown model class: {model_class}")

    def _resolve_policy(self, policy_class):
        # These are strings intentionally to avoid circular imports
        return policy_class  # If you define your policies as strings like "MlpPolicy", this is fine


def create_db():
    from pathlib import Path
    import sqlite3

    DB_PATH = Path("../rl_trading.db")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    TABLE_DEFINITIONS = {
        "episodes": """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                config_hash TEXT,
                config_json TEXT,
                timestamp TEXT
            )
        """,
        "environments": """
            CREATE TABLE IF NOT EXISTS environments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                version TEXT,
                config_hash TEXT,
                config_json TEXT,
                timestamp TEXT
            )
        """,
        "agents": """
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                model_class TEXT,
                policy_class TEXT,
                config_hash TEXT,
                config_json TEXT,
                timestamp TEXT
            )
        """,
        "experiments": """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT,
                episode_id INTEGER,
                env_id INTEGER,
                agent_id INTEGER,
                seed INTEGER,
                config_hash TEXT,
                config_json TEXT,
                result_json TEXT,
                timestamp TEXT
            )
        """
    }

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for table_name, create_sql in TABLE_DEFINITIONS.items():
        cursor.execute(create_sql)

    conn.commit()
    conn.close()
    print("✅ Database ../rl_trading.db initialized successfully.")
