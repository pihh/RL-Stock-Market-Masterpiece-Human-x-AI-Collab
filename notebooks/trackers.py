import hashlib
import os
import json
import pandas as pd
#from ..src.data.feature_pipeline import load_base_dataframe

STORAGE_PATH = "data/storage/"
def get_df_index_by_date(df,start_date="2024-01-01"):
    return df[df['date']>=start_date].iloc[0].name

def get_all_months(df):
    months = [str(m) +'-01' for m in df['date'].dt.to_period("M").unique()[24:]]
    return months


def generate_config_hash(config):
    raw = json.dumps(config, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()    
        
        
class EpisodeTracker:
    def __init__(self, path=STORAGE_PATH+"episodes.csv"):
        self.path = path
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            self.df = pd.read_csv(self.path)
        else:
            self.df = pd.DataFrame(columns=["id", "ticker", "start_date", "episode_length"])

    def findOrCreate(self, ticker, start_date, episode_length):
        match = self.df[
            (self.df["ticker"] == ticker) &
            (self.df["start_date"] == start_date) &
            (self.df["episode_length"] == episode_length)
        ]
        if not match.empty:
            return int(match.iloc[0]["id"])

        new_id = len(self.df)
        new_entry = {
            "id": new_id,
            "ticker": ticker,
            "start_date": start_date,
            "episode_length": episode_length
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_entry])], ignore_index=True)
        self.df.to_csv(self.path, index=False)
        return new_id

    def findById(self, id):
        match = self.df[self.df["id"] == id]
        return match.to_dict("records")[0] if not match.empty else None

    def findAll(self):
        return self.df.copy()
    
    def monthly_episode_walkforward(self,df,
        ticker,
        episode_length,
        start_date="2024-01-01",
        sample_train=True,
        sample_test=True):
    
        months = get_all_months(df)
        df_symbol = df[df['symbol']==ticker].reset_index().copy()
        episodes = []
        for month in months:
            episode_id = self.findOrCreate(ticker,start_date,episode_length)
            start_idx = get_df_index_by_date(df_symbol,start_date=month)
            episodes.append({
                "episode_id":episode_id,
                "date":month,
                "iloc":start_idx,
                "train_episode":df_symbol.iloc[start_idx-episode_length:start_idx],
                "test_episode": df_symbol.iloc[start_idx:start_idx+episode_length]
            })

        return episodes
    



class EnvironmentTracker:
    def __init__(self, path=STORAGE_PATH+"environments.csv"):
        self.path = path
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            self.df = pd.read_csv(self.path)
        else:
            self.df = pd.DataFrame(columns=["id", "version", "kwargs", "uuid"])

    def _createUuid(self,kwargs = {}):
        return generate_config_hash(kwargs)
    
    def findOrCreate(self,env_cls, kwargs):
        uuid = self._createUuid(kwargs)
        version = 'v'+str(env_cls.__version__)
        match = self.df[
            (self.df["version"] == version) &
            (self.df["uuid"] == uuid) 
        ]
        if not match.empty:
            record = match.iloc[0]
            record['kwargs']= json.loads(record['kwargs'])
            return record

        new_id = len(self.df)
        new_entry = {
            "id": new_id,
            "version": version,
            "uuid": uuid,
            "kwargs":json.dumps(kwargs)
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_entry])], ignore_index=True)
        self.df.to_csv(self.path, index=False)
        return new_id

    def findById(self, id):
        match = self.df[self.df["id"] == id]
        return match.to_dict("records")[0]['id'] if not match.empty else None

    def findAll(self):
        return self.df.copy()
    



class AgentTracker:
    def __init__(self, path=STORAGE_PATH+"agents.csv"):
        self.path = path
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            self.df = pd.read_csv(self.path)
        else:
            self.df = pd.DataFrame(columns=["id", "version", "kwargs", "uuid"])

    def _createUuid(self,kwargs = {}):
        return generate_config_hash(kwargs)
    
    def findOrCreate(self,agent_cls, kwargs):
        uuid = self._createUuid(kwargs)
        version = agent_cls.__version__
        match = self.df[
            (self.df["version"] == version) &
            (self.df["uuid"] == uuid) 
        ]
        if not match.empty:
            record = match.iloc[0]
            record['kwargs']= json.loads(record['kwargs'])
            return record

        new_id = len(self.df)
        new_entry = {
            "id": new_id,
            "version": version,
            "uuid": uuid,
            "kwargs":json.dumps(kwargs)
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_entry])], ignore_index=True)
        self.df.to_csv(self.path, index=False)
        return new_id

    def findById(self, id):
        match = self.df[self.df["id"] == id]
        return match.to_dict("records")[0] if not match.empty else None

    def findAll(self):
        return self.df.copy()
    