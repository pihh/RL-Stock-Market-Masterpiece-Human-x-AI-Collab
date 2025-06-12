import json
import hashlib
import sqlite3



def hash_config(config):
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()

def register_episode(conn, symbol, start_date, episode_length, lookback_window, feature_cols, learnability=None, extra_json=None):
    # Always sort feature_cols for consistent hashing
    episode_config = {
        "symbol": symbol,
        "start_date": start_date,
        "episode_length": episode_length,
        "lookback_window": lookback_window,
        "feature_cols": sorted(feature_cols),
    }
    config_json = json.dumps(episode_config, sort_keys=True)
    config_hash = hash_config(episode_config)

    # If extra_json is a dict, convert to string
    if isinstance(extra_json, dict):
        extra_json_str = json.dumps(extra_json, sort_keys=True)
    else:
        extra_json_str = extra_json

    c = conn.cursor()
    c.execute("""
        INSERT OR IGNORE INTO episode_traces
        (episode_hash, symbol, start_date, episode_length, lookback_window, feature_cols, learnability, extra_json, config_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        config_hash,  # episode_hash
        symbol,
        start_date,
        episode_length,
        lookback_window,
        json.dumps(sorted(feature_cols)),  # Store as JSON for easy parsing
        learnability,
        extra_json_str,
        config_hash  # config_hash (can be same as episode_hash)
    ))
    conn.commit()
    c.execute("""
        SELECT id FROM episode_traces WHERE episode_hash = ?
    """, (config_hash,))
    return c.fetchone()[0]

def register_environment(conn, name, path, config, version=None):
    config_json = json.dumps(config, sort_keys=True)
    config_hash = hash_config(config)
    c = conn.cursor()
    c.execute("""
        INSERT OR IGNORE INTO environments (name, path, version, hash, config_json)
        VALUES (?, ?, ?, ?, ?)
    """, (name, path, version, config_hash, config_json))
    conn.commit()   
    c.execute("""
        SELECT id FROM environments WHERE hash = ?
    """, (config_hash,))
    return c.fetchone()[0]

def register_model(conn, model_type, path, config, version=None):
    config_json = json.dumps(config, sort_keys=True)
    config_hash = hash_config(config)
    c = conn.cursor()
    c.execute("""
        INSERT OR IGNORE INTO models (model_type, path, version, hash, config_json)
        VALUES (?, ?, ?, ?, ?)
    """, (model_type, path, version, config_hash, config_json))
    conn.commit()
    c.execute("""
        SELECT id FROM models WHERE hash = ?
    """, (config_hash,))
    return c.fetchone()[0]

conn = None
class ConfigurableMixin:
    @property
    def conn(self):
        global conn
        if conn==None:
            conn = sqlite3.connect('../rl_trading.db')
        return conn 
    
    @property
    def config_json(self):
        return json.dumps(self.config, sort_keys=True)

    @property
    def config_hash(self):
        return hashlib.sha256(self.config_json.encode()).hexdigest()

    @property
    def config_uuid(self):
        # Optional: make it shorter, more readable
        return f"{self.__class__.__name__}_{self.config_hash[:10]}"
    
    def register_environment(self,name="", path="", config={}, version=None):
       return register_environment(self.conn,name=name,path=path,config=config,version=version)
   
    def register_model(self,model_type="", path="", config={}, version=None):
       return register_model(self.conn,model_type=model_type,path=path,config=config,version=version)
   
    def register_episode(self, symbol="", start_date="", episode_length=100, lookback_window=0, feature_cols=[], learnability=None, extra_json=None):
        return register_episode(
            self.conn,
            symbol=symbol,
            start_date=start_date,
            episode_length=episode_length,
            lookback_window=lookback_window,
            feature_cols=feature_cols,
            learnability=learnability,
            extra_json=extra_json
        )
    
    def get_episode_by_start_date(self,df, symbol, start_date, episode_length, window_size=0):
        """
        Returns the DataFrame slice for an episode of a given symbol,
        starting at `start_date` with `episode_length` rows, 
        and including a lookback window if needed.
        
        Args:
            df: full DataFrame with 'date', 'symbol', ...
            symbol: which ticker to use
            start_date: str or pd.Timestamp, start of the episode (inclusive)
            episode_length: int, number of steps in the episode
            window_size: int, number of days to look back for the first obs (0 = no lookback)
        Returns:
            episode_df: DataFrame slice for the episode (lookback+episode_length rows)
            episode_start_idx: index of the first day of the episode
        """
        sdf = df[df['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        start_idx = sdf[sdf['date'] == pd.to_datetime(start_date)].index
        if len(start_idx) == 0:
            raise ValueError(f"Date {start_date} not found for symbol {symbol}")
        start_idx = start_idx[0]
        # If lookback is used, get previous `window_size` rows too
        first_idx = max(0, start_idx - window_size)
        last_idx = start_idx + episode_length
        episode_df = sdf.iloc[first_idx:last_idx].reset_index(drop=True)
        
        return episode_df, start_idx