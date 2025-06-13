import pandas as pd
from sklearn.preprocessing import  RobustScaler,MinMaxScaler

# Default central tendency measurement 
def mean_policy(arr,span=5):
    # return np.median(arr)
    return pd.Series(arr).ewm(span=span).mean().iloc[-1]

def default_central_tendecy_measure(arr):
    # return np.median(arr)
    return mean_policy(arr)

DEFAULT_REGULAR_SCALER = MinMaxScaler
DEFAULT_SENSITIVE_SCALER = RobustScaler
DEFAULT_EPISODE_LENGTH = 100
# Random.org for it
RANDOM_SEEDS = [66923877,
203769678,
118482530,
135072239,
25166689,
9731925,
1674633,
644267,
89890134,
81048948]
EPISODE_LENGTH = DEFAULT_EPISODE_LENGTH

EXCLUDED_TICKERS = ['CEG', 'GEHC', 'GEV', 'KVUE', 'SOLV']
EXCLUDED_TICKERS.sort()

TOP2_STOCK_BY_SECTOR_DICT = {
    
}
TOP2_STOCK_BY_SECTOR = [
    "AAPL","MSFT",
    "JPM","V",
    'LLY','UNH',
    'AMZN','TSLA',
    'META','GOOGL',
    'GE','UBER',
    'COST','WMT',
    'XOM','CVX',
    'NEE','SO',
    'AMT','PLD',
    'LIN','SHW'
]
TOP2_STOCK_BY_SECTOR.sort()

FEATURE_COLS = [
  "day_of_month",                     
   "day_of_week",                                        
   "order_flow",                     
   "candle_body",                    
   "upper_shadow",                   
   "lower_shadow",                   
   "price_change",                   
   "candle_change",                  
   "order_flow_change",              
   "overnight_price_change",         
   "volume_change",                  
   "vwap_change",                    
   "trade_count_change",              
   "return_1d",                        
   "vix_norm",                          
   "market_return_1d",             
]
FEATURE_COLS.sort()