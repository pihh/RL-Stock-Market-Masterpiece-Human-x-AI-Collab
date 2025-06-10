import pandas as pd
def mean_policy(arr):
    # return np.median(arr)
    return pd.Series(arr).ewm(span=5).mean().iloc[-1]


EPISODE_LENGTH = 100

EXCLUDED_TICKERS = ['CEG', 'GEHC', 'GEV', 'KVUE', 'SOLV']
EXCLUDED_TICKERS.sort()

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