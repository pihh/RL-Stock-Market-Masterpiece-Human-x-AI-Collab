import numpy as np
import pandas as pd
from datetime import timedelta

from src.features.ohlcv_feature_extraction import add_daily_features

def realistic_synthetic_market_sample(
    symbol="SYNTH",
    start_date="2022-01-01",
    n=31500,
    seed=150,
    vix=20,
    sp500=4000,
    base_price=100,
    signal_coefs={"order_flow": 0.01, "candle_body": 0.005},# signal_coefs: dict of {feature: coef} to make return_1d depend on them
   
    noise_std=0.002,
):
    np.random.seed(seed)
    start_dt = pd.Timestamp(start_date)
    date_range = [start_dt + timedelta(days=i) for i in range(n)]
    timestamps = [d + pd.Timedelta(hours=5) for d in date_range]
    weekday = [d.weekday() for d in date_range]

    # Create a baseline random walk for open price
    open_ = base_price + np.cumsum(np.random.normal(0, 0.1, n))
    close = open_ + np.random.normal(0, 0.15, n)
    high = np.maximum(open_, close) + np.abs(np.random.normal(0.2, 0.1, n))
    low = np.minimum(open_, close) - np.abs(np.random.normal(0.2, 0.1, n))
    volume = np.random.randint(2e6, 7e6, n)
    trade_count = np.random.randint(25000, 90000, n)
    vwap = (open_ + high + low + close) / 4

    # Compose a DataFrame for feature extraction
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'trade_count': trade_count,
        'vwap': vwap,
        'symbol': symbol,
    })

    # Run your feature extractor to get all other columns
    df = add_daily_features(df)

    # --- Inject the true signal (for learnability) ---
    # Example: Make order_flow & candle_body "cause" return_1d
    signal = np.zeros(n)
    for k, v in signal_coefs.items():
        if k in df.columns:
            signal += df[k].values * v
    # Add noise
    signal += np.random.normal(0, noise_std, n)
    # Overwrite return_1d and (if you want) market_return_1d
    df['return_1d'] = signal
    df['market_return_1d'] = signal * 0.4 + np.random.normal(0, noise_std/2, n)

    # Insert "sector_id", "industry_id", vix, sp500, etc., to be complete
    df['id'] = np.arange(1, n+1)
    df['date'] = [d.date() for d in date_range]
    df['weekday'] = weekday
    df['sector_id'] = 8
    df['industry_id'] = 51
    df['vix'] = vix + np.random.normal(0, 2, n)
    df['vix_norm'] = (df['vix'] - vix) / 5
    df['sp500'] = sp500 + np.random.normal(0, 25, n)
    df['sp500_norm'] = (df['sp500'] - sp500) / 40
   
    # All columns as in your schema
    all_cols = [
        'id','symbol','timestamp','date','open','high','low','close','volume','trade_count','vwap','weekday','day_of_month','day_of_week',
        'candle_size','order_flow','candle_body','upper_shadow','lower_shadow','price_change','candle_change','order_flow_change',
        'overnight_price_change','volume_change','vwap_change','trade_count_change','sector_id','industry_id','return_1d','vix','vix_norm',
        'sp500','sp500_norm','market_return_1d'
    ]
    return df[all_cols]
