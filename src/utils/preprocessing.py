import numpy as np
from scipy.stats import mstats

def rolling_winsorized_mean(series, window=60, limits=(0.01, 0.01)):
    def winsorized(x):
        return mstats.winsorize(x, limits=limits).mean()
    return series.rolling(window).apply(winsorized, raw=True)

def rolling_weighted_mean(series, window=60):
    # Linear weights: 1, 2, ..., window
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()  # normalize to sum to 1
    def weighted_avg(x):
        return np.dot(x, weights)
    return series.rolling(window).apply(weighted_avg, raw=True)

def rolling_exponential_weighted_median(df,target,name,window=60,span=60):
    df[name] = df[target].rolling(window=window).median().ewm(span=span, adjust=False).median()
    return df

def rolling_custom_weighted_mean(series, window=60, weights=None):
    if weights is None:
        # Example: Gaussian weights, centered at most recent
        center = window - 1
        weights = np.exp(-0.5 * ((np.arange(window) - center)/ (window/4))**2)
    weights = weights / weights.sum()
    def weighted_avg(x):
        return np.dot(x, weights)
    return series.rolling(window).apply(weighted_avg, raw=True)