# PREDICTABILITY METRICS ===========================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def rolling_sharpe(returns, window=60):
    """
    Rolling Sharpe ratio for a return series.
    """
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    sharpe = mean / (std + 1e-8)
    return sharpe

def rolling_r2(returns, window=60):
    """
    Rolling R² of a linear model fitted to windowed returns.
    Measures how “linear/trend-following” the series is.
    """
    r2 = pd.Series(index=returns.index, dtype='float64')
    x = np.arange(window).reshape(-1, 1)
    for i in range(window, len(returns)):
        y = returns.iloc[i-window:i].values.reshape(-1, 1)
        if np.isnan(y).any():
            r2.iloc[i] = np.nan
            continue
        model = LinearRegression().fit(x, y)
        r2.iloc[i] = model.score(x, y)
    return r2

def rolling_info_ratio(returns, benchmark_returns, window=60):
    """
    Rolling Information Ratio: active return over tracking error to benchmark.
    """
    active_return = returns - benchmark_returns
    mean_active = active_return.rolling(window).mean()
    std_active = active_return.rolling(window).std()
    info_ratio = mean_active / (std_active + 1e-8)
    return info_ratio

def rolling_autocorr(returns, window=60, lag=1):
    """
    Rolling autocorrelation of returns (lag-1 by default).
    """
    autocorrs = pd.Series(index=returns.index, dtype='float64')
    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i]
        if window_returns.isnull().any():
            autocorrs.iloc[i] = np.nan
            continue
        autocorrs.iloc[i] = window_returns.autocorr(lag=lag)
    return autocorrs
