# PREDICTABILITY METRICS ===========================

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestRegressor

def rolling_sharpe(returns, window=60):
    """
    Rolling Sharpe ratio for a return series.
    """
    mean = returns.rolling(window).median()
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
    mean_active = active_return.rolling(window).median()
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
#A. Downside Risk & Drawdown
def rolling_sortino(returns, window=60, target=0):
    """Rolling Sortino Ratio: mean return over downside std."""
    downside = returns.copy()
    downside[downside > target] = 0
    mean_return = returns.rolling(window).median()
    downside_std = downside.rolling(window).std()
    sortino = mean_return / (downside_std + 1e-8)
    return sortino

def rolling_max_drawdown(returns, window=60):
    """Rolling max drawdown (as a positive number) in the window."""
    wealth = (1 + returns).cumprod()
    max_dd = pd.Series(index=returns.index, dtype='float64')
    for i in range(window, len(wealth)):
        window_wealth = wealth.iloc[i-window:i]
        peak = window_wealth.cummax()
        dd = (window_wealth / peak) - 1
        max_dd.iloc[i] = -dd.min()
    return max_dd

def rolling_calmar(returns, window=60):
    """Rolling Calmar ratio: mean return over max drawdown."""
    mean_return = returns.rolling(window).median()
    max_dd = rolling_max_drawdown(returns, window)
    calmar = mean_return / (max_dd + 1e-8)
    return calmar

# B. Nonlinear Predictability
def hurst_exponent(ts):
    """Compute Hurst exponent of a series."""
    N = len(ts)
    if N < 20:
        return np.nan
    ts = np.array(ts)
    mean_ts = np.median(ts)
    Z = np.cumsum(ts - mean_ts)
    R = np.max(Z) - np.min(Z)
    S = np.std(ts)
    if S == 0: return np.nan
    return np.log(R / S) / np.log(N) if R > 0 and S > 0 else np.nan

def rolling_hurst(returns, window=60):
    """Rolling Hurst exponent."""
    return returns.rolling(window).apply(hurst_exponent, raw=False)


def approximate_entropy(U, m=2, r=None):
    """
    Approximate Entropy (ApEn) for a 1D array U.
    r: threshold (if None, set as 0.2*std(U) as common in literature)
    """
    U = np.array(U)
    N = len(U)
    if N < m + 1: return np.nan
    if r is None:
        r = 0.2 * np.std(U)
    def _phi(m):
        x = np.array([U[i:i+m] for i in range(N-m+1)])
        C = np.sum(np.max(np.abs(x[:,None,:] - x[None,:,:]), axis=2) <= r, axis=0) / (N-m+1)
        return np.sum(np.log(C)) / (N-m+1)
    return abs(_phi(m) - _phi(m+1))

def rolling_apen(returns, window=60, m=2, r=None):
    return returns.rolling(window).apply(lambda x: approximate_entropy(x, m, r), raw=False)

def rolling_variance_ratio(returns, window=60, lag=2):
    """
    Variance ratio: variance of k-period returns divided by k times variance of 1-period returns.
    < 1: mean reversion, > 1: trending.
    """
    vr = pd.Series(index=returns.index, dtype='float64')
    for i in range(window, len(returns)):
        x = returns.iloc[i-window:i]
        if x.isnull().any():
            vr.iloc[i] = np.nan
            continue
        single_var = np.var(x)
        multi_var = np.var(x.rolling(lag).sum().dropna())
        if single_var == 0:
            vr.iloc[i] = np.nan
        else:
            vr.iloc[i] = multi_var / (lag * single_var)
    return vr

#C. ML-Based Predictability


def rolling_predictive_r2(df, features, target='return_1d', window=60):
    """
    For each rolling window, fit a RF on all available features to predict target.
    Returns rolling out-of-sample R² (1-step ahead).
    """
    r2s = pd.Series(index=df.index, dtype='float64')
    for i in range(window, len(df)):
        train = df.iloc[i-window:i]
        test = df.iloc[[i]]
        if train[features].isnull().any().any() or test[features].isnull().any().any():
            r2s.iloc[i] = np.nan
            continue
        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]
        try:
            model = RandomForestRegressor(n_estimators=10, max_depth=3)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            ss_res = np.sum((y_test.values - y_pred)**2)
            ss_tot = np.sum((y_test.values - np.median(y_train))**2)
            r2s.iloc[i] = 1 - ss_res / (ss_tot + 1e-8)
        except Exception:
            r2s.iloc[i] = np.nan
    return r2s
#D. Regime Strength

def rolling_hmm_loglik(returns, window=60, n_states=2):
    """
    Rolling HMM log-likelihood as regime strength indicator.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # run KMeans or HMM here
        logliks = pd.Series(index=returns.index, dtype='float64')
        for i in range(window, len(returns)):
            x = returns.iloc[i-window:i].dropna().values.reshape(-1,1)
            try:
                model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
                model.fit(x)
                logliks.iloc[i] = model.score(x)
            except Exception:
                logliks.iloc[i] = np.nan
        return logliks


def rolling_volatility_regimes(returns, window=60, threshold=None):
    """
    Proportion of high/low volatility regimes in window.
    """
    if threshold is None:
        threshold = returns.rolling(window).std().median()  # adaptive threshold
    vol = returns.rolling(window).std()
    regime = (vol > threshold).astype(int)  # 1 = high vol, 0 = low vol
    # Optionally: sum/regime transitions in window
    return regime

#E. Liquidity and Microstructure
def rolling_avg_dollar_volume(df, window=60):
    
    """
    Rolling average daily dollar volume for liquidity screening.
    """
    return (df['close'] * df['volume']).rolling(window).median()


def rolling_spread_proxy(df, window=60):
    """
    Rolling mean of (high - low) as a bid-ask spread proxy.
    """
    if 'high' not in df.columns or 'low' not in df.columns:
        return pd.Series(index=df.index, data=np.nan)
    return (df['high'] - df['low']).rolling(window).median()