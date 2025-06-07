# Registry for available metrics—add more as you invent them!
from src.predictability.easiness import rolling_r2, rolling_sharpe,rolling_info_ratio,rolling_autocorr,rolling_sortino,rolling_max_drawdown,rolling_calmar,rolling_hurst,rolling_apen,rolling_variance_ratio,rolling_predictive_r2,rolling_hmm_loglik,rolling_volatility_regimes,rolling_avg_dollar_volume,rolling_spread_proxy


METRIC_FUNCTIONS = {
    "rolling_sharpe": rolling_sharpe,
    "rolling_r2": rolling_r2,
    "rolling_info_ratio": rolling_info_ratio,
    "rolling_autocorr": rolling_autocorr,
    "rolling_sortino":rolling_sortino,
    "rolling_max_drawdown":rolling_max_drawdown,
    "rolling_calmar":rolling_calmar,
    "rolling_hurst":rolling_hurst,
    "rolling_apen":rolling_apen,
    "rolling_variance_ratio":rolling_variance_ratio,
    "rolling_predictive_r2":rolling_predictive_r2,
    "rolling_hmm_loglik":rolling_hmm_loglik,
    "rolling_volatility_regimes":rolling_volatility_regimes,
    "rolling_avg_dollar_volume":rolling_avg_dollar_volume,
    "rolling_spread_proxy":rolling_spread_proxy

}