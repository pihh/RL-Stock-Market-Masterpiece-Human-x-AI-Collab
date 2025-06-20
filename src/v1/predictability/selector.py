import pandas as pd
def rank_tickers_by_metric(df, metric_col, top_n=10, min_obs=100):
    """
    For each time period, select top_n tickers by the chosen predictability metric.
    Returns a DataFrame with selected tickers per date.
    """
    ranked = (
        df.groupby('date')
          .apply(lambda g: g.sort_values(metric_col, ascending=False)
                           .head(top_n) if len(g) > min_obs else pd.DataFrame())
          .reset_index(drop=True)
    )
    return ranked
