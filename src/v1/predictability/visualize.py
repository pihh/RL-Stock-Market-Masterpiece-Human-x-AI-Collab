import matplotlib.pyplot as plt

def plot_heatmap(metric_df, metric_col, tickers=None):
    """
    Plots a heatmap of the metric (e.g., rolling Sharpe) for selected tickers over time.
    """
    if tickers is not None:
        metric_df = metric_df[metric_df['ticker'].isin(tickers)]
    pivot = metric_df.pivot(index='date', columns='ticker', values=metric_col)
    plt.figure(figsize=(16, 8))
    plt.title(f"Heatmap of {metric_col} over time")
    plt.imshow(pivot.T, aspect='auto', interpolation='none', cmap='RdYlGn')
    plt.xlabel('Date')
    plt.ylabel('Ticker')
    plt.colorbar(label=metric_col)
    plt.show()
