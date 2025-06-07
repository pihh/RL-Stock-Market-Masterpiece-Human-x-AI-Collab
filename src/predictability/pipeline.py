import os 
import json
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from src.data.utils import deep_hash
from .visualize import plot_heatmap
from .easiness import (
    rolling_sharpe, rolling_r2, rolling_info_ratio, rolling_autocorr
)

# Registry for available metrics—add more as you invent them!
METRIC_FUNCTIONS = {
    "rolling_sharpe": rolling_sharpe,
    "rolling_r2": rolling_r2,
    "rolling_info_ratio": rolling_info_ratio,
    "rolling_autocorr": rolling_autocorr,
    # Add more: "rolling_entropy": rolling_entropy, etc.
}

def generate_universe_easiness_report(
    ohlcv_df,
    tickers,
    window_length=60,
    metrics=("rolling_sharpe", "rolling_r2", "rolling_info_ratio", "rolling_autocorr"),
    target="return_1d",
    benchmark_col="market_return_1d",
    visualize=True,
    top_n=10,
    cutoff_start_date=None,
    cutoff_end_date=None,
    save_csv_path="data/experiments/predictability_metrics-{hash}.csv"
):
    """
    Calculates chosen predictability metrics for each ticker, saves/returns results, and optionally visualizes.
    """
    config = dict(
        tickers=list(tickers),
        window_length=window_length,
        metrics=list(metrics),
        target=target,
        benchmark_col=benchmark_col,
        cutoff_start_date=str(cutoff_start_date) if cutoff_start_date else None,
        cutoff_end_date=str(cutoff_end_date) if cutoff_end_date else None,
    )
    config_hash = deep_hash(config)
    csv_path = save_csv_path
    if '{hash}' in csv_path:
        csv_path = csv_path.format(hash=config_hash)
    
    if os.path.exists(csv_path):
        print(f"[CACHE] Loading existing study from {csv_path}")
        all_metrics = pd.read_csv(csv_path)
        
    else:
        if cutoff_end_date is not None:
            ohlcv_df = ohlcv_df[ohlcv_df['date'] <= cutoff_end_date]
            
        dfs = []
        for ticker in tqdm(tickers):
            df = ohlcv_df[ohlcv_df['symbol'] == ticker].copy()
            df['ticker'] = ticker

            # Always keep date and ticker for later
            base = df[['ticker', 'date', target]].copy()

            for metric in metrics:
                if metric not in METRIC_FUNCTIONS:
                    print(f"[WARNING] Metric {metric} not recognized, skipping.")
                    continue

                if metric == "rolling_info_ratio":
                    # Info ratio needs both target and benchmark
                    if benchmark_col not in df.columns:
                        print(f"[WARNING] Benchmark column '{benchmark_col}' not in dataframe, skipping info ratio.")
                        continue
                    base['info_ratio'] = rolling_info_ratio(df[target], df[benchmark_col], window_length)
                elif metric == "rolling_autocorr":
                    base['autocorr'] = rolling_autocorr(df[target], window=window_length, lag=1)
                elif metric == "rolling_sharpe":
                    base['sharpe'] = rolling_sharpe(df[target], window_length)
                elif metric == "rolling_r2":
                    base['r2'] = rolling_r2(df[target], window_length)
                # You can add more metrics here
            if cutoff_start_date is not None:
                base = base[base['date'] >= cutoff_start_date]
            dfs.append(base)

        all_metrics = pd.concat(dfs, ignore_index=True)
        all_metrics['config_hash'] = config_hash
        all_metrics['config_json'] = json.dumps(config, sort_keys=True)
    
        # Save to CSV for future use
        if save_csv_path is not None:
            all_metrics.to_csv(csv_path, index=False)
            #all_metrics.to_csv(save_csv_path, index=False)
            print(f"Saved all metrics to {csv_path} (config hash: {config_hash})")
            #print(f"Saved all metrics to {save_csv_path}")
    all_metrics.dropna(inplace=True)
    
    if visualize:
        # Distributions for Sharpe and R² across the entire universe
        if 'sharpe' in all_metrics:
            all_metrics['sharpe'].hist(bins=100, alpha=0.6, figsize=(8, 4))
            plt.title(f"Distribution of {window_length}-day Rolling Sharpe (all stocks, all periods)")
            plt.show()
        if 'r2' in all_metrics:
            all_metrics['r2'].hist(bins=100, alpha=0.6, figsize=(8, 4))
            plt.title(f"Distribution of {window_length}-day Rolling R² (all stocks, all periods)")
            plt.show()

        # Top tickers by Sharpe
        if 'sharpe' in all_metrics:
            recent_metrics = (
                all_metrics
                .groupby('ticker')
                .tail(1)
                .sort_values('sharpe', ascending=False)
                .head(top_n)
            )
            print(f"Top tickers by {window_length}-day Sharpe in last period:")
            display(recent_metrics[['ticker', 'sharpe', 'r2']] if 'r2' in recent_metrics else recent_metrics[['ticker', 'sharpe']])

            top_sharpe_tickers = (
                all_metrics
                .groupby('ticker')['sharpe']
                .mean()
                .sort_values(ascending=False)
                .head(top_n)
                .index.tolist()
            )
            plot_heatmap(all_metrics, 'sharpe', tickers=top_sharpe_tickers)

        # For each date, select the top 5 tickers by Sharpe
        if 'sharpe' in all_metrics:
            top_per_date = (
                all_metrics
                .groupby('date')
                .apply(lambda g: g.sort_values('sharpe', ascending=False).head(5))
                .reset_index(drop=True)
            )
            print("Sample of top 5 per date:")
            display(top_per_date.head())

 

    return all_metrics
