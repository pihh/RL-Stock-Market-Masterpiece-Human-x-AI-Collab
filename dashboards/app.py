import streamlit as st
import pandas as pd
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

# --- Make sure src is importable ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predictability.pipeline import generate_universe_easiness_report
from src.predictability.easiness import METRIC_FUNCTIONS

st.set_page_config(page_title="Universe Predictability Explorer", layout="wide")
st.title("📈 Universe Predictability Study Dashboard")

# --- Sidebar config ---
st.sidebar.header("Study Configuration")

@st.cache_data
def get_available_tickers(df_path="data/ohlcv_all.csv"):
    df = pd.read_csv(df_path)
    return sorted(df['symbol'].unique()), df

tickers, ohlcv_df = get_available_tickers()
selected_tickers = st.sidebar.multiselect("Tickers", tickers, default=tickers[:10])

window = st.sidebar.slider("Rolling window length", 20, 120, 60, step=5)

all_metric_names = list(METRIC_FUNCTIONS.keys())
metrics = st.sidebar.multiselect(
    "Metrics", all_metric_names, default=all_metric_names[:2]
)

target = st.sidebar.selectbox("Target column", ["return_1d", "log_return_1d"])
benchmark = st.sidebar.selectbox("Benchmark column", ["market_return_1d"])
cutoff_date = st.sidebar.date_input("Cutoff date (for study)", pd.to_datetime(ohlcv_df['date']).max())

run_btn = st.sidebar.button("Run/Load Study")

if run_btn:
    with st.spinner("Running or loading study..."):
        all_metrics, config, config_hash = generate_universe_easiness_report(
            ohlcv_df=ohlcv_df,
            tickers=selected_tickers,
            window_length=window,
            metrics=metrics,
            target=target,
            benchmark_col=benchmark,
            cutoff_date=cutoff_date,
            visualize=False,  # Visualize here with Plotly!
            top_n=10
        )
        st.success(f"Loaded study: config hash {config_hash}")

        st.write("## Study Config")
        st.json(config)

        st.write("## Sample Results")
        st.dataframe(all_metrics.head(20), use_container_width=True)

        # --- Advanced Plotly Visualizations ---

        st.write("## Interactive Visualizations")

        for metric in metrics:
            metric_col = metric.replace("rolling_", "")
            if metric_col not in all_metrics.columns:
                continue

            st.subheader(f"{metric_col} (all tickers, all dates)")

            # Distribution histogram
            fig_hist = px.histogram(all_metrics, x=metric_col, nbins=80,
                                   title=f"Distribution of {window}-day {metric_col}")
            st.plotly_chart(fig_hist, use_container_width=True)

            # Heatmap: Ticker vs Date
            st.write(f"Heatmap: {metric_col} by Ticker and Date")
            # Pivot to Date x Ticker
            df_pivot = all_metrics.pivot(index='date', columns='ticker', values=metric_col)
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=df_pivot.T.values,
                x=df_pivot.index,
                y=df_pivot.columns,
                colorscale='RdYlGn',
                colorbar=dict(title=metric_col),
            ))
            fig_heatmap.update_layout(height=500, width=1000, title=f"{metric_col} heatmap")
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Time series plot for top 5 tickers by mean metric
            agg = (
                all_metrics
                .dropna(subset=[metric_col])
                .groupby('ticker')[metric_col]
                .mean()
                .sort_values(ascending=False)
            )
            top5 = agg.head(5).index.tolist()
            st.write(f"Top 5 tickers by average {metric_col}:")
            fig_lines = px.line(
                all_metrics[all_metrics['ticker'].isin(top5)],
                x="date", y=metric_col, color="ticker",
                title=f"{metric_col} over time (top 5 tickers)",
            )
            st.plotly_chart(fig_lines, use_container_width=True)

            # Show Top/Bottom Table
            st.write("Top 5 tickers:")
            st.dataframe(agg.head(5))
            st.write("Bottom 5 tickers:")
            st.dataframe(agg.tail(5))

        # --- Download results ---
        st.write("## Download Results")
        st.download_button("Download metrics as CSV", data=all_metrics.to_csv(index=False), file_name=f"predictability_{config_hash}.csv")
