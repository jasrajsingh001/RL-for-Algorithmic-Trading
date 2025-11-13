# streamlit_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

from performance_metrics import calculate_performance_metrics
from analyze import (
    calculate_cumulative_return,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    simulate_buy_and_hold,
    simulate_sma_crossover,
)

# ---------------------------------------
# Page setup
# ---------------------------------------
st.set_page_config(page_title="RL Trading – Analytics Dashboard", layout="wide")
st.title("📊 RL Trading – Full Performance Analytics Dashboard")

RESULTS_DIR = "results"
PROCESSED_DIR = "processed"

# Utility
def load_result_csv(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(path):
        return pd.read_csv(path, index_col="date", parse_dates=True)
    return None

# ---------------------------------------
# Load available RL backtest files
# ---------------------------------------
original_df = load_result_csv("backtest_results.csv")
tuned_df    = load_result_csv("backtest_results_tuned.csv")

if original_df is None:
    st.error("❌ No backtest_results.csv found in /results. Run evaluate.py first.")
    st.stop()

# Select model for analysis
model_choice = st.sidebar.selectbox(
    "Model to analyze",
    ["Original PPO", "Tuned PPO"]
)

df_rl = original_df if model_choice == "Original PPO" else tuned_df

# Ensure portfolio_value column exists
if "portfolio_value" not in df_rl.columns:
    st.error("portfolio_value column missing in RL results.")
    st.stop()

# ---------------------------------------
# Load processed market data for baselines
# ---------------------------------------
ticker_guess = df_rl.columns[0] if "ticker" in df_rl.columns else "AMZN"
processed_path = os.path.join(PROCESSED_DIR, f"{ticker_guess}_all_splits.csv")

try:
    full_market_df = pd.read_csv(processed_path, index_col="Date", parse_dates=True, dayfirst=True)
except:
    st.warning("Processed data not found, baseline simulations may be unavailable.")
    full_market_df = None


# =======================================================
# TAB 1: Overview
# =======================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📊 Comparison", "⚠️ Risk Analytics", "📘 Trades & Actions"])

with tab1:
    st.subheader(f"Equity Curve – {model_choice}")

    pv = df_rl["portfolio_value"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pv.index, y=pv, mode="lines", name=model_choice))
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # KPIs
    metrics = calculate_performance_metrics(pv)
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Cumulative Return", f"{metrics['Cumulative Return (%)']:.2f}%")
    colB.metric("Annualized Return (CAGR)", f"{metrics['Annualized Return (%)']:.2f}%")
    colC.metric("Annualized Volatility", f"{metrics['Annualized Volatility (%)']:.2f}%")
    colD.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")

    colE, colF = st.columns(2)
    colE.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")
    colF.metric("Max Drawdown", f"{metrics['Maximum Drawdown (%)']:.2f}%")

    # Drawdown chart
    st.subheader("🔻 Drawdown Curve")
    running_max = pv.cummax()
    dd = (pv - running_max) / running_max

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy", name="Drawdown"))
    fig_dd.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_dd, use_container_width=True)



# =======================================================
# TAB 2: Comparison
# =======================================================
with tab2:
    st.subheader("Performance Comparison")

    if full_market_df is not None:
        # Baselines
        buy_hold = simulate_buy_and_hold(full_market_df, 100000)
        sma = simulate_sma_crossover(full_market_df, 100000, 20, 50, 0.001)

        comparison = {
            "RL (Selected)": pv,
            "Buy & Hold": buy_hold,
            "SMA 20/50": sma,
        }
    else:
        comparison = {"RL (Selected)": pv}

    # Plot all equity curves
    fig_all = go.Figure()
    for name, series in comparison.items():
        fig_all.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=name))
    fig_all.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_all, use_container_width=True)

    # Metrics table
    perf_data = {}
    for name, series in comparison.items():
        perf_data[name] = calculate_performance_metrics(series)

    perf_df = pd.DataFrame(perf_data).T
    st.dataframe(perf_df.style.format("{:.2f}"), use_container_width=True)

    # Radar chart
    radar_df = perf_df.copy()
    radar_df = radar_df[["Cumulative Return (%)", "Annualized Return (%)",
                         "Annualized Volatility (%)", "Sharpe Ratio",
                         "Sortino Ratio", "Maximum Drawdown (%)"]]

    radar_fig = px.line_polar(radar_df.reset_index(),
                              r="Cumulative Return (%)",
                              theta="index",
                              line_close=True)
    st.plotly_chart(radar_fig, use_container_width=True)


# =======================================================
# TAB 3: Risk Analytics
# =======================================================
with tab3:
    st.subheader("📉 Rolling Risk Metrics")

    returns = pv.pct_change().dropna()

    # Rolling volatility
    rolling_vol = returns.rolling(30).std() * np.sqrt(252)
    st.plotly_chart(px.line(rolling_vol, title="30-day Rolling Volatility",
                            labels={"value": "Volatility"},
                            template="plotly_dark"), use_container_width=True)

    # Rolling sharpe
    rolling_sharpe = returns.rolling(60).mean() / returns.rolling(60).std()
    st.plotly_chart(px.line(rolling_sharpe, title="60-day Rolling Sharpe",
                            labels={"value": "Sharpe"},
                            template="plotly_dark"), use_container_width=True)

    st.subheader("Worst Drawdowns")
    running_max = pv.cummax()
    dd = (pv - running_max) / running_max
    dd_sorted = dd.nsmallest(10)
    st.dataframe(dd_sorted.to_frame("Drawdown"), use_container_width=True)




# =======================================================
# TAB 4: Trades & Actions
# =======================================================
with tab4:
    if "action" in df_rl.columns:
        st.subheader("Trade/Action Log")
        st.dataframe(df_rl.tail(50), use_container_width=True)

        # Action distribution
        fig_actions = px.histogram(df_rl, x="action", nbins=3, template="plotly_dark")
        st.plotly_chart(fig_actions, use_container_width=True)

        # Action over time
        fig_act_time = px.scatter(df_rl, x=df_rl.index, y="action",
                                  template="plotly_dark")
        st.plotly_chart(fig_act_time, use_container_width=True)
    else:
        st.info("No 'action' column in RL results — run evaluate with action logging enabled.")
