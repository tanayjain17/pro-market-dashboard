import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.analytics_engine import (
    init_analytics_db,
    calculate_win_rate,
    get_performance_metrics,
    get_daily_stats,
    get_ticker_stats,
    get_performance_timeline
)

st.set_page_config(page_title="Analytics", layout="wide", page_icon="📊")

st.title("📊 Trading Analytics Dashboard")

# Initialize database
init_analytics_db()

# ==================== SIDEBAR FILTERS ====================
with st.sidebar:
    st.header("⚙️ Filters")
    
    days = st.slider("Days to analyze", 7, 365, 30)
    ticker_filter = st.text_input("Filter by ticker (optional)", "")

# ==================== MAIN METRICS ====================
st.header("📈 Performance Overview")

metrics = get_performance_metrics(days=days, ticker=ticker_filter if ticker_filter else None)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "📊 Total Trades",
        metrics['total_trades'],
        delta=f"{metrics['wins']} wins, {metrics['losses']} losses"
    )

with col2:
    st.metric(
        "🎯 Win Rate",
        f"{metrics['win_rate']:.1f}%",
        delta=f"{metrics['wins']} / {metrics['total_trades']}"
    )

with col3:
    st.metric(
        "💰 Profit Factor",
        f"{metrics['profit_factor']:.2f}",
        delta="Gross Profit / Loss"
    )

with col4:
    st.metric(
        "⚡ Sharpe Ratio",
        f"{metrics['sharpe_ratio']:.2f}",
        delta="Risk-adjusted return"
    )

with col5:
    st.metric(
        "📉 Max Drawdown",
        f"{metrics['max_drawdown']:.2f}%",
        delta="Worst peak-to-trough"
    )

# ==================== PERFORMANCE CHARTS ====================
st.divider()

col1, col2 = st.columns(2)

# Win Rate Pie Chart
with col1:
    st.subheader("Win/Loss Distribution")
    fig = go.Figure(data=[go.Pie(
        labels=['Wins', 'Losses'],
        values=[metrics['wins'], metrics['losses']],
        marker=dict(colors=['#00d09c', '#ff4b4b'])
    )])
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

# Average Win vs Loss
with col2:
    st.subheader("Avg Win vs Avg Loss")
    fig = go.Figure(data=[
        go.Bar(x=['Average Win', 'Average Loss'], 
               y=[metrics['avg_win'], abs(metrics['avg_loss'])],
               marker=dict(color=['#00d09c', '#ff4b4b']))
    ])
    fig.update_layout(template="plotly_dark", height=400, yaxis_title="Return (%)")
    st.plotly_chart(fig, use_container_width=True)

# ==================== DAILY PERFORMANCE ====================
st.divider()
st.subheader("📅 Daily Performance Timeline")

timeline_df = get_performance_timeline(days=days)

if not timeline_df.empty:
    # Line chart of daily returns
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timeline_df['date'],
        y=timeline_df['daily_return'],
        mode='lines+markers',
        name='Daily Return',
        line=dict(color='#00d09c', width=2),
        fill='tozeroy',
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.dataframe(
        timeline_df.assign(
            win_rate=lambda x: (x['wins'] / x['total'] * 100).round(1)
        ).sort_values('date', ascending=False),
        use_container_width=True
    )
else:
    st.warning("No performance data available for the selected period.")

# ==================== TICKER ANALYSIS ====================
st.divider()
st.subheader("📍 Ticker-wise Performance")

# Get all tickers from database
import sqlite3
conn = sqlite3.connect("trading_analytics.db")
cursor = conn.cursor()
cursor.execute('SELECT DISTINCT ticker FROM outcomes ORDER BY ticker')
tickers = [row[0] for row in cursor.fetchall()]
conn.close()

if tickers:
    selected_ticker = st.selectbox("Select Ticker", tickers)
    
    ticker_stats = get_ticker_stats(selected_ticker)
    
    if ticker_stats:
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        
        with t_col1:
            st.metric("Total Trades", ticker_stats['total_trades'])
        with t_col2:
            st.metric("Win Rate", f"{ticker_stats['win_rate']:.1f}%")
        with t_col3:
            st.metric("Avg Return", f"{ticker_stats['avg_return']:.2f}%")
        with t_col4:
            st.metric("Target Hits", ticker_stats['target_hits'])
else:
    st.info("No trading data recorded yet.")

# ==================== METRICS EXPLANATION ====================
st.divider()
st.subheader("📚 Metrics Explained")

with st.expander("What do these metrics mean?"):
    st.markdown("""
    - **Win Rate**: Percentage of profitable trades
    - **Profit Factor**: Gross profit divided by gross loss (>1 is profitable)
    - **Sharpe Ratio**: Risk-adjusted returns (higher is better, >1 is good)
    - **Max Drawdown**: Worst peak-to-trough decline (lower is better)
    - **Avg Win/Loss**: Average return on winning vs losing trades
    - **Target Hits**: Number of times price reached target
    - **SL Hits**: Number of times stop loss was triggered
    """)

st.markdown("---")
st.caption("🔄 Data updates automatically every 30 seconds")
