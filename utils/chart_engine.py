import plotly.graph_objects as go
import pandas as pd

def plot_candlestick(df, ticker):
    """Generate candlestick chart with MACD"""
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # SMA 50
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_50'],
        mode='lines',
        name='SMA 50',
        line=dict(color='rgba(0, 208, 156, 0.7)', width=2)
    ))
    
    # SMA 200
    if 'SMA_200' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_200'],
            mode='lines',
            name='SMA 200',
            line=dict(color='rgba(255, 107, 53, 0.7)', width=2)
        ))
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

def plot_macd(df):
    """Separate MACD subplot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        mode='lines',
        name='MACD',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD_Signal'],
        mode='lines',
        name='Signal',
        line=dict(color='red', width=2)
    ))
    
    # MACD Histogram (bars)
    colors = ['green' if x > 0 else 'red' for x in df['MACD_Hist']]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD_Hist'],
        name='Histogram',
        marker=dict(color=colors),
        opacity=0.3
    ))
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        xaxis_rangeslider_visible=False
    )
    
    return fig

