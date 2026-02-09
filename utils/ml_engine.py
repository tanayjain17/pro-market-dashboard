import nltk
nltk.download('punkt')
import pandas as pd
import numpy as np
import streamlit as st

def calculate_features(df):
    """Calculate technical indicators with proper formulas"""
    df = df.copy()
    
    if len(df) < 50:
        return None
    
    try:
        # Moving Averages
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        # RSI - Wilder's Smoothing (CORRECT)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # ATR & Volume
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        
        if 'Volume' in df.columns:
            df['Vol_SMA'] = df['Volume'].rolling(20).mean()
            df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA'].replace(0, np.nan)
        else:
            df['Vol_Ratio'] = 1.0
        
        df.dropna(inplace=True)
        return df
    
    except Exception as e:
        st.error(f"Feature calculation error: {e}")
        return None

def run_smart_prediction(ticker, df):
    """Generate trading verdict with confidence"""
    try:
        if df is None or len(df) < 50:
            return None
        
        # Extract latest values
        curr_price = df['Close'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        macd_hist = df['MACD_Hist'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        vol_ratio = df['Vol_Ratio'].iloc[-1]
        
        # Decision Logic
        verdict = "HOLD"
        color = "#888888"
        reason = "No clear signal"
        confidence = 0.5
        
        # BUY Signals
        if vol_ratio > 1.5 and curr_price > sma_50 and macd_hist > 0:
            verdict = "MOMENTUM BUY 🚀"
            color = "#00d09c"
            reason = "Volume Breakout + MACD Positive"
            confidence = 0.75
        
        elif rsi < 30 and macd_hist > 0:
            verdict = "OVERSOLD BUY 🟢"
            color = "#00d09c"
            reason = "RSI < 30 + MACD Turning Up"
            confidence = 0.65
        
        elif curr_price > sma_50 and rsi > 40 and rsi < 70 and macd_hist > 0:
            verdict = "TREND BUY 📈"
            color = "#00d09c"
            reason = "Price > SMA50, Healthy RSI, MACD Positive"
            confidence = 0.60
        
        # SELL Signals
        elif rsi > 75 and macd_hist < 0:
            verdict = "SELL 🔻"
            color = "#ff4b4b"
            reason = "Overbought (RSI > 75) + MACD Negative"
            confidence = 0.70
        
        elif curr_price < sma_50 and macd_hist < 0:
            verdict = "AVOID 📉"
            color = "#ff4b4b"
            reason = "Below SMA50 + MACD Negative"
            confidence = 0.65
        
        # Risk/Reward
        sl = curr_price - (2.0 * atr) if "BUY" in verdict else curr_price + (1.5 * atr)
        tgt = curr_price + (3.0 * atr) if "BUY" in verdict else curr_price - (2.0 * atr)
        
        return {
            'verdict': verdict,
            'color': color,
            'reason': reason,
            'confidence': confidence,
            'sl': sl,
            'tgt': tgt,
            'curr': curr_price,
            'rsi': rsi,
            'macd': macd_hist
        }
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

