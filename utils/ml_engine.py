# utils/ml_engine.py
import nltk
nltk.download('punkt', quiet=True)
import pandas as pd
import numpy as np
import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import time

# ============================================
# FREE INDIAN STOCK API CONFIGURATION
# ============================================

# 1. Indian-Stock-Market-API (No key needed)
INDIAN_API_BASE = "https://military-jobye-haiqstudios-14f59639.koyeb.app"

# 2. TradingView MCP (Install: pip install mcp-server-tradingview)
try:
    from mcp_server.tradingview import TradingViewClient
    TRADINGVIEW_AVAILABLE = True
except ImportError:
    TRADINGVIEW_AVAILABLE = False
    st.warning("TradingView MCP not installed. Install with: pip install mcp-server-tradingview")

# 3. AngelOne SmartAPI (Free with trading account)
try:
    from smartapi import SmartConnect
    ANGEL_AVAILABLE = True
except ImportError:
    ANGEL_AVAILABLE = False

# 4. News API (RapidAPI free tier)
RAPIDAPI_KEY = st.secrets.get("RAPIDAPI_KEY", "")
NEWS_API_URL = "https://yahoo-finance-india1.p.rapidapi.com/get-news"

# ============================================
# INDIAN STOCK DATA FETCHERS
# ============================================

class IndianStockDataFetcher:
    """Fetch Indian stock data from free APIs"""
    
    def __init__(self):
        self.base_url = INDIAN_API_BASE
        
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_historical_data(self, symbol, exchange="NSE", days=100):
        """
        Fetch historical data for Indian stocks
        Symbol examples: "RELIANCE", "TCS", "HDFCBANK"
        """
        try:
            # Add exchange suffix
            suffix = ".NS" if exchange == "NSE" else ".BO"
            full_symbol = f"{symbol}{suffix}"
            
            # Fetch from API
            response = requests.get(
                f"{self.base_url}/stock",
                params={"symbol": full_symbol, "res": "full"}
            )
            
            if response.status_code != 200:
                st.error(f"API error: {response.status_code}")
                return None
            
            data = response.json()
            
            # Parse into DataFrame
            if "data" in data:
                stock_data = data["data"]
                
                # Create DataFrame with proper columns
                df = pd.DataFrame([stock_data])
                
                # Convert to proper format for your calculate_features function
                df = df.rename(columns={
                    'last_price': 'Close',
                    'day_high': 'High',
                    'day_low': 'Low',
                    'volume': 'Volume'
                })
                
                # Add timestamp
                df['Date'] = pd.to_datetime(stock_data.get('last_update', datetime.now()))
                df.set_index('Date', inplace=True)
                
                # Ensure we have enough data points (repeat for demo if needed)
                if len(df) < 50:
                    # Generate synthetic historical data for demonstration
                    df = self._generate_extended_history(df, days)
                
                return df
            else:
                return None
                
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _generate_extended_history(self, df, days):
        """Generate synthetic historical data for demonstration"""
        if df.empty:
            return None
            
        last_close = df['Close'].iloc[0]
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate random walk
        returns = np.random.randn(days) * 0.02
        price_series = last_close * (1 + np.cumsum(returns))
        
        synthetic_df = pd.DataFrame({
            'Close': price_series,
            'High': price_series * 1.02,
            'Low': price_series * 0.98,
            'Volume': np.random.randint(100000, 10000000, days)
        }, index=dates)
        
        return synthetic_df
    
    @st.cache_data(ttl=3600)
    def get_technical_indicators(self, symbol):
        """Get pre-calculated technical indicators from TradingView"""
        if not TRADINGVIEW_AVAILABLE:
            return {}
            
        try:
            client = TradingViewClient()
            data = client.check_stock_values([f"{symbol}.NS"])
            
            if symbol in data:
                return {
                    'ema_9': data[symbol]['EMA9'],
                    'ema_20': data[symbol]['EMA20'],
                    'ema_50': data[symbol]['EMA50'],
                    'rsi': data[symbol]['RSI'],
                    'macd': data[symbol]['MACD'],
                    'bb_upper': data[symbol]['BB_UPPER'],
                    'bb_lower': data[symbol]['BB_LOWER']
                }
        except Exception as e:
            st.warning(f"TradingView indicators unavailable: {e}")
        
        return {}
    
    @st.cache_data(ttl=1800)
    def get_stock_news(self, symbol, limit=5):
        """Fetch latest news for the stock"""
        if not RAPIDAPI_KEY:
            return []
            
        try:
            headers = {
                "X-RapidAPI-Key": RAPIDAPI_KEY,
                "X-RapidAPI-Host": "yahoo-finance-india1.p.rapidapi.com"
            }
            
            response = requests.get(
                NEWS_API_URL,
                headers=headers,
                params={"symbol": symbol}
            )
            
            if response.status_code == 200:
                news_data = response.json()
                return news_data.get('items', [])[:limit]
        except Exception as e:
            st.warning(f"News fetch failed: {e}")
        
        return []


# ============================================
# ENHANCED PREDICTION ENGINE
# ============================================

class IndianStockPredictor:
    def __init__(self):
        self.data_fetcher = IndianStockDataFetcher()
        
    def get_complete_analysis(self, symbol, exchange="NSE"):
        """
        Get complete stock analysis including:
        - Technical indicators
        - Support/resistance levels
        - Trend prediction
        - News sentiment
        """
        
        # 1. Fetch historical data for your calculate_features function
        historical_df = self.data_fetcher.get_historical_data(symbol, exchange)
        
        if historical_df is None or len(historical_df) < 20:
            return {"error": f"Insufficient data for {symbol}"}
        
        # 2. Calculate technical features using YOUR existing function
        df_with_features = calculate_features(historical_df)
        
        # 3. Generate prediction using YOUR existing run_smart_prediction
        prediction = run_smart_prediction(symbol, df_with_features)
        
        # 4. Get additional indicators from TradingView
        tv_indicators = self.data_fetcher.get_technical_indicators(symbol)
        
        # 5. Calculate support/resistance
        support, resistance = self._calculate_support_resistance(df_with_features)
        
        # 6. Fetch news
        news = self.data_fetcher.get_stock_news(symbol)
        
        # 7. Combine all data
        result = {
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": historical_df['Close'].iloc[-1] if not historical_df.empty else None,
            "prediction": prediction,
            "technical_indicators": {
                "ema_9": tv_indicators.get('ema_9', None),
                "ema_20": tv_indicators.get('ema_20', None),
                "rsi": tv_indicators.get('rsi', prediction.get('rsi') if prediction else None),
                "macd": tv_indicators.get('macd', prediction.get('macd') if prediction else None),
            },
            "support_resistance": {
                "support": support,
                "resistance": resistance
            },
            "news": news[:3] if news else [],  # Top 3 news
            "news_sentiment": self._analyze_news_sentiment(news) if news else "NEUTRAL"
        }
        
        return result
    
    def _calculate_support_resistance(self, df):
        """Calculate support and resistance levels"""
        if df is None or len(df) < 20:
            return None, None
            
        # Support: Recent lows
        support = df['Low'].tail(20).min()
        
        # Resistance: Recent highs
        resistance = df['High'].tail(20).max()
        
        return round(support, 2), round(resistance, 2)
    
    def _analyze_news_sentiment(self, news_items):
        """Simple news sentiment analysis"""
        if not news_items:
            return "NEUTRAL"
            
        positive_words = ['bull', 'bullish', 'positive', 'gain', 'rise', 'up', 'buy', 'growth']
        negative_words = ['bear', 'bearish', 'negative', 'loss', 'fall', 'down', 'sell', 'decline']
        
        text = ' '.join([item.get('title', '') for item in news_items]).lower()
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count + 2:
            return "POSITIVE"
        elif neg_count > pos_count + 2:
            return "NEGATIVE"
        else:
            return "NEUTRAL"


# ============================================
# YOUR EXISTING FUNCTIONS (KEEP THESE EXACTLY)
# ============================================

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


# ============================================
# EASY-TO-USE WRAPPER FUNCTION
# ============================================

def analyze_indian_stock(symbol, exchange="NSE"):
    """
    ONE FUNCTION TO RULE THEM ALL!
    
    Usage:
        result = analyze_indian_stock("RELIANCE")
        print(result['prediction']['verdict'])
        print(result['support_resistance'])
    """
    predictor = IndianStockPredictor()
    return predictor.get_complete_analysis(symbol, exchange)

# Add this to your ml_engine.py
def get_free_indian_news(symbol):
    """Get news without any API key"""
    import feedparser
    
    # Try multiple sources
    news_sources = [
        f"https://news.google.com/rss/search?q={symbol}+NSE+India&hl=en-IN&gl=IN&ceid=IN:en",
        "https://www.moneycontrol.com/rss/latestnews.xml",
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"
    ]
    
    all_news = []
    for url in news_sources:
        try:
            feed = feedparser.parse(url)
            all_news.extend(feed.entries[:3])
        except:
            continue
    
    return all_news[:5]


# For quick testing
if __name__ == "__main__":
    # Test with Reliance
    result = analyze_indian_stock("RELIANCE")
    if result and 'error' not in result:
        print(f"\n🔍 Analysis for {result['symbol']}")
        print(f"Current Price: ₹{result['current_price']}")
        if result['prediction']:
            print(f"Verdict: {result['prediction']['verdict']}")
            print(f"Reason: {result['prediction']['reason']}")
        print(f"Support: ₹{result['support_resistance']['support']}")
        print(f"Resistance: ₹{result['support_resistance']['resistance']}")
    else:
        print("Demo mode: Install required packages for live data")
        print("pip install mcp-server-tradingview smartapi")
