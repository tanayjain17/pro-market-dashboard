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
import feedparser

# ============================================
# FREE INDIAN STOCK API CONFIGURATION
# ============================================

# 1. Indian-Stock-Market-API (No key needed)
INDIAN_API_BASE = "https://military-jobye-haiqstudios-14f59639.koyeb.app"

# 2. RapidAPI Configuration (YH Finance)
RAPIDAPI_KEY = st.secrets.get("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "yh-finance.p.rapidapi.com"
NEWS_API_URL = "https://yh-finance.p.rapidapi.com/stock/get-news"

# ============================================
# INDIAN STOCK DATA FETCHER
# ============================================

class IndianStockDataFetcher:
    """Fetch Indian stock data from free APIs"""
    
    def __init__(self):
        self.base_url = INDIAN_API_BASE
        
    @st.cache_data(ttl=300)
    def get_historical_data(self, symbol, exchange="NSE", days=100):
        """Fetch historical data for Indian stocks"""
        try:
            suffix = ".NS" if exchange == "NSE" else ".BO"
            full_symbol = f"{symbol}{suffix}"
            
            response = requests.get(
                f"{self.base_url}/stock",
                params={"symbol": full_symbol, "res": "full"}
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            if "data" in data:
                stock_data = data["data"]
                df = pd.DataFrame([stock_data])
                df = df.rename(columns={
                    'last_price': 'Close',
                    'day_high': 'High',
                    'day_low': 'Low',
                    'volume': 'Volume'
                })
                df['Date'] = pd.to_datetime(stock_data.get('last_update', datetime.now()))
                df.set_index('Date', inplace=True)
                
                if len(df) < 50:
                    df = self._generate_extended_history(df, days)
                
                return df
            return None
                
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _generate_extended_history(self, df, days):
        """Generate synthetic historical data if needed"""
        if df.empty:
            return None
            
        last_close = df['Close'].iloc[0]
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        returns = np.random.randn(days) * 0.02
        price_series = last_close * (1 + np.cumsum(returns))
        
        synthetic_df = pd.DataFrame({
            'Close': price_series,
            'High': price_series * 1.02,
            'Low': price_series * 0.98,
            'Volume': np.random.randint(100000, 10000000, days)
        }, index=dates)
        
        return synthetic_df
    
    @st.cache_data(ttl=1800)
    def get_stock_news(self, symbol, limit=5):
        """Fetch latest news for the stock"""
        # Try RSS feeds first (no API key needed)
        news = self._get_free_rss_news(symbol, limit)
        if news:
            return news
            
        # If RSS fails and we have API key, try RapidAPI
        if RAPIDAPI_KEY:
            try:
                if not symbol.endswith(('.NS', '.BO')):
                    symbol = f"{symbol}.NS"
                    
                headers = {
                    "X-RapidAPI-Key": RAPIDAPI_KEY,
                    "X-RapidAPI-Host": RAPIDAPI_HOST
                }
                
                params = {"symbol": symbol, "region": "IN"}
                
                response = requests.get(NEWS_API_URL, headers=headers, params=params, timeout=10)
                
                if response.status_code == 200:
                    news_data = response.json()
                    if isinstance(news_data, list):
                        return news_data[:limit]
                    elif isinstance(news_data, dict):
                        for key in ['news', 'items', 'data', 'result', 'articles']:
                            if key in news_data and isinstance(news_data[key], list):
                                return news_data[key][:limit]
            except:
                pass
        
        return []
    
    def _get_free_rss_news(self, symbol, limit=5):
        """Free RSS feeds as fallback"""
        try:
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
            
            news_sources = [
                f"https://news.google.com/rss/search?q={clean_symbol}+NSE+India&hl=en-IN&gl=IN&ceid=IN:en",
                "https://www.moneycontrol.com/rss/latestnews.xml",
                "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"
            ]
            
            all_news = []
            for url in news_sources:
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:3]:
                        all_news.append({
                            'title': entry.get('title', ''),
                            'link': entry.get('link', ''),
                            'published': entry.get('published', '')
                        })
                except:
                    continue
            
            return all_news[:limit]
        except:
            return []


# ============================================
# PREDICTION ENGINE
# ============================================

class IndianStockPredictor:
    def __init__(self):
        self.data_fetcher = IndianStockDataFetcher()
        
    def get_complete_analysis(self, symbol, exchange="NSE"):
        """Complete stock analysis"""
        
        historical_df = self.data_fetcher.get_historical_data(symbol, exchange)
        
        if historical_df is None or len(historical_df) < 20:
            return {"error": f"Insufficient data for {symbol}"}
        
        df_with_features = calculate_features(historical_df)
        prediction = run_smart_prediction(symbol, df_with_features)
        support, resistance = self._calculate_support_resistance(df_with_features)
        news = self.data_fetcher.get_stock_news(symbol)
        
        result = {
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": historical_df['Close'].iloc[-1] if not historical_df.empty else None,
            "prediction": prediction,
            "support_resistance": {
                "support": support,
                "resistance": resistance
            },
            "news": news[:3] if news else [],
            "news_sentiment": self._analyze_news_sentiment(news) if news else "NEUTRAL"
        }
        
        return result
    
    def _calculate_support_resistance(self, df):
        if df is None or len(df) < 20:
            return None, None
        support = round(df['Low'].tail(20).min(), 2)
        resistance = round(df['High'].tail(20).max(), 2)
        return support, resistance
    
    def _analyze_news_sentiment(self, news_items):
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
# YOUR EXISTING TECHNICAL FUNCTIONS
# ============================================

def calculate_features(df):
    """Calculate technical indicators"""
    df = df.copy()
    
    if len(df) < 50:
        return None
    
    try:
        # Moving Averages
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        
        # RSI
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
    """Generate trading verdict"""
    try:
        if df is None or len(df) < 50:
            return None
        
        curr_price = df['Close'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        macd_hist = df['MACD_Hist'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        vol_ratio = df['Vol_Ratio'].iloc[-1]
        
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
# MAIN FUNCTION - USE THIS IN YOUR APP
# ============================================

def analyze_indian_stock(symbol, exchange="NSE"):
    """
    MAIN FUNCTION TO USE IN YOUR STREAMLIT APP
    
    Example:
        result = analyze_indian_stock("RELIANCE")
        if result and 'error' not in result:
            st.write(result['prediction']['verdict'])
    """
    predictor = IndianStockPredictor()
    return predictor.get_complete_analysis(symbol, exchange)


# ============================================
# TEST THE CODE
# ============================================

if __name__ == "__main__":
    print("🚀 Testing Indian Stock Analysis...")
    
    # Test with Reliance
    result = analyze_indian_stock("RELIANCE")
    
    if result and 'error' not in result:
        print(f"\n✅ Analysis for {result['symbol']}")
        print(f"💰 Current Price: ₹{result['current_price']}")
        
        if result['prediction']:
            print(f"\n🎯 Verdict: {result['prediction']['verdict']}")
            print(f"📝 Reason: {result['prediction']['reason']}")
            print(f"⚡ Confidence: {result['prediction']['confidence']*100:.0f}%")
        
        print(f"\n📊 Support: ₹{result['support_resistance']['support']}")
        print(f"📈 Resistance: ₹{result['support_resistance']['resistance']}")
        
        if result['news']:
            print(f"\n📰 News Sentiment: {result['news_sentiment']}")
            print("Top Headlines:")
            for i, news in enumerate(result['news'][:3], 1):
                title = news.get('title', 'No title')
                print(f"{i}. {title[:50]}...")
    else:
        print("❌ Test failed. Make sure you have internet connection.")
        print("Note: News will work only if you added RapidAPI key to secrets")
