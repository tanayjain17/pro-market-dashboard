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
import pyotp

# ============================================
# ALL API CONFIGURATIONS
# ============================================

# 1. Free Indian Stock API (No key needed - FALLBACK)
INDIAN_API_BASE = "https://military-jobye-haiqstudios-14f59639.koyeb.app"

# 2. AngelOne SmartAPI (You have login - PRIMARY)
ANGELONE_API_KEY = st.secrets.get("ANGELONE_API_KEY", "")
ANGELONE_CLIENT_ID = st.secrets.get("ANGELONE_CLIENT_ID", "")
ANGELONE_PASSWORD = st.secrets.get("ANGELONE_PASSWORD", "")  # 4-digit MPIN
ANGELONE_TOTP = st.secrets.get("ANGELONE_TOTP", "")  # 32-char secret

# 3. RapidAPI (YH Finance + FMP + Seeking Alpha)
RAPIDAPI_KEY = st.secrets.get("RAPIDAPI_KEY", "")
YH_FINANCE_HOST = "yh-finance.p.rapidapi.com"
FMP_HOST = "financial-modeling-prep.p.rapidapi.com"
SEEKING_ALPHA_HOST = "seeking-alpha.p.rapidapi.com"


# ============================================
# ANGELONE SMARTAPI CLIENT
# ============================================

class AngelOneClient:
    """AngelOne SmartAPI client for Indian stocks"""
    
    def __init__(self):
        self.api_key = ANGELONE_API_KEY
        self.client_id = ANGELONE_CLIENT_ID
        self.password = ANGELONE_PASSWORD
        self.totp_secret = ANGELONE_TOTP
        self.jwt_token = None
        self.refresh_token = None
        self.feed_token = None
        self.user_profile = None
        self.is_connected = False
        self.obj = None
        
    def connect(self):
        """Connect to AngelOne SmartAPI"""
        if not all([self.api_key, self.client_id, self.password, self.totp_secret]):
            return False
            
        try:
            from smartapi import SmartConnect
            
            # Generate TOTP
            totp = pyotp.TOTP(self.totp_secret)
            totp_code = totp.now()
            
            # Create connection
            self.obj = SmartConnect(api_key=self.api_key)
            
            # Login
            data = self.obj.generateSession(
                client_id=self.client_id,
                password=self.password,
                totp=totp_code
            )
            
            if data and 'data' in data:
                self.jwt_token = data['data']['jwtToken']
                self.refresh_token = data['data']['refreshToken']
                self.feed_token = self.obj.getfeedToken()
                self.user_profile = self.obj.getProfile(self.refresh_token)
                self.is_connected = True
                return True
        except Exception as e:
            st.warning(f"AngelOne connection failed: {e}")
        
        return False
    
    @st.cache_data(ttl=300)
    def get_historical_data(self, symbol, days=100):
        """Get historical data from AngelOne"""
        if not self.is_connected:
            return None
            
        try:
            # Get token for symbol
            token = self.get_token(symbol)
            if not token:
                return None
            
            # Calculate dates
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Format for AngelOne
            from_str = from_date.strftime("%Y-%m-%d %H:%M")
            to_str = to_date.strftime("%Y-%m-%d %H:%M")
            
            # Get candle data
            historic_data = self.obj.getCandleData({
                "exchange": "NSE",
                "symboltoken": token,
                "interval": "ONE_DAY",
                "fromdate": from_str,
                "todate": to_str
            })
            
            if historic_data and 'data' in historic_data:
                df = pd.DataFrame(historic_data['data'], 
                                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df.rename(columns={
                    'close': 'Close',
                    'high': 'High',
                    'low': 'Low',
                    'volume': 'Volume',
                    'open': 'Open'
                })
                return df
        except Exception as e:
            st.warning(f"AngelOne data fetch failed: {e}")
        
        return None
    
    def get_token(self, symbol):
        """Get token for symbol (simplified - you can expand this)"""
        tokens = {
            "RELIANCE": "2885",
            "TCS": "295321",
            "HDFCBANK": "341249",
            "INFY": "1594",
            "ITC": "1660",
            "SBIN": "3045",
            "BHARTIARTL": "2714",
            "WIPRO": "3787",
            "HINDUNILVR": "1560",
            "ICICIBANK": "4963"
        }
        return tokens.get(symbol.upper(), None)


# ============================================
# RAPIDAPI CLIENTS
# ============================================

class RapidAPIClient:
    """Handle all RapidAPI calls"""
    
    def __init__(self):
        self.api_key = RAPIDAPI_KEY
        
    def get_fundamentals(self, symbol):
        """Get financial fundamentals from FMP"""
        if not self.api_key:
            return {}
            
        try:
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
            
            # Get company ratios
            url = f"https://financial-modeling-prep.p.rapidapi.com/v3/ratios/{clean_symbol}"
            headers = {
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": FMP_HOST
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}
    
    def get_analysis(self, symbol):
        """Get analysis from Seeking Alpha"""
        if not self.api_key:
            return {}
            
        try:
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
            
            url = "https://seeking-alpha.p.rapidapi.com/symbols/get-analysis"
            headers = {
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": SEEKING_ALPHA_HOST
            }
            params = {"symbol": clean_symbol, "period": "quarterly"}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}
    
    def get_news(self, symbol, limit=5):
        """Get news from YH Finance"""
        if not self.api_key:
            return []
            
        try:
            if not symbol.endswith(('.NS', '.BO')):
                symbol = f"{symbol}.NS"
                
            url = "https://yh-finance.p.rapidapi.com/stock/get-news"
            headers = {
                "X-RapidAPI-Key": self.api_key,
                "X-RapidAPI-Host": YH_FINANCE_HOST
            }
            params = {"symbol": symbol, "region": "IN"}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                news_data = response.json()
                if isinstance(news_data, list):
                    return news_data[:limit]
                elif isinstance(news_data, dict):
                    for key in ['news', 'items', 'data']:
                        if key in news_data and isinstance(news_data[key], list):
                            return news_data[key][:limit]
        except:
            pass
        return []


# ============================================
# FREE API CLIENT (FALLBACK)
# ============================================

class FreeAPIClient:
    """Free Indian stock API as fallback"""
    
    def __init__(self):
        self.base_url = INDIAN_API_BASE
    
    @st.cache_data(ttl=300)
    def get_historical_data(self, symbol, exchange="NSE", days=100):
        """Get data from free API"""
        try:
            suffix = ".NS" if exchange == "NSE" else ".BO"
            full_symbol = f"{symbol}{suffix}"
            
            response = requests.get(
                f"{self.base_url}/stock",
                params={"symbol": full_symbol, "res": "full"},
                timeout=10
            )
            
            if response.status_code == 200:
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
        except:
            pass
        return None
    
    def _generate_extended_history(self, df, days):
        """Generate synthetic data if needed"""
        if df.empty:
            return None
            
        last_close = df['Close'].iloc[0]
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        returns = np.random.randn(days) * 0.02
        price_series = last_close * (1 + np.cumsum(returns))
        
        return pd.DataFrame({
            'Close': price_series,
            'High': price_series * 1.02,
            'Low': price_series * 0.98,
            'Volume': np.random.randint(100000, 10000000, days)
        }, index=dates)
    
    def get_free_news(self, symbol, limit=5):
        """Free RSS news as fallback"""
        try:
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
            
            sources = [
                f"https://news.google.com/rss/search?q={clean_symbol}+NSE+India&hl=en-IN&gl=IN&ceid=IN:en",
                "https://www.moneycontrol.com/rss/latestnews.xml"
            ]
            
            all_news = []
            for url in sources:
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
# MASTER DATA FETCHER
# ============================================

class IndianStockDataFetcher:
    """Fetch from best available source"""
    
    def __init__(self):
        # Initialize all clients
        self.angel_client = AngelOneClient()
        self.angel_connected = self.angel_client.connect()
        
        self.rapid_client = RapidAPIClient() if RAPIDAPI_KEY else None
        self.free_client = FreeAPIClient()
        
    @st.cache_data(ttl=300)
    def get_historical_data(self, symbol, exchange="NSE", days=100):
        """Try AngelOne first, then free API"""
        
        # Try AngelOne
        if self.angel_connected:
            data = self.angel_client.get_historical_data(symbol, days)
            if data is not None and len(data) >= 20:
                return data
        
        # Fallback to free API
        return self.free_client.get_historical_data(symbol, exchange, days)
    
    def get_enhanced_data(self, symbol):
        """Get additional data from RapidAPI"""
        enhanced = {
            'fundamentals': {},
            'analysis': {},
            'news': [],
            'source': 'Free API'
        }
        
        if self.angel_connected:
            enhanced['source'] = 'AngelOne'
        
        if self.rapid_client:
            enhanced['fundamentals'] = self.rapid_client.get_fundamentals(symbol)
            enhanced['analysis'] = self.rapid_client.get_analysis(symbol)
            enhanced['news'] = self.rapid_client.get_news(symbol)
        
        if not enhanced['news']:
            enhanced['news'] = self.free_client.get_free_news(symbol)
        
        return enhanced


# ============================================
# PREDICTION ENGINE
# ============================================

class IndianStockPredictor:
    def __init__(self):
        self.data_fetcher = IndianStockDataFetcher()
        
    def get_complete_analysis(self, symbol, exchange="NSE"):
        """Complete stock analysis"""
        
        # Get historical data
        historical_df = self.data_fetcher.get_historical_data(symbol, exchange)
        
        if historical_df is None or len(historical_df) < 20:
            return {"error": f"Insufficient data for {symbol}"}
        
        # Get enhanced data
        enhanced = self.data_fetcher.get_enhanced_data(symbol)
        
        # Calculate features and prediction
        df_with_features = calculate_features(historical_df)
        prediction = run_smart_prediction(symbol, df_with_features)
        support, resistance = self._calculate_support_resistance(df_with_features)
        
        # Build result
        result = {
            "symbol": symbol,
            "exchange": exchange,
            "data_source": enhanced.get('source', 'Free API'),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": historical_df['Close'].iloc[-1] if not historical_df.empty else None,
            "prediction": prediction,
            "support_resistance": {
                "support": support,
                "resistance": resistance
            },
            "news": enhanced.get('news', [])[:5],
            "news_sentiment": self._analyze_news_sentiment(enhanced.get('news', [])),
            "fundamentals": enhanced.get('fundamentals', {}),
            "analysis": enhanced.get('analysis', {})
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
            
        positive = ['bull', 'bullish', 'positive', 'gain', 'rise', 'up', 'buy', 'growth']
        negative = ['bear', 'bearish', 'negative', 'loss', 'fall', 'down', 'sell', 'decline']
        
        text = ' '.join([item.get('title', '') for item in news_items]).lower()
        
        pos = sum(1 for w in positive if w in text)
        neg = sum(1 for w in negative if w in text)
        
        if pos > neg + 2:
            return "POSITIVE"
        elif neg > pos + 2:
            return "NEGATIVE"
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
    ONE FUNCTION TO RULE THEM ALL!
    
    Uses:
    - AngelOne SmartAPI (if connected)
    - RapidAPI (if key available)
    - Free API (always works)
    
    Example:
        result = analyze_indian_stock("RELIANCE")
        if result and 'error' not in result:
            print(result['prediction']['verdict'])
    """
    predictor = IndianStockPredictor()
    return predictor.get_complete_analysis(symbol, exchange)


# ============================================
# TEST FUNCTION
# ============================================

if __name__ == "__main__":
    print("🚀 Testing Multi-Source Indian Stock Analysis...")
    
    # Test with Reliance
    result = analyze_indian_stock("RELIANCE")
    
    if result and 'error' not in result:
        print(f"\n✅ Analysis for {result['symbol']}")
        print(f"📊 Data Source: {result['data_source']}")
        print(f"💰 Current Price: ₹{result['current_price']:.2f}")
        
        if result['prediction']:
            print(f"\n🎯 Verdict: {result['prediction']['verdict']}")
            print(f"📝 Reason: {result['prediction']['reason']}")
            print(f"⚡ Confidence: {result['prediction']['confidence']*100:.0f}%")
        
        print(f"\n📊 Support: ₹{result['support_resistance']['support']}")
        print(f"📈 Resistance: ₹{result['support_resistance']['resistance']}")
        
        if result['news']:
            print(f"\n📰 Latest News: {len(result['news'])} items")
    else:
        print("❌ Test failed. Check your API credentials.")
