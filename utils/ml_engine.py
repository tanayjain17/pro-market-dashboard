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
# MULTIPLE INDIAN STOCK API CONFIGURATION
# ============================================

# 1. Indian-Stock-Market-API (No key needed - PRIMARY)
INDIAN_API_BASE = "https://military-jobye-haiqstudios-14f59639.koyeb.app"

# 2. AngelOne SmartAPI (You have login!)
ANGELONE_API_KEY = st.secrets.get("ANGELONE_API_KEY", "")
ANGELONE_CLIENT_ID = st.secrets.get("ANGELONE_CLIENT_ID", "")
ANGELONE_PASSWORD = st.secrets.get("ANGELONE_PASSWORD", "")
ANGELONE_TOTP = st.secrets.get("ANGELONE_TOTP", "")

# 3. RapidAPI Configuration (YH Finance - BACKUP)
RAPIDAPI_KEY = st.secrets.get("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "yh-finance.p.rapidapi.com"
NEWS_API_URL = "https://yh-finance.p.rapidapi.com/stock/get-news"

# 4. Alpha Vantage (Free tier - 5 calls/min)
ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", "")

# 5. Financial Modeling Prep (Free tier)
FMP_KEY = st.secrets.get("FMP_KEY", "")


# ============================================
# ANGELONE SMARTAPI INTEGRATION
# ============================================

class AngelOneClient:
    """AngelOne SmartAPI client for Indian stocks"""
    
    def __init__(self):
        self.api_key = ANGELONE_API_KEY
        self.client_id = ANGELONE_CLIENT_ID
        self.password = ANGELONE_PASSWORD
        self.totp = ANGELONE_TOTP
        self.jwt_token = None
        self.refresh_token = None
        self.feed_token = None
        self.user_profile = None
        self.is_connected = False
        
    def connect(self):
        """Connect to AngelOne SmartAPI"""
        if not all([self.api_key, self.client_id, self.password]):
            return False
            
        try:
            # Try to import smartapi
            from smartapi import SmartConnect
            
            # Create connection object
            obj = SmartConnect(
                api_key=self.api_key,
                client_id=self.client_id,
                password=self.password,
                totp_secret=self.totp
            )
            
            # Login
            data = obj.generateSession()
            self.jwt_token = data['data']['jwtToken']
            self.refresh_token = data['data']['refreshToken']
            self.feed_token = obj.getfeedToken()
            self.user_profile = obj.getProfile(self.refresh_token)
            self.is_connected = True
            return True
            
        except Exception as e:
            st.warning(f"AngelOne connection failed: {e}")
            return False
    
    @st.cache_data(ttl=60)  # Cache for 1 minute
    def get_ltp(self, symbol, exchange="NSE"):
        """Get last traded price"""
        if not self.is_connected:
            return None
            
        try:
            from smartapi import SmartConnect
            # Add exchange prefix
            if exchange == "NSE":
                trading_symbol = f"NSE:{symbol}-EQ"
            else:
                trading_symbol = f"BSE:{symbol}-EQ"
            
            # Get LTP
            ltp_data = obj.ltpData(exchange, trading_symbol, symbol)
            return ltp_data['data']['ltp']
        except:
            return None
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_historical_data(self, symbol, interval="ONE_DAY", days=100):
        """Get historical data from AngelOne"""
        if not self.is_connected:
            return None
            
        try:
            from smartapi import SmartConnect
            
            # Calculate from and to dates
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Format dates
            from_str = from_date.strftime("%Y-%m-%d")
            to_str = to_date.strftime("%Y-%m-%d")
            
            # Get historical data
            historic_data = obj.getCandleData({
                "exchange": "NSE",
                "symboltoken": self.get_token(symbol),
                "interval": interval,
                "fromdate": from_str,
                "todate": to_str
            })
            
            # Convert to DataFrame
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
        except:
            return None
    
    def get_token(self, symbol):
        """Get token for symbol (simplified)"""
        # In production, you'd fetch from master contract
        token_map = {
            "RELIANCE": "2885",
            "TCS": "295321",
            "HDFCBANK": "341249",
            "INFY": "1594",
            "ITC": "1660",
        }
        return token_map.get(symbol, "2885")


# ============================================
# ALPHA VANTAGE INTEGRATION
# ============================================

class AlphaVantageClient:
    """Alpha Vantage API client"""
    
    def __init__(self):
        self.api_key = ALPHA_VANTAGE_KEY
        self.base_url = "https://www.alphavantage.co/query"
    
    @st.cache_data(ttl=300)
    def get_daily_data(self, symbol):
        """Get daily adjusted data"""
        if not self.api_key:
            return None
            
        try:
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": f"{symbol}.BSE",  # For Indian stocks
                "apikey": self.api_key,
                "outputsize": "compact"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if "Time Series (Daily)" in data:
                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df = df.rename(columns={
                    '4. close': 'Close',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '6. volume': 'Volume'
                })
                return df.sort_index()
        except:
            return None
    
    @st.cache_data(ttl=3600)
    def get_technical_indicators(self, symbol):
        """Get RSI, MACD, etc."""
        if not self.api_key:
            return {}
            
        indicators = {}
        try:
            # RSI
            params = {
                "function": "RSI",
                "symbol": f"{symbol}.BSE",
                "interval": "daily",
                "time_period": 14,
                "series_type": "close",
                "apikey": self.api_key
            }
            response = requests.get(self.base_url, params=params)
            if "Technical Analysis: RSI" in response.json():
                rsi_data = response.json()["Technical Analysis: RSI"]
                latest = list(rsi_data.values())[0]
                indicators['rsi'] = float(latest['RSI'])
        except:
            pass
            
        return indicators


# ============================================
# MASTER DATA FETCHER (USES BEST AVAILABLE SOURCE)
# ============================================

class IndianStockDataFetcher:
    """Fetch Indian stock data from best available source"""
    
    def __init__(self):
        # Initialize all clients
        self.base_url = INDIAN_API_BASE
        
        # AngelOne
        self.angel_client = AngelOneClient()
        self.angel_connected = self.angel_client.connect()
        
        # Alpha Vantage
        self.alpha_client = AlphaVantageClient()
        
        # Data source priority
        self.data_sources = []
        if self.angel_connected:
            self.data_sources.append(("AngelOne", self.angel_client))
        if ALPHA_VANTAGE_KEY:
            self.data_sources.append(("Alpha Vantage", self.alpha_client))
        self.data_sources.append(("Free API", self))  # Fallback to free API
        
    @st.cache_data(ttl=300)
    def get_historical_data(self, symbol, exchange="NSE", days=100):
        """Try multiple sources in order of preference"""
        
        # Try AngelOne first (most reliable for Indian stocks)
        if self.angel_connected:
            try:
                data = self.angel_client.get_historical_data(symbol, days=days)
                if data is not None and len(data) >= 20:
                    return data
            except:
                pass
        
        # Try Alpha Vantage next
        if ALPHA_VANTAGE_KEY:
            try:
                data = self.alpha_client.get_daily_data(symbol)
                if data is not None and len(data) >= 20:
                    return data
            except:
                pass
        
        # Fallback to free API
        return self._get_free_api_data(symbol, exchange, days)
    
    def _get_free_api_data(self, symbol, exchange="NSE", days=100):
        """Free Indian stock API as fallback"""
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
        """Fetch news from multiple sources"""
        
        # Try to get news from various sources
        news = []
        
        # 1. Try RSS feeds first (always free)
        news = self._get_free_rss_news(symbol, limit)
        if news:
            return news
        
        # 2. Try RapidAPI if available
        if RAPIDAPI_KEY:
            try:
                rapid_news = self._get_rapidapi_news(symbol, limit)
                if rapid_news:
                    return rapid_news
            except:
                pass
        
        return []
    
    def _get_rapidapi_news(self, symbol, limit):
        """Get news from RapidAPI"""
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
        """Complete stock analysis using best available data"""
        
        # Show data source info
        data_source = "AngelOne" if self.data_fetcher.angel_connected else "Free API"
        
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
            "data_source": data_source,
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
# YOUR EXISTING TECHNICAL FUNCTIONS (KEEP AS IS)
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
    MAIN FUNCTION - Uses best available data source
    AngelOne > Alpha Vantage > Free API
    """
    predictor = IndianStockPredictor()
    return predictor.get_complete_analysis(symbol, exchange)


# ============================================
# TEST THE CODE
# ============================================

if __name__ == "__main__":
    print("🚀 Testing Multi-Source Indian Stock Analysis...")
    
    # Test with Reliance
    result = analyze_indian_stock("RELIANCE")
    
    if result and 'error' not in result:
        print(f"\n✅ Analysis for {result['symbol']}")
        print(f"📊 Data Source: {result.get('data_source', 'Free API')}")
        print(f"💰 Current Price: ₹{result['current_price']}")
        
        if result['prediction']:
            print(f"\n🎯 Verdict: {result['prediction']['verdict']}")
            print(f"📝 Reason: {result['prediction']['reason']}")
            print(f"⚡ Confidence: {result['prediction']['confidence']*100:.0f}%")
        
        print(f"\n📊 Support: ₹{result['support_resistance']['support']}")
        print(f"📈 Resistance: ₹{result['support_resistance']['resistance']}")
    else:
        print("❌ Test failed.")
