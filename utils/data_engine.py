import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=300)  # 5-minute cache
def robust_yf_download(ticker, period="2y"):
    """Download stock data with fallback logic"""
    try:
        interval = "1m" if period in ["1d", "5d"] else "1d"
        if period == "1mo":
            interval = "30m"
        
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Fallback if intraday is empty
        if df.empty and interval != "1d":
            logger.warning(f"Intraday data empty for {ticker}, retrying with daily")
            df = yf.download(ticker, period=period, interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        
        return df if not df.empty else None
    
    except requests.exceptions.Timeout:
        st.error("⏱️ Yahoo Finance timeout. Please try again.")
        return None
    except Exception as e:
        logger.error(f"Error downloading {ticker}: {str(e)}")
        st.error(f"❌ Data fetch failed: {str(e)}")
        return None

@st.cache_data(ttl=1800)  # 30-minute cache for fundamentals
def get_fundamentals(ticker):
    """Fetch fundamental data from Yahoo + Screener.in fallback"""
    try:
        info = yf.Ticker(ticker).info
        
        if info.get('trailingPE') is not None:
            return {
                "Market Cap": info.get("marketCap", 0),
                "P/E": info.get("trailingPE", 0),
                "P/B": info.get("priceToBook", 0),
                "ROE": (info.get("returnOnEquity", 0) or 0) * 100,
                "Book Value": info.get("bookValue", 0),
                "52W High": info.get("fiftyTwoWeekHigh", 0),
                "Dividend Yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                "Source": "Yahoo"
            }
    except Exception as e:
        logger.warning(f"Yahoo Finance failed for {ticker}: {e}")
    
    # Screener.in fallback for NSE stocks
    if ".NS" in ticker:
        try:
            clean_sym = ticker.replace(".NS", "")
            url = f"https://www.screener.in/company/{clean_sym}/"
            
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            ratios = soup.find('ul', {'id': 'top-ratios'})
            
            if ratios:
                data = {}
                for li in ratios.find_all('li'):
                    try:
                        name = li.find('span', {'class': 'name'}).text.strip()
                        val_text = li.find('span', {'class': 'number'}).text.replace(',', '')
                        val = float(val_text) if val_text else 0
                        
                        if "Market Cap" in name:
                            data["Market Cap"] = val * 10000000
                        elif "Stock P/E" in name:
                            data["P/E"] = val
                        elif "ROE" in name:
                            data["ROE"] = val
                        elif "Book Value" in name:
                            data["Book Value"] = val
                        elif "Dividend" in name:
                            data["Dividend Yield"] = val
                    except:
                        continue
                
                if data:
                    data["Source"] = "Screener.in"
                    return data
        except requests.exceptions.RequestException as e:
            logger.warning(f"Screener.in unavailable: {e}")
        except Exception as e:
            logger.error(f"Screener parsing error: {e}")
    
    return None

@st.cache_data(ttl=600)  # 10-minute cache
def get_news_sentiment(ticker):
    """Fetch news and calculate VADER sentiment"""
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        import nltk
        
        # Download VADER lexicon on first run
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        
        sia = SentimentIntensityAnalyzer()
        
        clean = ticker.replace(".NS", "").replace(".BO", "")
        url = f"https://news.google.com/rss/search?q={clean}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        
        feed = feedparser.parse(url)
        
        scores = []
        for entry in feed.entries[:5]:
            score = sia.polarity_scores(entry.title)['compound']
            scores.append(score)
        
        avg_sentiment = np.mean(scores) if scores else 0
        return avg_sentiment
    
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return 0

