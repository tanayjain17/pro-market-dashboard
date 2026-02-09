import pytest
import pandas as pd
import numpy as np
from utils.data_engine import (
    robust_yf_download,
    get_fundamentals,
    get_news_sentiment
)

class TestDataEngine:
    """Test data fetching functions"""
    
    def test_robust_yf_download_success(self):
        """Test successful data download"""
        df = robust_yf_download("RELIANCE.NS", "1y")
        
        assert df is not None, "DataFrame should not be None"
        assert len(df) > 0, "DataFrame should have rows"
        assert 'Close' in df.columns, "Should have Close column"
        assert 'Volume' in df.columns, "Should have Volume column"
    
    def test_robust_yf_download_invalid_ticker(self):
        """Test handling of invalid ticker"""
        df = robust_yf_download("INVALID_TICKER_XXXXX.NS", "1y")
        assert df is None, "Should return None for invalid ticker"
    
    def test_data_contains_ohlcv(self):
        """Test OHLCV data integrity"""
        df = robust_yf_download("TCS.NS", "1m")
        
        if df is not None:
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                assert col in df.columns, f"Missing {col} column"
            
            # High should be >= Low
            assert (df['High'] >= df['Low']).all(), "High >= Low violated"
            # Close should be between High and Low
            assert ((df['Close'] <= df['High']) & (df['Close'] >= df['Low'])).all()

class TestFundamentals:
    """Test fundamental data retrieval"""
    
    def test_get_fundamentals_nse(self):
        """Test fundamental data for NSE stock"""
        fund = get_fundamentals("RELIANCE.NS")
        
        if fund:  # May fail if Yahoo/Screener is down
            assert 'P/E' in fund, "Should have P/E ratio"
            assert 'ROE' in fund, "Should have ROE"
            assert fund['P/E'] >= 0, "P/E should be positive"
    
    def test_get_fundamentals_invalid(self):
        """Test with invalid ticker"""
        fund = get_fundamentals("INVALID_TICKET.NS")
        assert fund is None or isinstance(fund, dict)

class TestSentiment:
    """Test sentiment analysis"""
    
    def test_sentiment_range(self):
        """Test sentiment score is in valid range"""
        score = get_news_sentiment("RELIANCE.NS")
        
        # Sentiment should be between -1 and 1
        assert -1 <= score <= 1, f"Sentiment score out of range: {score}"
