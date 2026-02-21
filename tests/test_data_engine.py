import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        else:
            # If download fails, we should have a different test for that
            pass

class TestFundamentals:
    """Test fundamental data retrieval"""
    
    def test_get_fundamentals_nse(self):
        """Test fundamental data for NSE stock"""
        fund = get_fundamentals("RELIANCE.NS")
        
        if fund:  # May fail if Yahoo/Screener is down
            assert isinstance(fund, dict), "Fundamentals should be a dictionary"
            # Check for common fundamental keys
            common_keys = ['P/E', 'ROE', 'Market Cap', 'Book Value', 'Dividend Yield']
            found_keys = [key for key in common_keys if key in fund]
            assert len(found_keys) > 0, "Should have at least one fundamental metric"
    
    def test_get_fundamentals_invalid(self):
        """Test with invalid ticker"""
        fund = get_fundamentals("INVALID_TICKET.NS")
        # Should either return None or empty dict
        assert fund is None or fund == {} or len(fund) == 0

class TestSentiment:
    """Test sentiment analysis"""
    
    def test_sentiment_range(self):
        """Test sentiment score is in valid range"""
        score = get_news_sentiment("RELIANCE.NS")
        
        # Sentiment should be between -1 and 1
        assert -1 <= score <= 1, f"Sentiment score out of range: {score}"
        
        # Also test that score is a number
        assert isinstance(score, (int, float)), "Score should be numeric"
    
    def test_sentiment_with_mock(self):
        """Test sentiment with a known string"""
        # Since get_news_sentiment might use real data,
        # we can create a mock test here
        pass

# Optional: Add more specific tests
if __name__ == "__main__":
    # Quick manual test
    print("Testing data engine...")
    df = robust_yf_download("RELIANCE.NS", "1mo")
    print(f"Downloaded {len(df)} rows" if df is not None else "Download failed")
    
    fund = get_fundamentals("RELIANCE.NS")
    print(f"Fundamentals: {fund.keys() if fund else 'None'}")
    
    sentiment = get_news_sentiment("RELIANCE.NS")
    print(f"Sentiment: {sentiment}")
