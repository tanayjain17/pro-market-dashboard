#!/usr/bin/env python
"""Test ML engine module."""

import sys
import os
from pathlib import Path

# Add project root to Python path - MUST BE FIRST
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import pandas as pd
import numpy as np

# Now import from utils
from utils.ml_engine import (
    calculate_features,
    run_smart_prediction,
    analyze_indian_stock
)

class TestFeatureCalculation:
    """Test technical indicator calculations"""
    
    def test_calculate_features_with_valid_data(self):
        """Test feature calculation with sample data"""
        # Create sample data
        dates = pd.date_range(end='2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
            'Volume': np.random.randint(10000, 100000, 100)
        }, index=dates)
        
        result = calculate_features(df)
        
        assert result is not None, "Should return DataFrame"
        assert 'RSI' in result.columns, "Should have RSI"
        assert 'MACD' in result.columns, "Should have MACD"
        assert 'SMA_50' in result.columns, "Should have SMA_50"
    
    def test_calculate_features_insufficient_data(self):
        """Test with insufficient data (<50 rows)"""
        dates = pd.date_range(end='2024-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'Close': np.random.randn(30).cumsum() + 100,
            'High': np.random.randn(30).cumsum() + 102,
            'Low': np.random.randn(30).cumsum() + 98,
        }, index=dates)
        
        result = calculate_features(df)
        assert result is None, "Should return None for insufficient data"

class TestPrediction:
    """Test prediction logic"""
    
    @pytest.fixture
    def sample_df_with_features(self):
        """Create sample data with pre-calculated features"""
        dates = pd.date_range(end='2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
            'Volume': np.random.randint(10000, 100000, 100),
            'RSI': np.random.uniform(30, 70, 100),
            'MACD_Hist': np.random.randn(100) * 0.5,
            'SMA_50': np.random.randn(100).cumsum() + 100,
            'ATR': np.random.uniform(1, 5, 100),
            'Vol_Ratio': np.random.uniform(0.5, 2, 100)
        }, index=dates)
        return df
    
    @pytest.fixture
    def sample_raw_df(self):
        """Create sample raw data without features"""
        dates = pd.date_range(end='2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
        }, index=dates)
        return df
    
    def test_run_smart_prediction_with_features(self, sample_df_with_features):
        """Test prediction with pre-calculated features"""
        pred = run_smart_prediction("TEST", sample_df_with_features)
        
        assert pred is not None, "Should return prediction"
        assert 'verdict' in pred, "Should have verdict"
        assert 'confidence' in pred, "Should have confidence"
        assert 0 <= pred['confidence'] <= 1, "Confidence should be between 0-1"
        assert 'sl' in pred, "Should have stop loss"
        assert 'tgt' in pred, "Should have target"
        assert 'curr' in pred, "Should have current price"
    
    def test_run_smart_prediction_no_features(self, sample_raw_df):
        """Test prediction with raw data (should fail gracefully)"""
        pred = run_smart_prediction("TEST", sample_raw_df)
        assert pred is None, "Should return None for missing features"

class TestIntegration:
    """Test the main analysis function"""
    
    def test_analyze_indian_stock(self):
        """Test the main entry point"""
        result = analyze_indian_stock("RELIANCE", "NSE")
        
        # Should always return a dict, even on error
        assert isinstance(result, dict), "Should return a dictionary"
        
        if 'error' not in result:
            assert 'symbol' in result, "Should have symbol"
            assert 'current_price' in result, "Should have current price"
            assert 'prediction' in result, "Should have prediction"
            assert 'support_resistance' in result, "Should have S/R levels"
            assert 'news' in result, "Should have news"
            assert result['current_price'] > 0, "Price should be positive"
    
    def test_analyze_indian_stock_invalid(self):
        """Test with invalid symbol"""
        result = analyze_indian_stock("INVALID_SYMBOL", "NSE")
        
        # Should return error dict
        assert isinstance(result, dict), "Should return dictionary"
        assert 'error' in result, "Should have error message"

if __name__ == "__main__":
    # Quick manual test
    print("🧪 Testing ML engine manually...")
    result = analyze_indian_stock("RELIANCE")
    if result and 'error' not in result:
        print(f"✅ Analysis successful")
        print(f"📊 Symbol: {result['symbol']}")
        print(f"💰 Price: ₹{result['current_price']:.2f}")
        if result['prediction']:
            print(f"🎯 Verdict: {result['prediction']['verdict']}")
            print(f"📝 Reason: {result['prediction']['reason']}")
    else:
        print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
