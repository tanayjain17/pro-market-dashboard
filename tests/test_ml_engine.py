import pytest
import pandas as pd
import numpy as np
from utils.ml_engine import calculate_features, run_smart_prediction

class TestMLEngine:
    """Test machine learning predictions"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV data"""
        dates = pd.date_range('2024-01-01', periods=200)
        prices = np.cumsum(np.random.randn(200) * 2 + 0.5) + 100
        
        df = pd.DataFrame({
            'Open': prices - np.abs(np.random.randn(200)),
            'High': prices + np.abs(np.random.randn(200)),
            'Low': prices - np.abs(np.random.randn(200)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 200)
        }, index=dates)
        
        return df
    
    def test_calculate_features(self, sample_df):
        """Test feature calculation"""
        result = calculate_features(sample_df)
        
        assert result is not None
        assert 'SMA_20' in result.columns
        assert 'RSI' in result.columns
        assert 'MACD' in result.columns
        assert 'ATR' in result.columns
        
        # Check RSI range (0-100)
        assert (result['RSI'] >= 0).all() and (result['RSI'] <= 100).all()
    
    def test_prediction_output(self, sample_df):
        """Test prediction structure"""
        df = calculate_features(sample_df)
        result = run_smart_prediction("TEST.NS", df)
        
        if result:
            required_keys = ['verdict', 'color', 'confidence', 'sl', 'tgt', 'curr']
            for key in required_keys:
                assert key in result, f"Missing key: {key}"
            
            # Confidence should be 0-1
            assert 0 <= result['confidence'] <= 1
            # Current price should be positive
            assert result['curr'] > 0
