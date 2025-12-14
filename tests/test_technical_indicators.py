"""
Test suite for technical indicators module.
"""

import pytest
import pandas as pd
import numpy as np
from analysis.indicators import TechnicalIndicators
import warnings
warnings.filterwarnings('ignore')

def test_rsi_calculation():
    """Test RSI calculation with known values."""
    # Create sample data with clear upward and downward trends
    prices = pd.Series([10, 11, 12, 13, 14, 13, 12, 11, 10, 9])
    
    # Calculate RSI
    ti = TechnicalIndicators()
    rsi = ti.calculate_rsi(prices, period=5)
    
    # Check that RSI is calculated (not all NaN)
    assert not rsi.isna().all(), "RSI should not be all NaN"
    
    # Check that RSI values are within valid range (0-100)
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all(), "RSI values should be between 0 and 100"


def test_macd_calculation():
    """Test MACD calculation with known values."""
    # Create sample data with trend
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    
    # Calculate MACD
    ti = TechnicalIndicators()
    macd_line, signal_line, histogram = ti.calculate_macd(prices)
    
    # Check that all components are calculated
    assert len(macd_line) == len(prices), "MACD line should have same length as input"
    assert len(signal_line) == len(prices), "Signal line should have same length as input"
    assert len(histogram) == len(prices), "Histogram should have same length as input"
    
    # Check that values are numeric
    assert np.issubdtype(macd_line.dtype, np.number), "MACD line should be numeric"
    assert np.issubdtype(signal_line.dtype, np.number), "Signal line should be numeric"
    assert np.issubdtype(histogram.dtype, np.number), "Histogram should be numeric"


def test_sma_calculation():
    """Test SMA calculation with known values."""
    # Create sample data
    prices = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Calculate SMA
    ti = TechnicalIndicators()
    sma = ti.calculate_sma(prices, period=3)
    
    # Check that SMA is calculated correctly for a few points
    # SMA at index 2 should be (1+2+3)/3 = 2
    assert sma.iloc[2] == 2.0, "SMA should be calculated correctly"
    
    # SMA at index 3 should be (2+3+4)/3 = 3
    assert sma.iloc[3] == 3.0, "SMA should be calculated correctly"
    
    # First two values should be NaN (not enough data)
    assert pd.isna(sma.iloc[0]), "First SMA value should be NaN"
    assert pd.isna(sma.iloc[1]), "Second SMA value should be NaN"


def test_ema_calculation():
    """Test EMA calculation."""
    # Create sample data
    prices = pd.Series([10, 11, 12, 13, 14, 15])
    
    # Calculate EMA
    ti = TechnicalIndicators()
    ema = ti.calculate_ema(prices, period=3)
    
    # Check that EMA is calculated (not all NaN)
    assert not ema.isna().all(), "EMA should not be all NaN"
    
    # Check that values are numeric
    assert np.issubdtype(ema.dtype, np.number), "EMA should be numeric"


def test_bollinger_bands_calculation():
    """Test Bollinger Bands calculation."""
    # Create sample data
    prices = pd.Series([10, 11, 12, 13, 14, 15, 14, 13, 12, 11])
    
    # Calculate Bollinger Bands
    ti = TechnicalIndicators()
    upper_band, middle_band, lower_band = ti.calculate_bollinger_bands(prices)
    
    # Check that all bands are calculated
    assert len(upper_band) == len(prices), "Upper band should have same length as input"
    assert len(middle_band) == len(prices), "Middle band should have same length as input"
    assert len(lower_band) == len(prices), "Lower band should have same length as input"
    
    # Check that values are numeric
    assert np.issubdtype(upper_band.dtype, np.number), "Upper band should be numeric"
    assert np.issubdtype(middle_band.dtype, np.number), "Middle band should be numeric"
    assert np.issubdtype(lower_band.dtype, np.number), "Lower band should be numeric"


def test_atr_calculation():
    """Test ATR calculation."""
    # Create sample data
    high = pd.Series([10.5, 11.5, 12.5, 13.5, 14.5])
    low = pd.Series([9.5, 10.5, 11.5, 12.5, 13.5])
    close = pd.Series([10, 11, 12, 13, 14])
    
    # Calculate ATR
    ti = TechnicalIndicators()
    atr = ti.calculate_atr(high, low, close, period=3)
    
    # Check that ATR is calculated (not all NaN)
    assert not atr.isna().all(), "ATR should not be all NaN"
    
    # Check that values are numeric
    assert np.issubdtype(atr.dtype, np.number), "ATR should be numeric"


def test_obv_calculation():
    """Test OBV calculation."""
    # Create sample data
    close = pd.Series([10, 11, 12, 11, 10])
    volume = pd.Series([100, 150, 200, 120, 80])
    
    # Calculate OBV
    ti = TechnicalIndicators()
    obv = ti.calculate_obv(close, volume)
    
    # Check that OBV is calculated
    assert len(obv) == len(close), "OBV should have same length as input"
    
    # Check that values are numeric
    assert np.issubdtype(obv.dtype, np.number), "OBV should be numeric"


def test_calculate_all_indicators():
    """Test calculation of all indicators."""
    # Create sample DataFrame with more data points for indicators that need longer periods
    dates = pd.date_range('2023-01-01', periods=60, freq='D')
    df = pd.DataFrame({
        'open': [100 + i for i in range(60)],
        'high': [101 + i for i in range(60)],
        'low': [99 + i for i in range(60)],
        'close': [100 + i for i in range(60)],
        'volume': [1000 + i * 10 for i in range(60)]
    }, index=dates)
    
    # Calculate all indicators
    ti = TechnicalIndicators()
    indicators = ti.calculate_all_indicators(df)
    
    # Check that indicators dict is not empty
    assert len(indicators) > 0, "Should calculate at least some indicators"
    
    # Check that all expected indicators are present
    expected_indicators = ['rsi', 'macd_line', 'macd_signal', 'macd_histogram', 
                          'sma_20', 'sma_50', 'ema_12', 'ema_26', 
                          'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'obv']
    
    for indicator in expected_indicators:
        assert indicator in indicators, f"Indicator {indicator} should be present"
        # Check that indicator values are not all NaN (at least some should have values)
        if hasattr(indicators[indicator], 'isna'):
            nan_count = indicators[indicator].isna().sum()
            total_count = len(indicators[indicator])
            # Allow some NaN values at the beginning (due to rolling windows) but not all
            assert nan_count < total_count, f"Indicator {indicator} should not be all NaN"


if __name__ == "__main__":
    pytest.main([__file__])