#!/usr/bin/env python3
"""
Simple test script to verify technical indicators work
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from analysis.indicators import TechnicalIndicators

def test_indicators():
    print("Testing technical indicators...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(100) * 0.1,
        'high': prices + np.abs(np.random.randn(100) * 0.2),
        'low': prices - np.abs(np.random.randn(100) * 0.2),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    print(f"Created sample data with {len(df)} rows")
    
    # Test indicators
    ti = TechnicalIndicators()
    indicators = ti.calculate_all_indicators(df)
    
    print(f"Calculated {len(indicators)} indicators")
    
    # Show some results
    for name, data in indicators.items():
        if hasattr(data, 'iloc'):
            print(f"{name}: {data.iloc[-1] if len(data) > 0 else 'N/A'}")
        else:
            print(f"{name}: {data}")

if __name__ == "__main__":
    test_indicators()