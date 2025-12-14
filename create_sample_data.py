#!/usr/bin/env python3
"""
Create sample CSV data files for testing
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_sample_data():
    # Create sample data for different intervals
    intervals = ['1m', '5m', '15m', '30m', '1h', '1d']
    
    for interval in intervals:
        # Create timestamp range
        if interval == '1m':
            dates = pd.date_range('2023-01-01', periods=1000, freq='1min')
        elif interval == '5m':
            dates = pd.date_range('2023-01-01', periods=200, freq='5min')
        elif interval == '15m':
            dates = pd.date_range('2023-01-01', periods=100, freq='15min')
        elif interval == '30m':
            dates = pd.date_range('2023-01-01', periods=50, freq='30min')
        elif interval == '1h':
            dates = pd.date_range('2023-01-01', periods=24, freq='1H')
        elif interval == '1d':
            dates = pd.date_range('2023-01-01', periods=30, freq='1D')
        
        # Generate price data
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        # Save to CSV
        filename = f"data/sample_{interval}.csv"
        df.to_csv(filename, index=False)
        print(f"Created {filename} with {len(df)} rows")

if __name__ == "__main__":
    create_sample_data()