#!/usr/bin/env python3
"""
Simple test script to verify ML predictor works
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from analysis.ml_predictor import MLPredictor

def test_ml():
    print("Testing ML predictor...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(1000) * 0.1,
        'high': prices + np.abs(np.random.randn(1000) * 0.2),
        'low': prices - np.abs(np.random.randn(1000) * 0.2),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    print(f"Created sample data with {len(df)} rows")
    
    # Create sample technical indicators
    sample_indicators = {
        'rsi': pd.Series(np.random.rand(1000) * 100, index=dates),
        'macd_line': pd.Series(np.random.randn(1000), index=dates),
        'sma_20': pd.Series(prices + np.random.randn(1000) * 0.5, index=dates),
        'bb_upper': pd.Series(prices + np.random.randn(1000) * 2, index=dates),
        'bb_lower': pd.Series(prices - np.random.randn(1000) * 2, index=dates)
    }
    
    # Test ML predictor
    predictor = MLPredictor()
    
    # Prepare features and targets
    features = predictor.prepare_features(df, sample_indicators)
    targets = predictor.prepare_targets(df, prediction_horizon=1)
    
    print(f"Prepared features shape: {features.shape}")
    print(f"Prepared targets shape: {targets.shape}")
    
    # Train model
    if len(features) > 0 and len(targets) > 0:
        metrics = predictor.train(features, targets)
        print(f"Training metrics: {metrics}")
        
        # Make sample predictions
        latest_features = features.iloc[-10:]  # Last 10 samples
        predictions, probabilities = predictor.predict(latest_features)
        print(f"Sample predictions: {predictions}")
        print(f"Sample probabilities: {probabilities}")

if __name__ == "__main__":
    test_ml()