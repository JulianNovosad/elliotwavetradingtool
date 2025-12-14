"""
Test suite for ML predictor module.
"""

import pytest
import pandas as pd
import numpy as np
from analysis.ml_predictor import MLPredictor, EnsemblePredictor
import warnings
warnings.filterwarnings('ignore')

def test_ml_predictor_initialization():
    """Test MLPredictor initialization."""
    # Test default initialization
    predictor = MLPredictor()
    assert predictor is not None, "MLPredictor should initialize successfully"
    assert predictor.model is None, "Model should be None initially"
    assert not predictor.is_trained, "Should not be trained initially"


def test_feature_preparation():
    """Test feature preparation."""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'open': [100 + i for i in range(10)],
        'high': [101 + i for i in range(10)],
        'low': [99 + i for i in range(10)],
        'close': [100 + i for i in range(10)],
        'volume': [1000 + i * 10 for i in range(10)]
    }, index=dates)
    
    # Create sample technical indicators
    indicators = {
        'rsi': pd.Series([30, 35, 40, 45, 50, 55, 60, 65, 70, 75], index=dates),
        'macd_line': pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], index=dates),
        'sma_20': pd.Series([100 + i * 0.5 for i in range(10)], index=dates)
    }
    
    # Prepare features
    predictor = MLPredictor()
    features = predictor.prepare_features(df, indicators)
    
    # Check that features are created
    assert not features.empty, "Features should not be empty"
    assert len(features) > 0, "Should have some feature rows"
    assert 'close' in features.columns, "Should have close price feature"
    
    # Check that lag features are created
    assert 'close_lag1' in features.columns, "Should have lag features"
    assert 'close_diff1' in features.columns, "Should have diff features"


def test_target_preparation():
    """Test target preparation."""
    # Create sample data with clear trend
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    }, index=dates)
    
    # Prepare targets
    predictor = MLPredictor()
    targets = predictor.prepare_targets(df, prediction_horizon=1)
    
    # Check that targets are created
    assert len(targets) > 0, "Should have some targets"
    assert set(targets.unique()).issubset({0, 1}), "Targets should be 0 or 1"


def test_training_and_prediction():
    """Test model training and prediction."""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'high': 101 + np.cumsum(np.random.randn(100) * 0.1),
        'low': 99 + np.cumsum(np.random.randn(100) * 0.1),
        'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'volume': 1000 + np.random.randint(-100, 100, 100)
    }, index=dates)
    
    # Create sample technical indicators
    indicators = {
        'rsi': pd.Series(np.random.rand(100) * 100, index=dates),
        'macd_line': pd.Series(np.random.randn(100), index=dates),
        'sma_20': pd.Series(100 + np.random.randn(100), index=dates)
    }
    
    # Prepare features and targets
    predictor = MLPredictor()
    features = predictor.prepare_features(df, indicators)
    targets = predictor.prepare_targets(df, prediction_horizon=1)
    
    # Train model
    metrics = predictor.train(features, targets)
    
    # Check training results
    assert 'train_accuracy' in metrics, "Should have train accuracy"
    assert 'test_accuracy' in metrics, "Should have test accuracy"
    assert predictor.is_trained, "Should be trained after training"
    
    # Test prediction
    if len(features) > 0:
        # Take last few samples for prediction
        latest_features = features.tail(5)
        predictions, probabilities = predictor.predict(latest_features)
        
        # Check predictions
        assert len(predictions) == len(latest_features), "Should have prediction for each sample"
        assert len(probabilities) == len(latest_features), "Should have probability for each sample"
        assert set(predictions).issubset({0, 1}), "Predictions should be 0 or 1"


def test_ensemble_predictor():
    """Test ensemble predictor."""
    # Create ensemble predictor
    ensemble = EnsemblePredictor()
    assert ensemble is not None, "EnsemblePredictor should initialize successfully"
    
    # Test adding models (even though they're not trained)
    predictor1 = MLPredictor()
    predictor2 = MLPredictor()
    
    ensemble.add_model('model1', predictor1, weight=1.0)
    ensemble.add_model('model2', predictor2, weight=0.5)
    
    # Check that models are added
    assert len(ensemble.models) == 2, "Should have 2 models"
    assert len(ensemble.weights) == 2, "Should have 2 weights"


if __name__ == "__main__":
    pytest.main([__file__])