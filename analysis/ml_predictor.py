"""
Machine Learning Predictor Module
Implements ML models for financial price movement prediction based on technical indicators and wave patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MLPredictor:
    """
    A class to predict price movements using machine learning models.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ML predictor.
        
        Args:
            model_path: Path to a pre-trained model to load (optional)
        """
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        if model_path:
            self.load_model(model_path)
    
    def prepare_features(self, df: pd.DataFrame, technical_indicators: Dict[str, pd.Series], 
                        wave_features: Optional[Dict] = None) -> pd.DataFrame:
        """
        Prepare features for ML model from price data, technical indicators, and wave patterns.
        
        Args:
            df: DataFrame with price data ['open', 'high', 'low', 'close', 'volume']
            technical_indicators: Dictionary of computed technical indicators
            wave_features: Dictionary of wave pattern features (optional)
            
        Returns:
            DataFrame with prepared features
        """
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['close'] = df['close']
        features['high_low_pct'] = (df['high'] - df['low']) / df['close']
        features['open_close_pct'] = (df['close'] - df['open']) / df['open']
        features['volume'] = df['volume']
        
        # Technical indicator features
        for name, indicator in technical_indicators.items():
            # Only include numerical indicators, not the dictionary ones
            if isinstance(indicator, pd.Series) and np.issubdtype(indicator.dtype, np.number):
                features[name] = indicator.fillna(0)
        
        # Wave pattern features (if available)
        if wave_features:
            # Add wave-related features
            for key, value in wave_features.items():
                if isinstance(value, (int, float)):
                    features[f'wave_{key}'] = value
        
        # Lag features (previous values)
        for col in ['close', 'volume']:
            if col in features.columns:
                features[f'{col}_lag1'] = features[col].shift(1)
                features[f'{col}_lag2'] = features[col].shift(2)
                features[f'{col}_diff1'] = features[col].diff(1)
                features[f'{col}_diff2'] = features[col].diff(2)
        
        # Moving averages of features
        for col in ['close', 'volume']:
            if col in features.columns:
                features[f'{col}_ma5'] = features[col].rolling(window=5).mean()
                features[f'{col}_ma10'] = features[col].rolling(window=10).mean()
        
        # Clean data (remove NaN values)
        features = features.dropna()
        
        return features
    
    def prepare_targets(self, df: pd.DataFrame, prediction_horizon: int = 1) -> pd.Series:
        """
        Prepare target variables for classification (up/down movement).
        
        Args:
            df: DataFrame with price data
            prediction_horizon: Number of periods to look ahead for prediction
            
        Returns:
            Series with target values (1 for up, 0 for down/no change)
        """
        # Future price change
        future_price = df['close'].shift(-prediction_horizon)
        current_price = df['close']
        price_change = (future_price - current_price) / current_price
        
        # Classify as up (1) or down/no change (0)
        # Using a small threshold to account for transaction costs
        target = (price_change > 0.001).astype(int)
        
        # Remove last prediction_horizon rows (no future data)
        target = target.iloc[:-prediction_horizon]
        
        return target
    
    def train(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """
        Train the ML model.
        
        Args:
            features: DataFrame with feature values
            targets: Series with target values
            
        Returns:
            Dictionary with training metrics
        """
        if len(features) == 0 or len(targets) == 0:
            logger.warning("No data available for training")
            return {}
        
        # Ensure we have matching indices
        common_index = features.index.intersection(targets.index)
        if len(common_index) == 0:
            logger.warning("No matching indices between features and targets")
            return {}
            
        features = features.loc[common_index]
        targets = targets.loc[common_index]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42, stratify=targets
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Feature importance
        feature_importance = dict(zip(features.columns, self.model.feature_importances_))
        
        logger.info(f"Model trained. Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': feature_importance
        }
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained model.
        
        Args:
            features: DataFrame with feature values
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained yet")
            return np.array([]), np.array([])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        return predictions, probabilities
    
    def save_model(self, path: str):
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load the model from
        """
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")

class EnsemblePredictor:
    """
    Ensemble of multiple ML models for improved predictions.
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {}
    
    def add_model(self, name: str, model: MLPredictor, weight: float = 1.0):
        """
        Add a model to the ensemble.
        
        Args:
            name: Name of the model
            model: Trained MLPredictor instance
            weight: Weight for this model in ensemble predictions
        """
        self.models[name] = model
        self.weights[name] = weight
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions.
        
        Args:
            features: DataFrame with feature values
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.models:
            logger.warning("No models in ensemble")
            return np.array([]), np.array([])
        
        # Collect predictions from all models
        all_predictions = []
        all_probabilities = []
        weights = []
        
        for name, model in self.models.items():
            if model.is_trained:
                pred, prob = model.predict(features)
                if len(pred) > 0:
                    all_predictions.append(pred)
                    all_probabilities.append(prob[:, 1])  # Probability of class 1 (up movement)
                    weights.append(self.weights[name])
        
        if not all_predictions:
            logger.warning("No trained models available for prediction")
            return np.array([]), np.array([])
        
        # Ensemble prediction (weighted average)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Weighted average of probabilities
        ensemble_probs = np.average(all_probabilities, axis=0, weights=weights)
        
        # Convert to binary predictions (above 0.5 = up movement)
        ensemble_pred = (ensemble_probs > 0.5).astype(int)
        
        # Format probabilities as 2D array [prob_class_0, prob_class_1]
        ensemble_probabilities = np.column_stack([1 - ensemble_probs, ensemble_probs])
        
        return ensemble_pred, ensemble_probabilities

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    
    sample_df = pd.DataFrame({
        'open': prices + np.random.randn(1000) * 0.1,
        'high': prices + np.abs(np.random.randn(1000) * 0.2),
        'low': prices - np.abs(np.random.randn(1000) * 0.2),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Create sample technical indicators
    sample_indicators = {
        'rsi': pd.Series(np.random.rand(1000) * 100, index=dates),
        'macd_line': pd.Series(np.random.randn(1000), index=dates),
        'sma_20': pd.Series(prices + np.random.randn(1000) * 0.5, index=dates),
        'bb_upper': pd.Series(prices + np.random.randn(1000) * 2, index=dates),
        'bb_lower': pd.Series(prices - np.random.randn(1000) * 2, index=dates)
    }
    
    # Initialize predictor
    predictor = MLPredictor()
    
    # Prepare features and targets
    features = predictor.prepare_features(sample_df, sample_indicators)
    targets = predictor.prepare_targets(sample_df, prediction_horizon=1)
    
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