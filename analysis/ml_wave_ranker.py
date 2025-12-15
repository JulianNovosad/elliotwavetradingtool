"""
ML Wave Ranker
Uses machine learning models to rank admissible wave hypotheses based on probability,
market features, and historical patterns.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import logging
import datetime
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from .wave_hypothesis_engine import WaveHypothesis

logger = logging.getLogger(__name__)

class MLWaveRanker:
    """
    Ranks admissible wave hypotheses using machine learning models.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
        if model_path:
            self.load_model(model_path)
        else:
            # Initialize a default model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            
        logger.info("MLWaveRanker initialized")
        
    def extract_features(self, hypothesis: WaveHypothesis, market_data: pd.DataFrame, 
                        technical_indicators: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Extract features from a wave hypothesis and market data for ML ranking.
        
        Args:
            hypothesis: WaveHypothesis to extract features from
            market_data: DataFrame with market price data
            technical_indicators: Dictionary of computed technical indicators
            
        Returns:
            Dictionary of feature names and values
        """
        features = {}
        
        # Wave count features
        wave_pattern = hypothesis.wave_count.get('wave_pattern', {})
        features['wave_count_label'] = len(wave_pattern.get('label', '').split('-')) if wave_pattern.get('label') else 0
        features['wave_count_segments'] = len(wave_pattern.get('segments', []))
        features['wave_count_rank'] = hypothesis.wave_count.get('rank', 0)
        features['wave_count_confidence'] = hypothesis.wave_count.get('rule_compliance_score', 0.0)
        
        # Hypothesis features
        features['hypothesis_confidence'] = hypothesis.confidence_score
        features['hypothesis_age_seconds'] = (
            datetime.datetime.now(datetime.timezone.utc) - hypothesis.created_at
        ).total_seconds()
        features['hypothesis_violations'] = len(hypothesis.rule_violations)
        
        # Market data features (using latest values)
        if not market_data.empty:
            latest_data = market_data.iloc[-1]
            features['latest_price'] = latest_data.get('price', latest_data.get('close', 0))
            features['price_change_1h'] = self._calculate_price_change(market_data, '1H')
            features['price_change_4h'] = self._calculate_price_change(market_data, '4H')
            features['price_change_24h'] = self._calculate_price_change(market_data, '24H')
            features['volatility_1h'] = self._calculate_volatility(market_data, '1H')
            features['volatility_24h'] = self._calculate_volatility(market_data, '24H')
            
        # Technical indicator features
        for indicator_name, indicator_series in technical_indicators.items():
            if not indicator_series.empty:
                # Use the latest value of each indicator
                features[f'{indicator_name}_latest'] = indicator_series.iloc[-1]
                features[f'{indicator_name}_change_1h'] = self._calculate_indicator_change(indicator_series, '1H')
                
        # Wave segment features (if available)
        if hypothesis.segments:
            # Calculate average segment duration
            if len(hypothesis.segments) > 1:
                durations = []
                for seg in hypothesis.segments:
                    if 'start' in seg and 'end' in seg:
                        try:
                            duration = seg['end'] - seg['start']
                            durations.append(duration.total_seconds())
                        except:
                            pass
                if durations:
                    features['avg_segment_duration'] = np.mean(durations)
                    features['segment_duration_std'] = np.std(durations)
                    
            # Count of different wave types
            impulse_count = sum(1 for seg in hypothesis.segments if seg.get('type') == 'impulse')
            corrective_count = sum(1 for seg in hypothesis.segments if seg.get('type') == 'corrective')
            features['impulse_segments'] = impulse_count
            features['corrective_segments'] = corrective_count
            
        return features
        
    def _calculate_price_change(self, df: pd.DataFrame, period: str) -> float:
        """Calculate price change over a given period."""
        if df.empty:
            return 0.0
            
        try:
            # Resample to the specified period and get the last value
            resampled = df.resample(period).last()
            if len(resampled) >= 2:
                return (resampled['price'].iloc[-1] / resampled['price'].iloc[-2]) - 1.0
            else:
                return 0.0
        except:
            return 0.0
            
    def _calculate_volatility(self, df: pd.DataFrame, period: str) -> float:
        """Calculate price volatility over a given period."""
        if df.empty:
            return 0.0
            
        try:
            # Resample to the specified period and get the last value
            resampled = df.resample(period).last()
            if len(resampled) >= 2:
                returns = resampled['price'].pct_change().dropna()
                return np.std(returns) * np.sqrt(len(returns))
            else:
                return 0.0
        except:
            return 0.0
            
    def _calculate_indicator_change(self, series: pd.Series, period: str) -> float:
        """Calculate change in indicator over a given period."""
        if series.empty:
            return 0.0
            
        try:
            # Resample to the specified period and get the last value
            resampled = series.resample(period).last()
            if len(resampled) >= 2:
                return (resampled.iloc[-1] / resampled.iloc[-2]) - 1.0
            else:
                return 0.0
        except:
            return 0.0
            
    def prepare_feature_matrix(self, hypotheses: List[WaveHypothesis], 
                              market_data: pd.DataFrame,
                              technical_indicators: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Prepare feature matrix for ML model from multiple hypotheses.
        
        Args:
            hypotheses: List of WaveHypothesis objects
            market_data: DataFrame with market price data
            technical_indicators: Dictionary of computed technical indicators
            
        Returns:
            DataFrame with features for all hypotheses
        """
        if not hypotheses:
            return pd.DataFrame()
            
        # Extract features for each hypothesis
        feature_dicts = []
        for hypothesis in hypotheses:
            features = self.extract_features(hypothesis, market_data, technical_indicators)
            features['hypothesis_id'] = hypothesis.id
            feature_dicts.append(features)
            
        # Create DataFrame
        feature_df = pd.DataFrame(feature_dicts)
        
        # Store feature names for later use
        if not feature_df.empty:
            self.feature_names = [col for col in feature_df.columns if col != 'hypothesis_id']
            
        return feature_df
        
    def rank_hypotheses(self, hypotheses: List[WaveHypothesis], 
                       market_data: pd.DataFrame,
                       technical_indicators: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Rank hypotheses using the ML model.
        
        Args:
            hypotheses: List of WaveHypothesis objects to rank
            market_data: DataFrame with market price data
            technical_indicators: Dictionary of computed technical indicators
            
        Returns:
            Dictionary mapping hypothesis IDs to ranking scores (higher is better)
        """
        if not hypotheses:
            return {}
            
        # Prepare features
        feature_df = self.prepare_feature_matrix(hypotheses, market_data, technical_indicators)
        
        if feature_df.empty:
            # Return equal scores if no features
            return {h.id: 0.5 for h in hypotheses}
            
        # Separate hypothesis IDs and features
        hypothesis_ids = feature_df['hypothesis_id'].tolist()
        feature_columns = [col for col in feature_df.columns if col != 'hypothesis_id']
        X = feature_df[feature_columns]
        
        # Handle missing values
        X = X.fillna(0.0)
        
        # Scale features
        if not self.is_trained:
            # For untrained model, just return random scores
            logger.warning("ML model not trained, returning random scores")
            return {hid: np.random.random() for hid in hypothesis_ids}
            
        # Scale features using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Get prediction probabilities (use probability of positive class)
        try:
            probabilities = self.model.predict_proba(X_scaled)
            # If binary classification, take probability of positive class
            if probabilities.shape[1] == 2:
                scores = probabilities[:, 1]
            else:
                # For multi-class, take the max probability
                scores = np.max(probabilities, axis=1)
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            # Fall back to random scores
            scores = np.random.random(len(hypothesis_ids))
            
        # Create mapping from hypothesis ID to score
        return dict(zip(hypothesis_ids, scores))
        
    def train_model(self, training_data: pd.DataFrame, labels: pd.Series):
        """
        Train the ML model on historical data.
        
        Args:
            training_data: DataFrame with features
            labels: Series with labels (1 for good hypotheses, 0 for bad)
        """
        if training_data.empty or labels.empty:
            logger.warning("Empty training data, skipping training")
            return
            
        # Separate features (exclude hypothesis_id if present)
        feature_columns = [col for col in training_data.columns if col != 'hypothesis_id']
        X = training_data[feature_columns]
        y = labels
        
        # Handle missing values
        X = X.fillna(0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        self.feature_names = feature_columns
        
        logger.info(f"ML model trained on {len(X)} samples")
        
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load model from
        """
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.feature_names = model_data.get('feature_names', [])
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {e}")
            # Reinitialize default model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            self.is_trained = False
            self.feature_names = []