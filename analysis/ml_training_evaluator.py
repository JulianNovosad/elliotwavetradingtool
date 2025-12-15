"""
ML Training and Evaluation System
Provides enhanced training and evaluation capabilities for the ML wave ranker.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import logging
import datetime
import json
import os
from pathlib import Path
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import joblib

from .ml_wave_ranker import MLWaveRanker
from .data_storage import DataStorage
from .wave_hypothesis_engine import WaveHypothesis

logger = logging.getLogger(__name__)

class MLTrainingEvaluator:
    """
    Enhanced ML training and evaluation system for the wave ranker.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_storage = DataStorage(config.get('database_path', './data/elliott_wave_data.db'))
        self.ml_ranker = MLWaveRanker()
        self.models_dir = Path("./models")
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info("MLTrainingEvaluator initialized")
        
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from historical hypotheses and trade outcomes.
        
        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        logger.info("Preparing training data")
        
        # Get historical hypotheses from database
        active_hypotheses = self.data_storage.get_active_hypotheses()
        
        # For demonstration, we'll create synthetic labels based on confidence scores
        # In a real system, labels would come from actual trade outcomes
        features_list = []
        labels_list = []
        
        for hypothesis_data in active_hypotheses:
            # Create a temporary hypothesis object for feature extraction
            # In practice, we'd need to reconstruct the full object with segments
            hypothesis = WaveHypothesis(
                wave_count=hypothesis_data.get('wave_count', {}),
                segments=[],  # Would need actual segments in practice
                id=hypothesis_data['id']
            )
            
            # Set attributes from stored data
            hypothesis.confidence_score = hypothesis_data.get('confidence_score', 0.0)
            hypothesis.ml_ranking_score = hypothesis_data.get('ml_ranking_score', 0.0)
            hypothesis.is_valid = hypothesis_data.get('is_valid', True)
            hypothesis.rule_violations = hypothesis_data.get('rule_violations', [])
            
            # Extract features (placeholder - would need actual market data)
            features = {
                'confidence_score': hypothesis.confidence_score,
                'wave_count_rank': hypothesis.wave_count.get('rank', 0),
                'wave_segments': len(hypothesis.wave_count.get('wave_pattern', {}).get('segments', [])),
                'rule_violations': len(hypothesis.rule_violations),
                'hypothesis_age_hours': (
                    datetime.datetime.now(datetime.timezone.utc) - 
                    datetime.datetime.fromisoformat(hypothesis_data['created_at'].replace('Z', '+00:00'))
                ).total_seconds() / 3600
            }
            
            features_list.append(features)
            
            # Create synthetic labels based on confidence and validity
            # In practice, this would be based on actual trade profitability
            label = 1 if (hypothesis.is_valid and hypothesis.confidence_score > 0.7) else 0
            labels_list.append(label)
            
        if not features_list:
            logger.warning("No training data available")
            return pd.DataFrame(), pd.Series(dtype=int)
            
        # Create DataFrames
        features_df = pd.DataFrame(features_list)
        labels_series = pd.Series(labels_list, name='label')
        
        logger.info(f"Prepared {len(features_df)} training samples")
        return features_df, labels_series
    
    def train_model(self, features_df: pd.DataFrame, labels_series: pd.Series, 
                   hyperparameter_search: bool = False) -> Dict:
        """
        Train the ML model with optional hyperparameter search.
        
        Args:
            features_df: DataFrame with features
            labels_series: Series with labels
            hyperparameter_search: Whether to perform hyperparameter search
            
        Returns:
            Dictionary with training results
        """
        if features_df.empty or labels_series.empty:
            logger.warning("Empty training data, skipping training")
            return {'success': False, 'message': 'Empty training data'}
            
        logger.info(f"Training model on {len(features_df)} samples")
        
        # Handle missing values
        features_df = features_df.fillna(0.0)
        
        # Split data for validation
        split_idx = int(len(features_df) * 0.8)
        X_train = features_df.iloc[:split_idx]
        y_train = labels_series.iloc[:split_idx]
        X_val = features_df.iloc[split_idx:]
        y_val = labels_series.iloc[split_idx:]
        
        if hyperparameter_search:
            # Perform grid search for hyperparameter optimization
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            
            grid_search = GridSearchCV(
                self.ml_ranker.model,
                param_grid,
                cv=3,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Use best model
            self.ml_ranker.model = grid_search.best_estimator_
            self.ml_ranker.is_trained = True
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            # Evaluate on validation set
            val_predictions = self.ml_ranker.model.predict(X_val)
            val_probabilities = self.ml_ranker.model.predict_proba(X_val)[:, 1]
        else:
            # Standard training
            self.ml_ranker.train_model(features_df, labels_series)
            
            # Evaluate on validation set
            val_predictions = self.ml_ranker.model.predict(X_val)
            val_probabilities = self.ml_ranker.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        results = self._calculate_evaluation_metrics(y_val, val_predictions, val_probabilities)
        
        # Cross-validation scores
        try:
            cv_scores = cross_val_score(self.ml_ranker.model, features_df, labels_series, cv=5, scoring='roc_auc')
            results['cv_mean'] = cv_scores.mean()
            results['cv_std'] = cv_scores.std()
            logger.info(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        except Exception as e:
            logger.error(f"Error calculating cross-validation scores: {e}")
            results['cv_mean'] = None
            results['cv_std'] = None
        
        results['success'] = True
        results['samples_trained'] = len(features_df)
        
        logger.info(f"Model training completed. Validation accuracy: {results['accuracy']:.4f}")
        return results
    
    def _calculate_evaluation_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                   y_prob: np.ndarray) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = np.mean(y_true == y_pred)
        
        # Classification report
        try:
            report = classification_report(y_true, y_pred, output_dict=True)
            metrics['precision'] = report['weighted avg']['precision']
            metrics['recall'] = report['weighted avg']['recall']
            metrics['f1_score'] = report['weighted avg']['f1-score']
        except Exception as e:
            logger.error(f"Error calculating classification report: {e}")
            metrics['precision'] = 0
            metrics['recall'] = 0
            metrics['f1_score'] = 0
        
        # ROC-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except Exception as e:
            logger.error(f"Error calculating ROC-AUC: {e}")
            metrics['roc_auc'] = 0
        
        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
        except Exception as e:
            logger.error(f"Error calculating confusion matrix: {e}")
            metrics['confusion_matrix'] = []
        
        return metrics
    
    def evaluate_model_performance(self) -> Dict:
        """
        Evaluate the current model's performance on historical data.
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating model performance")
        
        # Prepare data
        features_df, labels_series = self.prepare_training_data()
        
        if features_df.empty or labels_series.empty:
            logger.warning("No data available for evaluation")
            return {'success': False, 'message': 'No data available'}
        
        # Handle missing values
        features_df = features_df.fillna(0.0)
        
        # Make predictions
        if not self.ml_ranker.is_trained:
            logger.warning("Model not trained, cannot evaluate")
            return {'success': False, 'message': 'Model not trained'}
        
        predictions = self.ml_ranker.model.predict(features_df)
        probabilities = self.ml_ranker.model.predict_proba(features_df)[:, 1]
        
        # Calculate metrics
        results = self._calculate_evaluation_metrics(labels_series, predictions, probabilities)
        results['success'] = True
        results['samples_evaluated'] = len(features_df)
        
        logger.info(f"Model evaluation completed. Accuracy: {results['accuracy']:.4f}")
        return results
    
    def save_trained_model(self, model_name: str = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            model_name: Name for the model file (optional)
            
        Returns:
            Path to saved model file
        """
        if model_name is None:
            model_name = f"wave_ranker_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_path = self.models_dir / f"{model_name}.joblib"
        self.ml_ranker.save_model(str(model_path))
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    def load_model_for_training(self, model_path: str) -> bool:
        """
        Load a model for further training.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.ml_ranker.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return False
    
    def run_complete_training_cycle(self, hyperparameter_search: bool = False) -> Dict:
        """
        Run a complete training cycle: prepare data, train model, evaluate, and save.
        
        Args:
            hyperparameter_search: Whether to perform hyperparameter search
            
        Returns:
            Dictionary with complete training results
        """
        logger.info("Starting complete training cycle")
        
        results = {
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'hyperparameter_search': hyperparameter_search
        }
        
        try:
            # Prepare training data
            features_df, labels_series = self.prepare_training_data()
            results['data_preparation'] = {
                'samples': len(features_df),
                'features': list(features_df.columns) if not features_df.empty else []
            }
            
            if features_df.empty or labels_series.empty:
                results['success'] = False
                results['message'] = 'No training data available'
                return results
            
            # Train model
            training_results = self.train_model(features_df, labels_series, hyperparameter_search)
            results['training'] = training_results
            
            if not training_results.get('success', False):
                results['success'] = False
                results['message'] = 'Training failed'
                return results
            
            # Evaluate model
            evaluation_results = self.evaluate_model_performance()
            results['evaluation'] = evaluation_results
            
            # Save model
            model_path = self.save_trained_model()
            results['model_path'] = model_path
            
            results['success'] = True
            results['message'] = 'Training cycle completed successfully'
            
            logger.info("Complete training cycle finished successfully")
            
        except Exception as e:
            logger.error(f"Error in training cycle: {e}")
            results['success'] = False
            results['message'] = f'Training cycle failed: {str(e)}'
        
        return results

# Example usage function
def run_ml_training_session(config: Dict = None):
    """
    Run a complete ML training session.
    
    Args:
        config: Configuration dictionary (optional)
    """
    if config is None:
        config = {
            'database_path': './data/elliott_wave_data.db'
        }
    
    # Initialize trainer
    trainer = MLTrainingEvaluator(config)
    
    # Run complete training cycle
    results = trainer.run_complete_training_cycle(hyperparameter_search=True)
    
    # Print results
    print(json.dumps(results, indent=2))
    
    return trainer

if __name__ == "__main__":
    # Run training when script is executed directly
    run_ml_training_session()