"""
Wave Hypothesis Engine
Maintains multiple live wave hypotheses and prunes them when hard rules are violated.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import logging
import datetime
import json
from collections import deque

from .wave_detector import WaveDetector, ElliottRuleEngine
from .enhanced_rule_engine import EnhancedElliottRuleEngine
from .nms import NMS
from .confidence import ConfidenceScorer

logger = logging.getLogger(__name__)

class WaveHypothesis:
    """
    Represents a single wave count hypothesis with validity tracking.
    """
    def __init__(self, wave_count: Dict, segments: List[Dict], id: str):
        self.id = id
        self.wave_count = wave_count
        self.segments = segments
        self.created_at = datetime.datetime.now(datetime.timezone.utc)
        self.last_updated = self.created_at
        self.is_valid = True
        self.rule_violations = []
        self.confidence_score = 0.0
        self.ml_ranking_score = 0.0
        
    def add_violation(self, rule_name: str, message: str):
        """Add a rule violation and mark hypothesis as invalid."""
        self.rule_violations.append({
            'rule': rule_name,
            'message': message,
            'timestamp': datetime.datetime.now(datetime.timezone.utc)
        })
        self.is_valid = False
        self.last_updated = datetime.datetime.now(datetime.timezone.utc)
        logger.debug(f"Hypothesis {self.id} invalidated due to {rule_name}: {message}")
        
    def update_confidence(self, score: float):
        """Update confidence score for this hypothesis."""
        self.confidence_score = score
        self.last_updated = datetime.datetime.now(datetime.timezone.utc)
        
    def update_ml_ranking(self, score: float):
        """Update ML ranking score for this hypothesis."""
        self.ml_ranking_score = score
        self.last_updated = datetime.datetime.now(datetime.timezone.utc)
        
    def to_dict(self) -> Dict:
        """Convert hypothesis to dictionary for serialization."""
        return {
            'id': self.id,
            'wave_count': self.wave_count,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'is_valid': self.is_valid,
            'rule_violations': self.rule_violations,
            'confidence_score': self.confidence_score,
            'ml_ranking_score': self.ml_ranking_score
        }

class WaveHypothesisEngine:
    """
    Maintains multiple live wave hypotheses and ensures they comply with hard rules.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.rule_engine = ElliottRuleEngine(config)
        self.enhanced_rule_engine = EnhancedElliottRuleEngine(config)
        self.nms = NMS()
        # Initialize ConfidenceScorer with proper weights
        weights = {
            'rule_compliance': 0.6,
            'amplitude_duration_norm': 0.2,
            'volatility_adjustment': 0.1,
            'pattern_consistency': 0.1
        }
        self.confidence_scorer = ConfidenceScorer(weights)
        self.wave_detector = WaveDetector(config, self.rule_engine, self.nms, self.confidence_scorer)
        
        # Store active hypotheses
        self.hypotheses = {}  # id -> WaveHypothesis
        self.hypothesis_history = deque(maxlen=1000)  # Keep last 1000 hypotheses for analysis
        
        # Track invalidated hypotheses for post-analysis
        self.invalidated_hypotheses = deque(maxlen=1000)
        
        logger.info("WaveHypothesisEngine initialized")
        
    def generate_new_hypotheses(self, price_data: pd.DataFrame) -> List[str]:
        """
        Generate new wave hypotheses from price data.
        
        Args:
            price_data: DataFrame with timestamp index and price data
            
        Returns:
            List of new hypothesis IDs
        """
        logger.info("Generating new wave hypotheses")
        
        # Use existing wave detector to get candidates
        result = self.wave_detector.detect_waves(price_data)
        candidates = result.get('candidates', [])
        segments = result.get('wave_levels', [{}])[0].get('segments', [])
        
        new_hypothesis_ids = []
        
        for candidate in candidates:
            # Create a unique ID for this hypothesis
            hypothesis_id = f"hypothesis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.hypotheses)}"
            
            # Create hypothesis object
            hypothesis = WaveHypothesis(candidate, segments, hypothesis_id)
            
            # Validate against hard rules immediately
            if self._validate_hypothesis(hypothesis):
                self.hypotheses[hypothesis_id] = hypothesis
                new_hypothesis_ids.append(hypothesis_id)
                logger.debug(f"Created new valid hypothesis {hypothesis_id}")
            else:
                logger.debug(f"Created hypothesis {hypothesis_id} but it was immediately invalidated")
                
        logger.info(f"Generated {len(new_hypothesis_ids)} new hypotheses")
        return new_hypothesis_ids
        
    def _validate_hypothesis(self, hypothesis: WaveHypothesis) -> bool:
        """
        Validate a hypothesis against hard rules using enhanced validation.
        
        Args:
            hypothesis: WaveHypothesis to validate
            
        Returns:
            True if valid, False if invalidated
        """
        # Use enhanced rule engine for more accurate validation
        validation_result = self.enhanced_rule_engine.validate_hypothesis_immediately(
            hypothesis.wave_count, 
            hypothesis.segments
        )
        
        is_valid = validation_result['is_valid']
        compliance_score = validation_result['compliance_score']
        violations = validation_result['violations']
        
        if not is_valid:
            # Add all violations to the hypothesis
            for violation in violations:
                hypothesis.add_violation(violation['rule'], violation['message'])
            return False
            
        # Update confidence score
        hypothesis.update_confidence(compliance_score)
        return True
        
    def prune_invalid_hypotheses(self) -> List[str]:
        """
        Prune hypotheses that are no longer valid.
        
        Returns:
            List of invalidated hypothesis IDs
        """
        invalidated_ids = []
        
        # Create a list of keys to avoid modifying dict during iteration
        hypothesis_ids = list(self.hypotheses.keys())
        
        for hypothesis_id in hypothesis_ids:
            hypothesis = self.hypotheses[hypothesis_id]
            
            # Check if hypothesis is already marked as invalid
            if not hypothesis.is_valid:
                # Move to invalidated collection
                self.invalidated_hypotheses.append(hypothesis.to_dict())
                del self.hypotheses[hypothesis_id]
                invalidated_ids.append(hypothesis_id)
                logger.debug(f"Pruned invalidated hypothesis {hypothesis_id}")
                
        if invalidated_ids:
            logger.info(f"Pruned {len(invalidated_ids)} invalidated hypotheses")
            
        return invalidated_ids
        
    def get_active_hypotheses(self) -> List[WaveHypothesis]:
        """
        Get all currently active (valid) hypotheses.
        
        Returns:
            List of active WaveHypothesis objects
        """
        return list(self.hypotheses.values())
        
    def get_admissible_hypotheses(self) -> List[WaveHypothesis]:
        """
        Get all admissible hypotheses (valid ones that haven't been disqualified).
        
        Returns:
            List of admissible WaveHypothesis objects
        """
        # For now, admissible = active
        # In the future, this could include additional filtering
        return self.get_active_hypotheses()
        
    def update_hypotheses_with_ml_scores(self, ml_scores: Dict[str, float]):
        """
        Update hypotheses with ML ranking scores.
        
        Args:
            ml_scores: Dictionary mapping hypothesis IDs to ML scores
        """
        for hypothesis_id, score in ml_scores.items():
            if hypothesis_id in self.hypotheses:
                self.hypotheses[hypothesis_id].update_ml_ranking(score)
                
    def get_best_hypothesis(self) -> Optional[WaveHypothesis]:
        """
        Get the best hypothesis based on ML ranking scores.
        
        Returns:
            Best WaveHypothesis or None if no valid hypotheses
        """
        admissible = self.get_admissible_hypotheses()
        if not admissible:
            return None
            
        # Sort by ML ranking score (higher is better)
        sorted_hypotheses = sorted(admissible, key=lambda h: h.ml_ranking_score, reverse=True)
        return sorted_hypotheses[0] if sorted_hypotheses else None
        
    def check_for_invalidation_triggers(self, latest_price: float, timestamp: datetime.datetime) -> List[str]:
        """
        Check if any ongoing patterns have been invalidated by new price data.
        
        Args:
            latest_price: Latest price
            timestamp: Timestamp of latest price
            
        Returns:
            List of invalidated hypothesis IDs
        """
        # This would contain logic to check if ongoing wave patterns are invalidated
        # For now, we'll just prune any already invalidated hypotheses
        return self.prune_invalid_hypotheses()
        
    def serialize_state(self) -> Dict:
        """
        Serialize the current state of the hypothesis engine.
        
        Returns:
            Dictionary representation of the engine state
        """
        return {
            'active_hypotheses': {hid: h.to_dict() for hid, h in self.hypotheses.items()},
            'invalidated_hypotheses': list(self.invalidated_hypotheses),
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        
    def restore_state(self, state: Dict):
        """
        Restore the engine state from serialized data.
        
        Args:
            state: Dictionary representation of engine state
        """
        # Clear current state
        self.hypotheses.clear()
        self.invalidated_hypotheses.clear()
        
        # Restore active hypotheses
        for hypothesis_id, hypothesis_data in state.get('active_hypotheses', {}).items():
            # Recreate WaveHypothesis objects
            hypothesis = WaveHypothesis(
                hypothesis_data['wave_count'],
                [],  # Segments would need to be restored separately
                hypothesis_id
            )
            
            # Restore attributes
            hypothesis.is_valid = hypothesis_data['is_valid']
            hypothesis.rule_violations = hypothesis_data['rule_violations']
            hypothesis.confidence_score = hypothesis_data['confidence_score']
            hypothesis.ml_ranking_score = hypothesis_data['ml_ranking_score']
            
            # Parse timestamps
            hypothesis.created_at = datetime.datetime.fromisoformat(hypothesis_data['created_at'].replace('Z', '+00:00'))
            hypothesis.last_updated = datetime.datetime.fromisoformat(hypothesis_data['last_updated'].replace('Z', '+00:00'))
            
            self.hypotheses[hypothesis_id] = hypothesis
            
        # Restore invalidated hypotheses
        for hypothesis_data in state.get('invalidated_hypotheses', []):
            self.invalidated_hypotheses.append(hypothesis_data)
            
        logger.info(f"Restored hypothesis engine state with {len(self.hypotheses)} active hypotheses")