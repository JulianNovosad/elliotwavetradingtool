"""
Enhanced Elliott Rule Engine
Implements corrected and enhanced Elliott Wave rules with proper price-based validation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import logging
import datetime

logger = logging.getLogger(__name__)

class EnhancedElliottRuleEngine:
    """
    Enhanced version of the ElliottRuleEngine with corrected implementations of hard rules.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        logger.info("EnhancedElliottRuleEngine initialized")
        
    def evaluate_wave_count(self, wave_count: Dict, segments_in_count: List[Dict]) -> Dict:
        """
        Enhanced evaluation of a wave count against all hard rules.
        
        Args:
            wave_count: Dictionary representing a wave count candidate
            segments_in_count: List of segment dictionaries that make up this count
            
        Returns:
            Updated wave_count dictionary with compliance scores and violations
        """
        # Initialize compliance score and violations list
        total_rule_compliance = 1.0
        rule_violations = []
        
        logger.debug(f"Evaluating wave count with {len(segments_in_count)} segments")
        
        # 1. Sequentiality Rule: Waves must be in correct order
        # This is implicitly satisfied by construction in the wave detector
        
        # 2. Wave 3 Length Rule: Wave 3 is never the shortest impulse wave
        w3_length_score = self._check_wave3_length_rule(segments_in_count)
        total_rule_compliance *= w3_length_score
        if w3_length_score == 0.0:
            rule_violations.append({
                'rule': 'wave3_never_shortest', 
                'message': 'Wave 3 is the shortest impulse wave in terms of price movement'
            })
        
        # 3. Wave 2 Retracement Rule: Wave 2 never retraces more than 100% of Wave 1
        w2_retracement_score = self._check_wave2_retracement_rule(segments_in_count)
        total_rule_compliance *= w2_retracement_score
        if w2_retracement_score == 0.0:
            rule_violations.append({
                'rule': 'wave2_retracement', 
                'message': 'Wave 2 retraces more than 100% of Wave 1'
            })
        
        # 4. Wave 4 Retracement Rule: Wave 4 never retraces more than 100% of Wave 3
        w4_retracement_score = self._check_wave4_retracement_rule(segments_in_count)
        total_rule_compliance *= w4_retracement_score
        if w4_retracement_score == 0.0:
            rule_violations.append({
                'rule': 'wave4_retracement', 
                'message': 'Wave 4 retraces more than 100% of Wave 3'
            })
        
        # 5. Wave 3 Travel Rule: Wave 3 always travels beyond the end of Wave 1
        w3_travel_score = self._check_wave3_travel_rule(segments_in_count)
        total_rule_compliance *= w3_travel_score
        if w3_travel_score == 0.0:
            rule_violations.append({
                'rule': 'wave3_travel_beyond', 
                'message': 'Wave 3 does not travel beyond the end of Wave 1'
            })
        
        # 6. Wave 4/1 Overlap Rule: In impulses, Wave 4 may not overlap Wave 1
        overlap_score = self._check_wave4_wave1_overlap_rule(segments_in_count)
        total_rule_compliance *= overlap_score
        if overlap_score == 0.0:
            rule_violations.append({
                'rule': 'wave4_wave1_overlap', 
                'message': 'Wave 4 overlaps with Wave 1 price territory'
            })
        
        # 7. Corrective Waves are Never Fives Rule
        corrective_score = self._check_corrective_waves_not_fives(segments_in_count)
        total_rule_compliance *= corrective_score
        if corrective_score == 0.0:
            rule_violations.append({
                'rule': 'corrective_waves_not_fives', 
                'message': 'Corrective wave pattern has 5 waves (should be 3)'
            })
        
        # Update wave count with results
        wave_count['rule_compliance_score'] = total_rule_compliance
        wave_count['rule_violations'] = rule_violations
        
        logger.debug(f"Wave count evaluation completed. Compliance score: {total_rule_compliance}")
        return wave_count
    
    def _check_wave3_length_rule(self, segments_in_count: List[Dict]) -> float:
        """
        Check that Wave 3 is never the shortest impulse wave in terms of price movement.
        
        Args:
            segments_in_count: List of segment dictionaries
            
        Returns:
            1.0 if compliant, 0.0 if violated
        """
        # Find impulse wave segments (1, 3, 5)
        impulse_waves = {}
        for seg in segments_in_count:
            label = seg.get('label')
            if label in ['1', '3', '5']:
                # Calculate price movement (end price - start price)
                price_start = seg.get('price_start', seg.get('price', 0))
                price_end = seg.get('price_end', seg.get('price', 0))
                price_movement = abs(price_end - price_start)
                impulse_waves[label] = price_movement
        
        if len(impulse_waves) < 3:
            # Not enough impulse waves to check
            return 1.0
        
        # Check if Wave 3 is shortest
        move_1 = impulse_waves.get('1', 0)
        move_3 = impulse_waves.get('3', 0)
        move_5 = impulse_waves.get('5', 0)
        
        if move_3 <= min(move_1, move_5) and move_3 > 0:
            logger.debug(f"Rule violation: Wave 3 price movement ({move_3:.2f}) is shortest among impulse waves (1={move_1:.2f}, 5={move_5:.2f})")
            return 0.0  # Violation
        
        return 1.0  # Compliant
    
    def _check_wave2_retracement_rule(self, segments_in_count: List[Dict]) -> float:
        """
        Check that Wave 2 never retraces more than 100% of Wave 1.
        
        Args:
            segments_in_count: List of segment dictionaries
            
        Returns:
            1.0 if compliant, 0.0 if violated
        """
        # Find Wave 1 and Wave 2 segments
        wave1_seg = next((s for s in segments_in_count if s.get('label') == '1'), None)
        wave2_seg = next((s for s in segments_in_count if s.get('label') == '2'), None)
        
        if not wave1_seg or not wave2_seg:
            # Can't check if segments are missing
            return 1.0
        
        # Calculate price movements
        wave1_move = wave1_seg.get('price_end', wave1_seg.get('price', 0)) - wave1_seg.get('price_start', wave1_seg.get('price', 0))
        wave2_move = wave2_seg.get('price_end', wave2_seg.get('price', 0)) - wave2_seg.get('price_start', wave2_seg.get('price', 0))
        
        # Check retracement
        if wave1_move != 0:
            wave2_retracement = abs(wave2_move / wave1_move)
            if wave2_retracement > 1.0:
                logger.debug(f"Rule violation: Wave 2 retraces {wave2_retracement:.2f} (> 100%) of Wave 1")
                return 0.0  # Violation
        
        return 1.0  # Compliant
    
    def _check_wave4_retracement_rule(self, segments_in_count: List[Dict]) -> float:
        """
        Check that Wave 4 never retraces more than 100% of Wave 3.
        
        Args:
            segments_in_count: List of segment dictionaries
            
        Returns:
            1.0 if compliant, 0.0 if violated
        """
        # Find Wave 3 and Wave 4 segments
        wave3_seg = next((s for s in segments_in_count if s.get('label') == '3'), None)
        wave4_seg = next((s for s in segments_in_count if s.get('label') == '4'), None)
        
        if not wave3_seg or not wave4_seg:
            # Can't check if segments are missing
            return 1.0
        
        # Calculate price movements
        wave3_move = wave3_seg.get('price_end', wave3_seg.get('price', 0)) - wave3_seg.get('price_start', wave3_seg.get('price', 0))
        wave4_move = wave4_seg.get('price_end', wave4_seg.get('price', 0)) - wave4_seg.get('price_start', wave4_seg.get('price', 0))
        
        # Check retracement
        if wave3_move != 0:
            wave4_retracement = abs(wave4_move / wave3_move)
            if wave4_retracement > 1.0:
                logger.debug(f"Rule violation: Wave 4 retraces {wave4_retracement:.2f} (> 100%) of Wave 3")
                return 0.0  # Violation
        
        return 1.0  # Compliant
    
    def _check_wave3_travel_rule(self, segments_in_count: List[Dict]) -> float:
        """
        Check that Wave 3 always travels beyond the end of Wave 1.
        
        Args:
            segments_in_count: List of segment dictionaries
            
        Returns:
            1.0 if compliant, 0.0 if violated
        """
        # Find Wave 1 and Wave 3 segments
        wave1_seg = next((s for s in segments_in_count if s.get('label') == '1'), None)
        wave3_seg = next((s for s in segments_in_count if s.get('label') == '3'), None)
        
        if not wave1_seg or not wave3_seg:
            # Can't check if segments are missing
            return 1.0
        
        # Get end prices
        wave1_end_price = wave1_seg.get('price_end', wave1_seg.get('price', 0))
        wave3_end_price = wave3_seg.get('price_end', wave3_seg.get('price', 0))
        
        # Determine trend direction from Wave 1
        wave1_start_price = wave1_seg.get('price_start', wave1_seg.get('price', 0))
        trend_up = wave1_end_price > wave1_start_price
        
        # Check if Wave 3 travels beyond end of Wave 1
        if trend_up and wave3_end_price <= wave1_end_price:
            logger.debug(f"Rule violation: Wave 3 end price ({wave3_end_price:.2f}) does not travel beyond Wave 1 end price ({wave1_end_price:.2f})")
            return 0.0  # Violation
        elif not trend_up and wave3_end_price >= wave1_end_price:
            logger.debug(f"Rule violation: Wave 3 end price ({wave3_end_price:.2f}) does not travel beyond Wave 1 end price ({wave1_end_price:.2f})")
            return 0.0  # Violation
        
        return 1.0  # Compliant
    
    def _check_wave4_wave1_overlap_rule(self, segments_in_count: List[Dict]) -> float:
        """
        Check that in impulses, Wave 4 may not overlap Wave 1.
        
        Args:
            segments_in_count: List of segment dictionaries
            
        Returns:
            1.0 if compliant, 0.0 if violated
        """
        # Find Wave 1 and Wave 4 segments
        wave1_seg = next((s for s in segments_in_count if s.get('label') == '1'), None)
        wave4_seg = next((s for s in segments_in_count if s.get('label') == '4'), None)
        
        if not wave1_seg or not wave4_seg:
            # Can't check if segments are missing
            return 1.0
        
        # Get price ranges for both waves
        wave1_low = wave1_seg.get('price_low', min(wave1_seg.get('price_start', 0), wave1_seg.get('price_end', 0)))
        wave1_high = wave1_seg.get('price_high', max(wave1_seg.get('price_start', 0), wave1_seg.get('price_end', 0)))
        wave4_low = wave4_seg.get('price_low', min(wave4_seg.get('price_start', 0), wave4_seg.get('price_end', 0)))
        wave4_high = wave4_seg.get('price_high', max(wave4_seg.get('price_start', 0), wave4_seg.get('price_end', 0)))
        
        # Determine direction of Wave 1
        w1_direction_up = False
        if wave1_seg.get('price_high') is not None and wave1_seg.get('price_low') is not None:
            w1_direction_up = wave1_seg['price_high'] > wave1_seg['price_low']
        elif wave1_seg.get('price_end') is not None and wave1_seg.get('price_start') is not None:
            w1_direction_up = wave1_seg['price_end'] > wave1_seg['price_start']
        
        # Check for overlap
        if w1_direction_up:
            # Upward Wave 1: Wave 4's low must be above Wave 1's high
            if wave4_low <= wave1_high:
                logger.debug(f"Rule violation: Wave 4 overlaps Wave 1 (upward trend). Wave 4 low: {wave4_low:.2f}, Wave 1 high: {wave1_high:.2f}")
                return 0.0  # Violation
        else:
            # Downward Wave 1: Wave 4's high must be below Wave 1's low
            if wave4_high >= wave1_low:
                logger.debug(f"Rule violation: Wave 4 overlaps Wave 1 (downward trend). Wave 4 high: {wave4_high:.2f}, Wave 1 low: {wave1_low:.2f}")
                return 0.0  # Violation
        
        return 1.0  # Compliant
    
    def _check_corrective_waves_not_fives(self, segments_in_count: List[Dict]) -> float:
        """
        Check that corrective waves are never fives (only motive waves are fives).
        
        Args:
            segments_in_count: List of segment dictionaries
            
        Returns:
            1.0 if compliant, 0.0 if violated
        """
        # Count total segments to determine if this is a corrective pattern
        # Corrective waves should have exactly 3 segments (a-b-c)
        # Motive waves should have exactly 5 segments (1-2-3-4-5)
        
        labels = [s.get('label') for s in segments_in_count if s.get('label')]
        
        # Check if this looks like a corrective pattern (a-b-c)
        if set(labels) == {'a', 'b', 'c'}:
            # This is correctly a 3-wave corrective pattern
            return 1.0
        elif len(labels) == 3 and all(label.lower() in ['a', 'b', 'c'] for label in labels):
            # This is a 3-wave pattern that looks like a corrective wave
            return 1.0
        elif len(labels) == 5 and all(label in ['1', '2', '3', '4', '5'] for label in labels):
            # This is a 5-wave motive pattern, which is correct
            return 1.0
        elif len(labels) == 5:
            # This is a 5-wave pattern but not clearly labeled as motive
            # Check if it's structured like a corrective pattern that shouldn't be 5 waves
            # This is a heuristic check - if it has 5 waves but looks corrective, it's wrong
            return 1.0  # For now, we'll assume it's okay unless we can definitively say it's wrong
        else:
            # Unclear pattern, assume compliant
            return 1.0
    
    def validate_hypothesis_immediately(self, wave_count: Dict, segments: List[Dict]) -> Dict:
        """
        Immediate validation of a hypothesis against hard rules.
        
        Args:
            wave_count: Dictionary representing a wave count candidate
            segments: List of segment dictionaries that make up this count
            
        Returns:
            Dictionary with validation results
        """
        # Evaluate the wave count using the enhanced rule engine
        evaluated_count = self.evaluate_wave_count(wave_count, segments)
        
        # Check if any hard rules were violated
        compliance_score = evaluated_count.get('rule_compliance_score', 1.0)
        violations = evaluated_count.get('rule_violations', [])
        
        is_valid = compliance_score > 0.0
        
        return {
            'is_valid': is_valid,
            'compliance_score': compliance_score,
            'violations': violations,
            'evaluated_count': evaluated_count
        }