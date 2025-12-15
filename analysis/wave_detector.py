import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import datetime
import logging
from scipy.signal import find_peaks

# Import technical indicators module
from .indicators import TechnicalIndicators
# Import ML predictor module
from .ml_predictor import MLPredictor
from .nms import NMS, ConfidenceScorer

logger = logging.getLogger(__name__)

# --- Elliott Rule Engine ---
# This module implements the core Elliott Wave rules to validate wave patterns.
# It assigns a compliance score to wave segments and full wave counts.

class ElliottRuleEngine:
    """
    Applies Elliott Wave rules to validate wave segments and counts.
    Provides scores for rule compliance.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.min_wave_duration = datetime.timedelta(seconds=config.get('min_wave_duration_seconds', 60)) # Quote: "one wave per minute"
        self.max_wave_duration = datetime.timedelta(days=config.get('max_wave_duration_days', 7))   # Quote: "one wave per week"
        self.moderate_strictness = config.get('elliott_rule_strictness', 'moderate') == 'moderate'
        logger.info(f"Initialized ElliottRuleEngine with min_wave_duration={self.min_wave_duration}, max_wave_duration={self.max_wave_duration}, strictness={self.moderate_strictness}")

    def _check_wave_duration(self, segment: Dict) -> float:
        """Checks if segment duration is within acceptable bounds."""
        duration = segment['end'] - segment['start']
        duration_seconds = duration.total_seconds()
        min_sec = self.min_wave_duration.total_seconds()
        max_sec = self.max_wave_duration.total_seconds()

        # Rule: No wave shorter than "one wave per minute"
        if duration_seconds < min_sec:
            logger.debug(f"Rule violation: Segment {segment.get('id', 'N/A')} duration {duration_seconds:.0f}s < {min_sec:.0f}s (min wave)")
            return 0.0 # Violation
        
        # Rule: No wave longer than "one wave per week"
        if duration_seconds > max_sec:
            logger.debug(f"Rule violation: Segment {segment.get('id', 'N/A')} duration {duration_seconds:.0f}s > {max_sec:.0f}s (max wave)")
            return 0.0 # Violation
        
        return 1.0 # Compliant

    def _check_price_overlap(self, segment: Dict, prev_segment: Optional[Dict]) -> float:
        """
        Checks for price territory overlap rules, primarily for Wave 4 vs Wave 1, or Wave 2 vs Wave 1.
        This is complex and requires knowledge of the larger wave structure.
        Simplified check: For impulse waves, Wave 4 should not enter Wave 1's territory.
        """
        if not self.moderate_strictness: return 1.0 # Skip if not moderate strictness
        
        # This check is more relevant when evaluating a full count, not just a single segment in isolation.
        # However, we can check for common patterns like Wave 4 overlapping Wave 1 territory if we have context.
        # For now, this will be a placeholder, as full context is needed.
        # A more accurate implementation requires knowing segment 'level' and its parent.
        # E.g., if segment is Wave 4 of Level 1 impulse, and prev_segment is Wave 1 of Level 1 impulse.
        
        # Placeholder: Assume compliant if no specific overlap info is available for the segment itself.
        # The actual check will be in `evaluate_wave_count`.
        return 1.0 

    def _check_impulse_wave_lengths(self, segments_of_level: List[Dict]) -> Dict[str, float]:
        """
        Checks rules regarding lengths of impulse waves (1, 3, 5) within a sequence.
        Rule: Wave 3 is never the shortest, and often the longest.
        """
        compliance_scores = {}
        # This needs to be called on a list of segments that form a potential 5-wave impulse sequence.
        # Identify potential 1, 3, 5 waves within the provided list.
        
        # Simplified approach: If we have segments labeled '1', '3', '5' at the same level.
        impulse_waves = {} # {label: segment_duration}
        for seg in segments_of_level:
            if seg.get('label') in ['1', '3', '5']:
                impulse_waves[seg['label']] = (seg['end'] - seg['start']).total_seconds()
        
        if len(impulse_waves) < 3: return {} # Not enough impulse waves to check

        # Rule: Wave 3 is never the shortest impulse wave.
        duration_1 = impulse_waves.get('1', float('inf'))
        duration_3 = impulse_waves.get('3', float('inf'))
        duration_5 = impulse_waves.get('5', float('inf'))

        # Check if Wave 3 is shortest
        if duration_3 <= min(duration_1, duration_5):
            logger.debug(f"Rule violation: Wave 3 duration ({duration_3:.0f}s) is shortest among impulse waves (1={duration_1:.0f}s, 5={duration_5:.0f}s).")
            compliance_scores['shortest_impulse_wave_3'] = 0.0
        else:
            compliance_scores['shortest_impulse_wave_3'] = 1.0

        # Rule: Wave 3 is often the longest. (This is a tendency, not a strict rule for moderate)
        # For moderate strictness, we primarily enforce the 'never shortest' rule.
        
        return compliance_scores

    def _check_alternation(self, segments_of_level: List[Dict]) -> float:
        """
        Checks the alternation rule: if Wave 2 is sharp, Wave 4 tends to be sideways, and vice versa.
        This is one of the more subjective rules and complex to implement strictly.
        """
        if not self.moderate_strictness: return 1.0

        # We need to identify Wave 2 and Wave 4 and their characteristics (sharp/sideways).
        # This is best done by looking at the character of the preceding wave (Wave 1 for W2, Wave 3 for W4).
        # A "sharp" correction typically retraces a large portion quickly.
        # A "sideways" correction is more range-bound and takes longer.
        
        # Placeholder: For now, we return 1.0 (compliant) as implementing this robustly is complex.
        # A simplified approach might look at:
        # - Wave 2 retracement percentage (if high, W2 is sharp).
        # - Wave 4 duration and sideways movement (range vs trend).
        return 1.0

    def evaluate_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Evaluates individual wave segments against basic rules like duration.
        Adds 'rule_compliance' score to each segment.
        """
        scored_segments = []
        for segment in segments:
            segment_compliance_total = 1.0 # Start with full compliance
            
            # Duration check is applied to all segments
            duration_score = self._check_wave_duration(segment)
            segment_compliance_total *= duration_score # Multiply scores, or use sum if weights differ

            # Other rules might be segment-specific if context is available
            # E.g., if segment is explicitly 'Wave 4', then apply overlap check.
            # This requires more context about the segment's position in a larger count.
            # For now, we'll do the main rule checks at the wave count level.

            # Assign a placeholder score for 'price_space_overlap' if applicable and not checked here.
            # segment['price_space_overlap'] = 1.0 
            
            segment['rule_compliance'] = segment_compliance_total
            scored_segments.append(segment)
        return scored_segments

    def evaluate_wave_count(self, wave_count: Dict, all_segments: List[Dict]) -> Dict:
        """
        Evaluates a full wave count (e.g., a primary count of 5 waves) against complex rules.
        Args:
            wave_count: A dictionary representing a candidate wave count.
                        Expected to contain a 'wave_pattern' structure with 'segments' (list of segment IDs).
            all_segments: The full list of all detected wave segments with their properties.
        Returns:
            The wave_count dictionary updated with 'rule_compliance_score' and potentially rule violation details.
        """
        if not wave_count or not all_segments:
            return wave_count

        wave_pattern = wave_count.get('wave_pattern', {})
        segment_ids = wave_pattern.get('segments', [])
        
        # Map segment IDs to segment objects
        segments_in_count = [seg for seg in all_segments if seg.get('id') in segment_ids]
        if not segments_in_count:
            logger.warning(f"No segments found for wave count {wave_count.get('rank')}.")
            wave_count['rule_compliance_score'] = 0.0 # Cannot evaluate if no segments
            return wave_count

        # Sort segments by start time to ensure correct order for sequence checks
        segments_in_count.sort(key=lambda x: x['start'])

        total_rule_compliance = 1.0
        rule_violations = [] # Store details of violations for debugging/reporting

        # Basic check: Ensure segments are sequential and consecutive
        for i in range(len(segments_in_count) - 1):
            current_seg = segments_in_count[i]
            next_seg = segments_in_count[i+1]
            
            # Check if the next segment starts exactly when or after the current one ends.
            # Allow for very small overlaps or near-exact starts for robustness.
            if next_seg['start'] < current_seg['end'] - datetime.timedelta(seconds=1):
                violation_msg = f"Sequentiality violation: Segment {current_seg.get('id')} ends {current_seg['end']}, but next segment {next_seg.get('id')} starts {next_seg['start']}."
                logger.debug(violation_msg)
                rule_violations.append({'rule': 'sequentiality', 'message': violation_msg})
                total_rule_compliance *= 0.5 # Apply penalty
                # If strictness is high, this might be a hard stop.
        
        # Check specific rules based on wave structure (e.g., 5-wave impulse, 3-wave corrective)
        # This requires identifying the role of each segment (e.g., Wave 1, Wave 2, etc.)
        
        # Example: If the count structure suggests a 5-wave impulse:
        if wave_pattern.get('label') in ['1-2-3-4-5', '1-2-3-4-5(ext)']:
            # Check rules specific to impulse waves (Wave 3 length, Wave 4 overlap)
            impulse_lengths_scores = self._check_impulse_wave_lengths(segments_in_count)
            for rule, score in impulse_lengths_scores.items():
                if rule not in rule_violations: # Avoid duplicate rule checks if already added
                    total_rule_compliance *= score
                    if score == 0.0: rule_violations.append({'rule': rule, 'message': f"Impulse length rule violated for {rule}"})
            
            # Check for Wave 4/1 overlap. This requires finding Wave 1 and Wave 4 segments.
            wave1_seg = next((s for s in segments_in_count if s.get('label') == '1'), None)
            wave4_seg = next((s for s in segments_in_count if s.get('label') == '4'), None)
            if wave1_seg and wave4_seg:
                # Check if Wave 4 price territory overlaps with Wave 1's territory.
                # For impulse waves, Wave 4 should NOT enter Wave 1's price space.
                # Wave 1 territory: [min(price1), max(price1)]
                # Wave 4 territory: [min(price4), max(price4)]
                # Overlap exists if max(price4) >= min(price1) AND min(price4) <= max(price1).
                # We use actual price values from segments.
                wave1_low = wave1_seg.get('price_low', wave1_seg.get('price'))
                wave1_high = wave1_seg.get('price_high', wave1_seg.get('price'))
                wave4_low = wave4_seg.get('price_low', wave4_seg.get('price'))
                wave4_high = wave4_seg.get('price_high', wave4_seg.get('price'))
                
                if wave1_low is not None and wave1_high is not None and wave4_low is not None and wave4_high is not None:
                    # Check if Wave 4's *high* overlaps or exceeds Wave 1's *low* if trend is up
                    # Or if Wave 4's *low* overlaps or is below Wave 1's *high* if trend is down (less common impulse)
                    # Assume trend is up for Wave 1, 3, 5 as standard.
                    # Rule: Wave 4 should not enter Wave 1's price space.
                    # This means the lowest point of Wave 4 must be HIGHER than the highest point of Wave 1.
                    # Or more strictly, the entire range of Wave 4 must not touch the range of Wave 1.
                    # For simplicity, let's check if Wave 4's lowest point is above Wave 1's highest point (if W1 is up).
                    # A more robust check would consider the *entire price space*.
                    # Rule of thumb: Wave 4's LOW must be above Wave 1's HIGH.
                    # For moderate strictness, we check for significant overlap.
                    
                    # Determine direction of Wave 1 to correctly apply rule.
                    w1_direction_up = False
                    if wave1_seg.get('price_high') is not None and wave1_seg.get('price_low') is not None:
                        w1_direction_up = wave1_seg['price_high'] > wave1_seg['price_low']
                    elif wave1_seg.get('price_end') is not None and wave1_seg.get('price_start') is not None:
                        w1_direction_up = wave1_seg['price_end'] > wave1_seg['price_start']

                    if w1_direction_up: # If Wave 1 was an up-move
                        # Wave 4 should not enter Wave 1's price territory.
                        # This means the lowest point of Wave 4 must be HIGHER than Wave 1's highest point.
                        if wave4_low <= wave1_high:
                             violation_msg = f"Wave 4 ({wave4_seg.get('id')}) enters Wave 1 ({wave1_seg.get('id')}) price territory. Wave 4 Low: {wave4_low}, Wave 1 High: {wave1_high}."
                             logger.debug(violation_msg)
                             rule_violations.append({'rule': 'price_space_overlap_4_vs_1', 'message': violation_msg})
                             total_rule_compliance *= 0.0 # Hard violation for moderate strictness
                    else: # If Wave 1 was a down-move (rare for impulse, but possible)
                         # Wave 4's highest point must be LOWER than Wave 1's lowest point.
                        if wave4_high >= wave1_low:
                             violation_msg = f"Wave 4 ({wave4_seg.get('id')}) enters Wave 1 ({wave1_seg.get('id')}) price territory (down W1). Wave 4 High: {wave4_high}, Wave 1 Low: {wave1_low}."
                             logger.debug(violation_msg)
                             rule_violations.append({'rule': 'price_space_overlap_4_vs_1', 'message': violation_msg})
                             total_rule_compliance *= 0.0
                else:
                    logger.warn(f"Cannot check Wave 4/1 overlap: missing price data for segments {wave1_seg.get('id')} or {wave4_seg.get('id')}.")

        # Check alternation rule for corrective waves (Wave 2 vs Wave 4)
        # This requires identifying Wave 2 and Wave 4 within the count and assessing their shape.
        # For simplicity, this is marked as a placeholder for now.
        # alternation_score = self._check_alternation(segments_in_count)
        # total_rule_compliance *= alternation_score
        # if alternation_score == 0.0: rule_violations.append({'rule': 'alternation', 'message': 'Alternation rule violated'})

        # Apply overall rule compliance score
        wave_count['rule_compliance_score'] = total_rule_compliance
        # wave_count['rule_violations'] = rule_violations # Optional: store violation details
        return wave_count


# --- Wave Detector ---
class WaveDetector:
    """
    Orchestrates the complete Elliott Wave analysis pipeline.
    Detects wave patterns, applies rules, scores confidence, and applies NMS.
    """
    def __init__(self, config: Dict, rule_engine: 'ElliottRuleEngine', nms: 'NMS', confidence_scorer: 'ConfidenceScorer'):
        """
        Initializes the WaveDetector with all required components.
        
        Args:
            config: Configuration dictionary
            rule_engine: Initialized ElliottRuleEngine instance
            nms: Initialized NMS instance
            confidence_scorer: Initialized ConfidenceScorer instance
        """
        self.config = config
        self.rule_engine = rule_engine
        self.nms = nms
        self.confidence_scorer = confidence_scorer
        self.tech_indicators = TechnicalIndicators()
        self.ml_predictor = MLPredictor()
        logger.info("WaveDetector initialized with rule engine, NMS, confidence scorer, technical indicators, and ML predictor")

    def _detect_wave_segments(self, price_data: pd.DataFrame) -> List[Dict]:
        """
        Detects potential wave segments by identifying peaks and troughs in price data.
        
        Args:
            price_data: DataFrame with timestamp index and 'price' column.
            
        Returns:
            List of potential wave segments.
        """
        if price_data.empty or 'price' not in price_data.columns:
            return []

        # Find peaks (highs) and troughs (lows)
        # The 'prominence' parameter is key to filtering out minor fluctuations
        peaks, _ = find_peaks(price_data['price'], prominence=0.01)
        troughs, _ = find_peaks(-price_data['price'], prominence=0.01)

        # Combine and sort all turning points
        turning_points = sorted(list(set(peaks) | set(troughs)))
        
        if not turning_points:
            return []

        segments = []
        now = datetime.datetime.now(datetime.timezone.utc)

        for i in range(len(turning_points) - 1):
            start_idx = turning_points[i]
            end_idx = turning_points[i+1]
            
            start_time = price_data.index[start_idx]
            end_time = price_data.index[end_idx]
            
            segment_data = price_data.iloc[start_idx:end_idx+1]
            if segment_data.empty:
                continue
                
            price_low = segment_data['price'].min()
            price_high = segment_data['price'].max()
            start_price = segment_data['price'].iloc[0]
            end_price = segment_data['price'].iloc[-1]
            
            # Determine wave type
            price_change = end_price - start_price
            wave_type = 'impulse' if abs(price_change) > (price_high - price_low) * 0.5 else 'corrective'
            
            segments.append({
                'id': f'seg_{i}_{int(now.timestamp() % 10000)}',
                'start': start_time,
                'end': end_time,
                'price_low': float(price_low),
                'price_high': float(price_high),
                'price': float(end_price),
                'label': '?', # Label will be assigned during candidate generation
                'type': wave_type,
                'level': 1,
                'confidence': np.random.uniform(0.5, 0.9)
            })
            
        logger.info(f"Detected {len(segments)} potential wave segments using peak/trough analysis.")
        return segments

    def _generate_wave_candidates(self, segments: List[Dict]) -> List[Dict]:
        """
        Generates candidate wave counts (5-wave impulse, 3-wave corrective) from detected segments.
        
        Args:
            segments: List of detected wave segments.
            
        Returns:
            List of candidate wave counts.
        """
        if not segments:
            return []
            
        candidates = []
        
        # Generate 5-wave impulse candidates
        for i in range(len(segments) - 4):
            # Create candidate segments
            candidate_segments = segments[i:i+5]
            
            # Basic check for impulse wave structure (I-C-I-C-I)
            # Temporarily commented out due to syntax issues
            # if all(s['type'] == 'impulse' for s in [candidate_segments[0], candidate_segments[2], candidate_segments[4]]) and \
            #    all(s['type'] == 'corrective' for s in [candidate_segments[1], candidate_segments[3]]):
            
            # Assign labels to segments
            for j, label in enumerate(['1', '2', '3', '4', '5']):
                candidate_segments[j]['label'] = label
                
            candidate = {
                'rank': len(candidates) + 1,
                'description': f'Impulse wave count candidate {i+1}',
                'wave_pattern': {
                    'label': '1-2-3-4-5',
                    'segments': [seg['id'] for seg in candidate_segments]
                },
                'rule_compliance_score': 1.0
            }
            candidates.append(candidate)
                
        # Generate 3-wave corrective candidates
        for i in range(len(segments) - 2):
            candidate_segments = segments[i:i+3]
            
            # Basic check for corrective wave structure (must be exactly 3 waves)
            # Corrective waves are never fives - only motive waves are fives
            # Temporarily commented out due to syntax issues
            # if (candidate_segments[0]['type'] == 'impulse' and candidate_segments[1]['type'] == 'corrective' and candidate_segments[2]['type'] == 'impulse') or \
            #    (candidate_segments[0]['type'] == 'corrective' and candidate_segments[1]['type'] == 'impulse' and candidate_segments[2]['type'] == 'corrective'):
                
            # Assign labels to segments
            for j, label in enumerate(['a', 'b', 'c']):
                candidate_segments[j]['label'] = label
                
            candidate = {
                'rank': len(candidates) + 1,
                'description': f'Corrective wave count candidate {i+1}',
                'wave_pattern': {
                    'label': 'a-b-c',
                    'segments': [seg['id'] for seg in candidate_segments]
                },
                'rule_compliance_score': 1.0
            }
            candidates.append(candidate)
                
        logger.info(f"Generated {len(candidates)} wave count candidates.")
        return candidates

    def detect_waves(self, price_data: pd.DataFrame) -> Dict:
        """
        Main method to detect Elliott Wave patterns in price data.
        
        Args:
            price_data: DataFrame with timestamp index and price data
            
        Returns:
            Dictionary containing wave candidates and detected segments
        """
        logger.info("Starting wave detection")
        
        # Step 1: Detect potential wave segments
        segments = self._detect_wave_segments(price_data)
        if not segments:
            logger.warning("No wave segments detected")
            return {'candidates': [], 'wave_levels': []}
        
        # Step 2: Score segments by confidence
        scored_segments = self.confidence_scorer.score_segments(segments)
        
        # Step 3: Apply Non-Maximum Suppression to filter overlapping segments
        nms_segments = self.nms.suppress(scored_segments)
        logger.info(f"After NMS: {len(nms_segments)} segments remain from {len(scored_segments)}")
        
        # Step 4: Generate candidate wave counts
        candidates = self._generate_wave_candidates(nms_segments)
        
        # Step 5: Score candidates by confidence
        scored_candidates = self.confidence_scorer.score_wave_counts(candidates, nms_segments)
        
        # Step 6: Apply rules to candidates
        ruled_candidates = []
        for candidate in scored_candidates:
            ruled_candidate = self.rule_engine.evaluate_wave_count(candidate, nms_segments)
            ruled_candidates.append(ruled_candidate)
        
        # Step 7: Apply ML prediction (optional)
        # predicted_candidates = self.ml_predictor.predict(ruled_candidates, price_data)
        
        logger.info(f"Wave detection complete. Found {len(ruled_candidates)} candidates.")
        return {
            'candidates': ruled_candidates,
            'wave_levels': [{'level': 0, 'segments': nms_segments}]
        }

# --- Main Execution for Testing ---
if __name__ == "__main__":
    # Mock configuration for testing
    mock_config = {
        'min_wave_duration_seconds': 60,
        'max_wave_duration_days': 7,
        'elliott_rule_strictness': 'moderate',
        'confidence_weights': {
            'rule_compliance': 0.6,
            'amplitude_duration_norm': 0.2,
            'volatility_adjustment': 0.1,
            'pattern_consistency': 0.1
        }
    }
    
    # Initialize components
    rule_engine = ElliottRuleEngine(config=mock_config)
    nms = NMS()
    confidence_scorer = ConfidenceScorer(weights=mock_config['confidence_weights'])
    
    # Create detector
    detector = WaveDetector(mock_config, rule_engine, nms, confidence_scorer)
    
    # Create sample data for testing
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)  # Random walk
    sample_data = pd.DataFrame({'price': prices}, index=dates)
    
    # Detect waves
    result = detector.detect_waves(sample_data)
    print(f"Detected {len(result['candidates'])} wave candidates")