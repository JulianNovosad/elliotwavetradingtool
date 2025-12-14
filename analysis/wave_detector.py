import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import datetime
import logging

# Import technical indicators module
from .indicators import TechnicalIndicators
# Import ML predictor module
from .ml_predictor import MLPredictor

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
                    w1_direction_up = wave1_seg.get('price', 0) > wave1_seg.get('price', 0) # Simplified check, need actual price comparison
                    if wave1_seg.get('price_high') is not None and wave1_seg.get('price_low') is not None:
                        w1_direction_up = wave1_seg['price_high'] > wave1_seg['price_low']

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
        Detects potential wave segments in price data.
        This is a simplified implementation that generates mock segments.
        In a real implementation, this would use advanced pattern recognition.
        
        Args:
            price_data: DataFrame with timestamp index and price columns
            
        Returns:
            List of potential wave segments
        """
        if price_data.empty:
            return []
            
        segments = []
        now = datetime.datetime.now(datetime.timezone.utc)
        
        # For demonstration, generate some mock segments
        # In a real implementation, this would use technical analysis
        num_segments = min(10, len(price_data) // 10)  # Roughly one segment per 10 data points
        
        for i in range(num_segments):
            # Generate timestamps
            start_idx = i * (len(price_data) // num_segments)
            end_idx = min((i + 1) * (len(price_data) // num_segments) - 1, len(price_data) - 1)
            
            if start_idx >= len(price_data) or end_idx >= len(price_data):
                continue
                
            start_time = price_data.index[start_idx]
            end_time = price_data.index[end_idx]
            
            # Get price data for this segment
            segment_data = price_data.iloc[start_idx:end_idx+1]
            if segment_data.empty:
                continue
                
            price_low = segment_data['price'].min() if 'price' in segment_data.columns else segment_data.iloc[:, 1].min()
            price_high = segment_data['price'].max() if 'price' in segment_data.columns else segment_data.iloc[:, 1].max()
            start_price = segment_data['price'].iloc[0] if 'price' in segment_data.columns else segment_data.iloc[0, 1]
            end_price = segment_data['price'].iloc[-1] if 'price' in segment_data.columns else segment_data.iloc[-1, 1]
            
            # Determine wave type based on price movement
            price_change = end_price - start_price
            wave_type = 'impulse' if abs(price_change) > (price_high - price_low) * 0.3 else 'corrective'
            wave_label = str((i % 5) + 1) if wave_type == 'impulse' else chr(ord('a') + (i % 3))
            
            segments.append({
                'id': f'seg_{i}_{int(now.timestamp() % 10000)}',
                'start': start_time,
                'end': end_time,
                'price_low': float(price_low),
                'price_high': float(price_high),
                'price': float(end_price),
                'label': wave_label,
                'type': wave_type,
                'level': 1,
                'confidence': np.random.uniform(0.5, 0.9)  # Initial confidence
            })
            
        logger.info(f"Detected {len(segments)} potential wave segments")
        return segments

    def _generate_wave_candidates(self, segments: List[Dict]) -> List[Dict]:
        """
        Generates candidate wave counts from detected segments.
        
        Args:
            segments: List of detected wave segments
            
        Returns:
            List of candidate wave counts
        """
        if not segments:
            return []
            
        candidates = []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        # Generate a few candidate wave counts
        # In a real implementation, this would use more sophisticated pattern matching
        for i in range(min(3, len(sorted_segments))):
            # Create a 5-wave impulse pattern candidate
            if len(sorted_segments) >= i + 5:
                impulse_segments = sorted_segments[i:i+5]
                candidate = {
                    'rank': i + 1,
                    'description': f'Impulse wave count candidate {i+1}',
                    'wave_pattern': {
                        'label': '1-2-3-4-5',
                        'segments': [seg['id'] for seg in impulse_segments]
                    },
                    'rule_compliance_score': 1.0  # Will be updated by rule engine
                }
                candidates.append(candidate)
            
            # Create a corrective pattern candidate
            if len(sorted_segments) >= i + 3:
                corrective_segments = sorted_segments[i:i+3]
                candidate = {
                    'rank': i + 4,
                    'description': f'Corrective wave count candidate {i+1}',
                    'wave_pattern': {
                        'label': 'a-b-c',
                        'segments': [seg['id'] for seg in corrective_segments]
                    },
                    'rule_compliance_score': 1.0  # Will be updated by rule engine
                }
                candidates.append(candidate)
        
        logger.info(f"Generated {len(candidates)} wave count candidates")
        return candidates

    def detect_waves(self, price_data: pd.DataFrame) -> Dict:
        """
        Main method to detect Elliott Wave patterns in price data.
        
        Args:
            price_data: DataFrame with timestamp index and price data
            
        Returns:
            Dictionary containing detected wave patterns, candidates, and confidence scores
        """
        logger.info("Starting wave detection process")
        
        # Step 1: Detect potential wave segments
        segments = self._detect_wave_segments(price_data)
        if not segments:
            logger.warning("No wave segments detected")
            return {
                'wave_levels': [],
                'candidates': [],
                'predictions': []
            }
        
        # Step 2: Apply rule engine to score segments
        logger.info("Applying rule engine to segments")
        scored_segments = self.rule_engine.evaluate_segments(segments)
        
        # Step 3: Apply Non-Maximum Suppression to remove overlapping segments
        logger.info("Applying Non-Maximum Suppression")
        nms_segments = self.nms.suppress(scored_segments)
        
        # Step 4: Generate candidate wave counts
        logger.info("Generating wave count candidates")
        candidates = self._generate_wave_candidates(nms_segments)
        
        # Step 5: Apply rule engine to score candidates
        logger.info("Applying rule engine to candidates")
        scored_candidates = []
        for candidate in candidates:
            scored_candidate = self.rule_engine.evaluate_wave_count(candidate, nms_segments)
            scored_candidates.append(scored_candidate)
        
        # Step 6: Score segments and candidates with confidence scorer
        logger.info("Calculating confidence scores")
        try:
            # Score segments
            final_scored_segments = self.confidence_scorer.score_segments(
                nms_segments, 
                price_data, 
                overall_volatility=0.1  # Placeholder volatility
            )
            
            # Score candidates
            final_scored_candidates = self.confidence_scorer.score_wave_counts(
                scored_candidates, 
                final_scored_segments
            )
        except Exception as e:
            logger.error(f"Error in confidence scoring: {e}")
            # Fallback to rule engine scores only
            final_scored_segments = nms_segments
            final_scored_candidates = scored_candidates
        
        # Step 7: Sort candidates by confidence
        final_scored_candidates.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Step 8: Generate predictions (top candidates)
        top_candidates = final_scored_candidates[:self.config.get('max_candidate_counts', 3)]
        
        # Step 9: Calculate technical indicators
        logger.info("Calculating technical indicators")
        try:
            # Ensure we have the required columns for technical indicators
            if all(col in price_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                tech_indicators = self.tech_indicators.calculate_all_indicators(price_data)
            else:
                # If we don't have all required columns, create a minimal dataset
                # This is a fallback for cases where we only have close prices
                logger.warning("Missing required columns for technical indicators, using close prices only")
                tech_indicators = {}
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            tech_indicators = {}
        
        # Step 10: Generate ML predictions
        logger.info("Generating ML predictions")
        ml_predictions = []
        try:
            if tech_indicators and len(price_data) > 10:  # Need sufficient data for ML
                # Prepare features for ML model
                features = self.ml_predictor.prepare_features(price_data, tech_indicators)
                
                if len(features) > 0:
                    # Make predictions
                    predictions, probabilities = self.ml_predictor.predict(features.tail(1))  # Predict for latest data point
                    
                    if len(predictions) > 0:
                        ml_predictions = [{
                            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            'prediction': int(predictions[0]),
                            'probability_up': float(probabilities[0][1]) if len(probabilities[0]) > 1 else 0.0,
                            'confidence': float(max(probabilities[0])) if len(probabilities) > 0 else 0.0
                        }]
        except Exception as e:
            logger.error(f"Error generating ML predictions: {e}")
        
        result = {
            'wave_levels': [{
                'level': 1,
                'segments': final_scored_segments
            }],
            'candidates': top_candidates,
            'predictions': [{
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'symbol': 'N/A',  # Will be filled by caller
                'wave_count': candidate,
                'confidence': candidate.get('confidence', 0)
            } for candidate in top_candidates],
            'technical_indicators': tech_indicators,
            'ml_predictions': ml_predictions
        }
        
        logger.info(f"Wave detection complete. Found {len(final_scored_segments)} segments and {len(top_candidates)} candidates")
        return result


# --- Dummy data generation for testing ---
def generate_sample_rule_data(num_segments: int = 10) -> Tuple[List[Dict], List[Dict]]:
    """Generates mock segments and wave counts for rule evaluation."""
    logger.info("Generating sample rule data...")
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # Generate sample segments, simulating different types and labels
    segments = []
    for i in range(num_segments):
        start_time = now - datetime.timedelta(hours=num_segments - i)
        duration_minutes = 10 + i * 5 + np.random.randint(-5, 5) # Vary durations
        end_time = start_time + datetime.timedelta(minutes=duration_minutes)
        
        # Simulate price data for segments
        price_low = 70000 + i * 10 + np.random.uniform(-20, 20)
        price_high = price_low + 30 + np.random.uniform(0, 20)
        price = (price_low + price_high) / 2

        # Assign labels and types (simulating detection output)
        if i == 0: label, seg_type, level = '1', 'impulse', 1
        elif i == 1: label, seg_type, level = '2', 'corrective', 1
        elif i == 2: label, seg_type, level = '3', 'impulse', 1
        elif i == 3: label, seg_type, level = '4', 'corrective', 1
        elif i == 4: label, seg_type, level = '5', 'impulse', 1
        elif i == 5: label, seg_type, level = 'a', 'impulse', 2 # Wave 'a' of level 2
        elif i == 6: label, seg_type, level = 'b', 'corrective', 2
        elif i == 7: label, seg_type, level = 'c', 'impulse', 2
        elif i == 8: label, seg_type, level = '1', 'impulse', 2 # Another Wave 1 at level 2
        else: label, seg_type, level = 'ext', 'unknown', 0 # Generic/unknown

        segments.append({
            'id': f'seg_{i}',
            'start': start_time,
            'end': end_time,
            'price_low': price_low,
            'price_high': price_high,
            'price': price,
            'label': label,
            'type': seg_type,
            'level': level,
            # 'rule_compliance' will be added by the engine.
            # 'rule_compliance_score' for wave counts will also be added.
        })

    # Generate sample wave counts
    wave_counts = []
    # Count 1: A valid 5-wave impulse (1-2-3-4-5 at level 1)
    impulse_segments_ids = [s['id'] for s in segments if s['level'] == 1 and s['label'] in ['1', '2', '3', '4', '5']]
    if len(impulse_segments_ids) >= 5: # Ensure enough segments exist
        wave_counts.append({
            'rank': 1,
            'description': 'Primary impulse count',
            'wave_pattern': {'label': '1-2-3-4-5', 'segments': impulse_segments_ids},
            'rule_compliance_score': 1.0 # Placeholder, will be evaluated
        })
    
    # Count 2: A corrective count (a-b-c at level 2)
    corrective_segments_ids = [s['id'] for s in segments if s['level'] == 2 and s['label'] in ['a', 'b', 'c']]
    if len(corrective_segments_ids) >= 3:
        wave_counts.append({
            'rank': 2,
            'description': 'Alternate corrective count',
            'wave_pattern': {'label': 'a-b-c', 'segments': corrective_segments_ids},
            'rule_compliance_score': 1.0 # Placeholder
        })

    # Count 3: A potentially invalid count (e.g., short duration waves, overlap)
    # Let's create a short duration segment or one that might violate rules
    short_seg_id = 'seg_short_dur'
    segments.append({
        'id': short_seg_id,
        'start': now - datetime.timedelta(seconds=30), # Too short
        'end': now - datetime.timedelta(seconds=10),
        'price_low': 71000, 'price_high': 71050, 'price': 71025,
        'label': '1', 'type': 'impulse', 'level': 1,
    })
    impulse_segments_ids_with_short = impulse_segments_ids + [short_seg_id] if len(impulse_segments_ids) >= 4 else [short_seg_id]
    # Ensure order for duration check
    impulse_segments_ids_with_short.sort(key=lambda seg_id: next(s['start'] for s in segments if s['id'] == seg_id))

    wave_counts.append({
        'rank': 3,
        'description': 'Invalid: Short duration wave 1',
        'wave_pattern': {'label': '1-2-3-4-5', 'segments': impulse_segments_ids_with_short},
        'rule_compliance_score': 1.0 # Placeholder
    })
    
    # Count 4: A count that might violate the Wave 4/1 overlap rule
    # Manually create segments that might cause overlap for testing
    overlap_segments = [
        {'id': 'w1_ov', 'start': now - datetime.timedelta(hours=5), 'end': now - datetime.timedelta(hours=4), 'price_low': 70000, 'price_high': 70100, 'price': 70050, 'label': '1', 'type': 'impulse', 'level': 1},
        {'id': 'w2_ov', 'start': now - datetime.timedelta(hours=4), 'end': now - datetime.timedelta(hours=3.5), 'price_low': 70080, 'price_high': 70050, 'price': 70065, 'label': '2', 'type': 'corrective', 'level': 1},
        {'id': 'w3_ov', 'start': now - datetime.timedelta(hours=3.5), 'end': now - datetime.timedelta(hours=2), 'price_low': 70020, 'price_high': 70300, 'price': 70180, 'label': '3', 'type': 'impulse', 'level': 1},
        {'id': 'w4_ov', 'start': now - datetime.timedelta(hours=2), 'end': now - datetime.timedelta(hours=1), 'price_low': 70050, 'price_high': 70120, 'label': '4', 'type': 'corrective', 'level': 1}, # Wave 4 high (70120) overlaps Wave 1 high (70100)
        {'id': 'w5_ov', 'start': now - datetime.timedelta(hours=1), 'end': now, 'price_low': 70110, 'price_high': 70250, 'price': 70180, 'label': '5', 'type': 'impulse', 'level': 1},
    ]
    all_segments_for_overlap_count = segments + overlap_segments
    overlap_segment_ids = [s['id'] for s in overlap_segments]

    wave_counts.append({
        'rank': 4,
        'description': 'Invalid: Wave 4 overlaps Wave 1',
        'wave_pattern': {'label': '1-2-3-4-5', 'segments': overlap_segment_ids},
        'rule_compliance_score': 1.0 # Placeholder
    })
    
    return segments, wave_counts


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Load config (or use defaults)
    # This example assumes config is available, e.g., from config.yaml
    # For testing, we can hardcode or mock it.
    mock_config = {
        'min_wave_duration_seconds': 60, # "one wave per minute"
        'max_wave_duration_days': 7,     # "one wave per week"
        'elliott_rule_strictness': 'moderate',
    }

    engine = ElliottRuleEngine(config=mock_config)

    # Generate sample data
    sample_segments, sample_wave_counts = generate_sample_rule_data(num_segments=10)

    # --- Evaluate individual segments ---
    print("\n--- Evaluating Segments ---")
    evaluated_segments = engine.evaluate_segments(sample_segments)
    for seg in evaluated_segments:
        print(f"Segment ID: {seg['id']}, Label: {seg['label']}, Type: {seg['type']}, Level: {seg['level']}, Duration: {(seg['end'] - seg['start']).total_seconds():.0f}s, Rule Compliance: {seg['rule_compliance']:.1f}")

    # --- Evaluate wave counts ---
    print("\n--- Evaluating Wave Counts ---")
    # We need to pass the full list of segments to evaluate_wave_count because it might need to find specific segments by ID.
    for count in sample_wave_counts:
        evaluated_count = engine.evaluate_wave_count(count, sample_segments) # Pass all segments found so far
        print(f"\nWave Count Rank: {evaluated_count['rank']}, Desc: {evaluated_count['description']}, Pattern: {evaluated_count['wave_pattern']['label']}")
        print(f"  Raw Rule Compliance Score: {evaluated_count['rule_compliance_score']:.1f}")
        # If rule violations were stored:
        # if 'rule_violations' in evaluated_count:
        #     print("  Violations:")
        #     for violation in evaluated_count['rule_violations']:
        #         print(f"    - {violation['rule']}: {violation['message']}")

    print("\nElliottRuleEngine module created. Example usage demonstrated.")
    print("Dependencies: pandas, numpy, datetime")