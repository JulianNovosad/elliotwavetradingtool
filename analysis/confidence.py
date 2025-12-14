import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import datetime
import logging

logger = logging.getLogger(__name__)

# --- Non-Maximum Suppression (NMS) for Wave Segments ---

class NMS:
    """
    Implements Non-Maximum Suppression for wave segments.
    Suppresses overlapping wave candidates based on confidence and time interval overlap.
    """
    def __init__(self, iou_threshold: float = 0.5, confidence_weight: float = 0.7):
        """
        Args:
            iou_threshold (float): Intersection over Union threshold. Segments with IoU > threshold
                                   and lower confidence will be suppressed.
            confidence_weight (float): Weight given to confidence score for suppression decision.
                                       Higher weight means confidence is more critical.
        """
        self.iou_threshold = iou_threshold
        self.confidence_weight = confidence_weight
        logger.info(f"Initialized NMS with IoU threshold: {self.iou_threshold}, confidence weight: {self.confidence_weight}")

    def _calculate_iou(self, segment1: Dict, segment2: Dict) -> float:
        """
        Calculates the Intersection over Union (IoU) for two time segments.
        Segments are expected to have 'start' and 'end' timestamps (datetime objects).
        """
        start1, end1 = segment1['start'], segment1['end']
        start2, end2 = segment2['start'], segment2['end']

        # Ensure segments are valid
        if start1 > end1 or start2 > end2:
            logger.error("Invalid segment: start time is after end time.")
            return 0.0

        # Calculate intersection
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection_duration = max(0, (intersection_end - intersection_start).total_seconds())

        # Calculate union
        union_duration = max(0, (end1 - start1).total_seconds()) + max(0, (end2 - start2).total_seconds()) - intersection_duration

        if union_duration == 0:
            return 1.0 if intersection_duration > 0 else 0.0 # Both segments are identical and non-empty
        
        iou = intersection_duration / union_duration
        return iou

    def suppress(self, segments: List[Dict]) -> List[Dict]:
        """
        Applies NMS to a list of wave segments.
        Each segment is a dictionary with at least:
        'start': datetime.datetime (start time)
        'end': datetime.datetime (end time)
        'confidence': float (confidence score, 0.0 to 1.0)
        'label': str (e.g., '1', '2', 'a', 'b')
        'type': str (e.g., 'impulse', 'corrective')
        'level': int
        """
        if not segments:
            return []

        # Sort segments by confidence in descending order
        segments.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)

        # Keep track of suppressed segment indices
        suppressed_indices = set()
        
        # Result list for non-suppressed segments
        nms_segments = []

        for i in range(len(segments)):
            if i in suppressed_indices:
                continue # Skip if this segment has already been suppressed

            current_segment = segments[i]
            nms_segments.append(current_segment) # Add current segment to result

            # Compare current segment with all subsequent segments
            for j in range(i + 1, len(segments)):
                if j in suppressed_indices:
                    continue # Skip if already suppressed

                next_segment = segments[j]

                # Check if segments are on the same level and have significant overlap
                if current_segment.get('level') == next_segment.get('level'):
                    iou = self._calculate_iou(current_segment, next_segment)

                    # Suppression criteria:
                    # If IoU is high AND current_segment has significantly higher confidence
                    # (or if IoU is very high, even with similar confidence)
                    confidence_diff = current_segment.get('confidence', 0.0) - next_segment.get('confidence', 0.0)
                    
                    # Use a combined score for suppression logic
                    # A segment is suppressed if it overlaps significantly with a higher-confidence segment.
                    # The 'confidence_weight' determines how much the confidence difference matters relative to IoU.
                    # A simple condition: if IoU is high and confidence is also high, suppress lower confidence.
                    
                    suppression_condition = False
                    if iou >= self.iou_threshold:
                        # High overlap, now consider confidence
                        # If current confidence is much higher, suppress next
                        if confidence_diff > (1.0 - self.confidence_weight) * (1.0 - iou): # Heuristic: more weight to confidence when IoU is high
                             suppression_condition = True
                        # If IoU is very high, suppress regardless of confidence unless they are identical
                        elif iou > 0.8 and abs(confidence_diff) < 0.01: # Nearly identical segments, suppress if confidence is similar
                            suppression_condition = True

                    if suppression_condition:
                        logger.debug(f"Suppressing segment (index {j}) due to overlap with segment (index {i}). IoU: {iou:.2f}, Confidence diff: {confidence_diff:.2f}")
                        suppressed_indices.add(j)
        
        # Filter out suppressed segments
        # The suppressed_indices set contains indices of segments to be REMOVED.
        # We can reconstruct the nms_segments list more cleanly.
        
        # Let's re-collect from the original sorted list, skipping suppressed ones.
        final_nms_segments_reordered = []
        for i in range(len(segments)):
            if i not in suppressed_indices:
                final_nms_segments_reordered.append(segments[i])
        
        return final_nms_segments_reordered

# --- Confidence Scoring ---

class ConfidenceScorer:
    """
    Computes confidence scores for wave segments and overall wave counts.
    Confidence is blended from various metrics.
    """
    def __init__(self, weights: Dict[str, float]):
        """
        Args:
            weights (Dict[str, float]): Dictionary of weights for different metrics.
                                        Example: {'rule_compliance': 0.6, 'amplitude_duration_norm': 0.2, ...}
        """
        self.weights = weights
        # Ensure weights sum up to 1.0 (or normalize if they don't)
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"Confidence weights do not sum to 1.0 (sum={total_weight}). Normalizing.")
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

        logger.info(f"Initialized ConfidenceScorer with weights: {self.weights}")

    def _normalize_amplitude_duration(self, segments: List[Dict], max_price_range: float, max_duration_seconds: float) -> List[Dict]:
        """
        Normalizes amplitude (price change) and duration for segments.
        Higher amplitude/duration might imply higher confidence for impulse waves,
        but should be capped and considered in context (e.g., not overly long waves).
        """
        normalized_segments = []
        for segment in segments:
            segment = segment.copy()
            
            # Amplitude normalization (price range of the segment)
            amplitude = segment['price_high'] - segment['price_low'] if 'price_high' in segment and 'price_low' in segment else 0
            norm_amplitude = amplitude / max_price_range if max_price_range > 0 else 0
            norm_amplitude = min(norm_amplitude, 1.0) # Cap at 1.0

            # Duration normalization
            duration_seconds = (segment['end'] - segment['start']).total_seconds()
            norm_duration = duration_seconds / max_duration_seconds if max_duration_seconds > 0 else 0
            norm_duration = min(norm_duration, 1.0) # Cap at 1.0

            # Combine them into a single normalized score (e.g., simple average or weighted)
            # This could be more sophisticated, e.g., considering wave type.
            # For impulse waves, higher amplitude/duration might be good, for corrective, shorter is typical.
            # For now, a simple average for demonstration.
            segment['norm_amplitude_duration'] = (norm_amplitude + norm_duration) / 2.0
            normalized_segments.append(segment)
        return normalized_segments

    def _calculate_volatility_penalty(self, segments: List[Dict], volatility_metric: float) -> List[Dict]:
        """
        Applies a penalty based on market volatility.
        Higher volatility might reduce confidence in clear wave formation.
        """
        # volatility_metric is a single value representing overall market volatility (e.g., ATR, std dev).
        # Lower volatility -> higher confidence. Higher volatility -> lower confidence.
        # We want a penalty that *reduces* score as volatility increases.
        # Example: penalty = 1 - (volatility_metric / max_expected_volatility)
        # Or a simpler inverse relationship.
        
        # For now, let's assume volatility_metric is normalized between 0 and 1.
        # penalty_factor = max(0, 1 - volatility_metric) # 1 if vol=0, 0 if vol=1
        # A simple approach: higher volatility reduces confidence linearly.
        # Let's say max_volatility corresponds to a 0.5 confidence reduction.
        max_assumed_volatility = 0.5 # Assume a max volatility value that would halve confidence from this component
        penalty_score = max(0, 1 - (volatility_metric / max_assumed_volatility)) # Score from 0 to 1, higher for lower volatility
        
        for segment in segments:
            segment['volatility_penalty_score'] = penalty_score
        return segments

    def _calculate_chaos_metric(self, segments: List[Dict]) -> List[Dict]:
        """
        Placeholder for chaos metrics (e.g., from nonlinear_time_series.pdf).
        This is complex and usually requires specialized libraries or calculations.
        For now, we'll assign a neutral score or a dummy value.
        """
        # Example: If chaos metric indicates strong trend/order, confidence might increase.
        # If metric indicates high randomness, confidence might decrease.
        # Assign a dummy value (e.g., 0.5) for now.
        for segment in segments:
            segment['chaos_metric_score'] = 0.5 # Default neutral score
        return segments

    def score_segments(self, segments: List[Dict], price_data: pd.DataFrame, raw_wave_counts: List[Dict], overall_volatility: float) -> List[Dict]:
        """
        Scores each individual wave segment and the overall wave counts.
        `segments` should be the output from NMS.
        `price_data` is the raw OHLCV DataFrame for context.
        `raw_wave_counts` are candidate wave counts (e.g., top 3).
        `overall_volatility` is a metric for market volatility.
        """
        if not segments:
            return []

        # --- Pre-calculations for normalization ---
        # Max price range across all data for normalization
        max_price_range = price_data['high'].max() - price_data['low'].min() if not price_data.empty else 1.0
        if max_price_range == 0: max_price_range = 1.0
        
        # Max duration across all data for normalization (e.g., longest wave expected)
        # This should be derived from config or expected wave durations.
        # Let's use a rough estimate based on typical trading days.
        # For 1m data, max duration for a wave like '1w' is ~10000 minutes.
        # For 1h data, max duration for '6 months' is ~4320 hours.
        # This needs to be adaptive based on resolution.
        # For now, let's use a large fixed value for demonstration, or infer from data.
        data_duration_seconds = (price_data.index.max() - price_data.index.min()).total_seconds() if not price_data.empty else 86400 * 30 # Default to 30 days if empty
        max_duration_seconds = max(data_duration_seconds, 86400 * 180) # At least 6 months of data in seconds

        # --- Normalize segments ---
        normalized_segments = self._normalize_amplitude_duration(segments, max_price_range, max_duration_seconds)
        
        # --- Add other scores ---
        # Volatility penalty
        volatility_penalized_segments = self._calculate_volatility_penalty(normalized_segments, overall_volatility)
        
        # Chaos metric (placeholder)
        chaos_scored_segments = self._calculate_chaos_metric(volatility_penalized_segments)

        # --- Score individual segments ---
        scored_segments = []
        for segment in chaos_scored_segments:
            segment_score = 0.0
            
            # Rule compliance score (this would be determined by a rule engine component)
            # For now, let's assume a placeholder: 1.0 if rules are followed, 0.5 otherwise.
            # This needs to be integrated with the actual rule checking.
            rule_compliance = segment.get('rule_compliance', 1.0) # Default to 1.0 if not set by rule engine
            segment_score += self.weights.get('rule_compliance', 0) * rule_compliance

            # Amplitude/Duration score
            norm_amp_dur = segment.get('norm_amplitude_duration', 0.5) # Default to 0.5 if not calculated
            segment_score += self.weights.get('amplitude_duration_norm', 0) * norm_amp_dur

            # Volatility penalty score
            vol_penalty = segment.get('volatility_penalty_score', 1.0) # Default to 1.0 (no penalty)
            segment_score += self.weights.get('volatility_penalty', 0) * vol_penalty

            # Chaos metric score
            chaos_score = segment.get('chaos_metric_score', 0.5) # Default to 0.5
            segment_score += self.weights.get('chaos_metric', 0) * chaos_score

            segment['confidence'] = max(0.0, min(1.0, segment_score)) # Clamp confidence between 0 and 1
            scored_segments.append(segment)
        
        # --- Score overall wave counts ---
        # For each candidate wave count, aggregate the confidence of its constituent segments.
        # This part is more complex and depends on how 'raw_wave_counts' are structured.
        # Assuming raw_wave_counts is a list of dictionaries, each representing a count.
        # e.g., `{'rank': 1, 'description': 'primary', 'segments': [segment_id1, segment_id2, ...]}`
        # We need to map segment IDs to their calculated confidence scores.
        
        scored_wave_counts = []
        segment_confidence_map = {seg['id']: seg['confidence'] for seg in scored_segments if 'id' in seg} # Assuming segments have unique IDs
        
        for count in raw_wave_counts:
            count_score = 0.0
            valid_segments_count = 0
            
            # Check if 'segments' key exists and is a list of segment identifiers
            if 'segments' in count and isinstance(count['segments'], list):
                for segment_id in count['segments']:
                    if segment_id in segment_confidence_map:
                        count_score += segment_confidence_map[segment_id]
                        valid_segments_count += 1
            
            # Average confidence across segments in the count
            if valid_segments_count > 0:
                avg_confidence = count_score / valid_segments_count
            else:
                avg_confidence = 0.0 # No valid segments found for this count
            
            # Blend rule compliance for the *entire count*
            # This would require a rule engine to evaluate entire wave sequences.
            # For now, use a placeholder.
            rule_compliance_count = count.get('rule_compliance_score', 1.0) # Placeholder
            
            # Combine scores for the count
            final_count_confidence = (self.weights.get('rule_compliance', 0) * rule_compliance_count +
                                      self.weights.get('segment_average_confidence', 0) * avg_confidence) # Assuming a weight for average segment confidence
            
            final_count_confidence = max(0.0, min(1.0, final_count_confidence))
            
            scored_wave_counts.append({
                'rank': count.get('rank'),
                'confidence': final_count_confidence,
                'description': count.get('description'),
                'wave_pattern': count.get('wave_pattern') # Store the actual wave pattern if available
            })

        # Sort wave counts by confidence
        scored_wave_counts.sort(key=lambda x: x['confidence'], reverse=True)

        return scored_segments, scored_wave_counts

# --- Helper function for sample confidence data generation ---
def generate_sample_confidence_data(num_segments: int = 5, num_counts: int = 3) -> Tuple[List[Dict], List[Dict], pd.DataFrame, float]:
    """Generates mock data for testing ConfidenceScorer."""
    logger.info("Generating sample confidence data...")
    now = datetime.datetime.now(datetime.timezone.utc)
    segments = []
    for i in range(num_segments):
        start_time = now - datetime.timedelta(hours=num_segments - i)
        end_time = start_time + datetime.timedelta(minutes=15 + i*5)
        confidence = np.random.uniform(0.3, 0.9)
        rule_compliance = np.random.choice([0.5, 0.8, 1.0], p=[0.2, 0.5, 0.3])
        
        segments.append({
            'id': f'seg_{i}',
            'start': start_time,
            'end': end_time,
            'price_low': 70000 + i*10,
            'price_high': 70100 + i*10 + 50,
            'label': str(i+1), # Dummy label
            'type': 'impulse' if i % 2 == 0 else 'corrective',
            'level': 1,
            'rule_compliance': rule_compliance, # Value from rule engine
            'norm_amplitude_duration': np.random.uniform(0.4, 0.8), # Pre-normalized values
            'volatility_penalty_score': np.random.uniform(0.7, 1.0),
            'chaos_metric_score': np.random.uniform(0.3, 0.7),
            # 'confidence' will be calculated by the scorer
        })

    # Mock price data for normalization context
    ts = pd.date_range(start=now - datetime.timedelta(hours=20), end=now, freq='1m', tz=datetime.timezone.utc)
    price_data = pd.DataFrame({
        'timestamp': ts,
        'open': 70000 + np.random.rand(len(ts))*200,
        'high': 70100 + np.random.rand(len(ts))*200 + 50,
        'low': 69900 + np.random.rand(len(ts))*200,
        'close': 70050 + np.random.rand(len(ts))*200,
        'volume': 1000 + np.random.rand(len(ts))*500
    })
    price_data.set_index('timestamp', inplace=True)

    # Mock wave counts
    wave_counts = []
    for i in range(num_counts):
        wave_pattern_labels = ['1-2-3-4-5', 'a-b-c', 'W-X-Y'] # Example patterns
        segments_for_count = np.random.choice([seg['id'] for seg in segments], size=min(len(segments), 3+i), replace=False).tolist()
        wave_counts.append({
            'rank': i+1,
            'description': f'Candidate {i+1}',
            'segments': segments_for_count,
            'rule_compliance_score': np.random.choice([0.7, 0.9, 1.0], p=[0.3, 0.5, 0.2]) # Rule compliance for entire pattern
        })
    
    overall_volatility = np.random.uniform(0.05, 0.3) # Example volatility score

    return segments, wave_counts, price_data, overall_volatility

if __name__ == '__main__':
    # Example usage for ConfidenceScorer
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Define weights based on config.yaml
    # Let's assume config has a 'confidence_weights' dict
    config_weights = {
        'rule_compliance': 0.6,
        'amplitude_duration_norm': 0.2,
        'volatility_penalty': 0.1,
        'chaos_metric': 0.1,
    }
    
    # Separate weights for segment scoring and overall count scoring
    segment_weights = {k: v for k, v in config_weights.items()}
    # For count scoring, we use rule compliance and average segment confidence
    count_weights = {
        'rule_compliance': 0.7, # Major weight for overall rule compliance
        'avg_segment_conf': 0.3, # Weight for the average confidence of its segments
    }

    scorer = ConfidenceScorerAdjusted(segment_weights=segment_weights, count_weights=count_weights)

    # Generate mock data
    sample_segments, sample_wave_counts, sample_price_data, sample_volatility = generate_sample_confidence_data(num_segments=10, num_counts=5)

    # Score segments
    scored_segments = scorer.score_segments(sample_segments, sample_price_data, sample_volatility)
    print(f"\n--- Scored Segments ({len(scored_segments)}) ---")
    for seg in scored_segments[:3]: # Print first 3 scored segments
        print(f"  ID: {seg['id']}, Level: {seg['level']}, Type: {seg['type']}, Confidence: {seg['confidence']:.3f}")

    # Score wave counts
    scored_counts = scorer.score_wave_counts(sample_wave_counts, scored_segments)
    print(f"\n--- Scored Wave Counts ({len(scored_counts)}) ---")
    for count in scored_counts:
        print(f"  Rank: {count['rank']}, Description: {count['description']}, Confidence: {count['confidence']:.3f}, Pattern: {count['wave_pattern']}")

    print("\nConfidenceScorer module created. Example usage demonstrated.")
    print("To use, instantiate ConfidenceScorer with your weights and call score_segments/score_wave_counts.")
