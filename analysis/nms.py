import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import datetime
import logging

logger = logging.getLogger(__name__)

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

        if start1 > end1 or start2 > end2:
            logger.error("Invalid segment: start time is after end time.")
            return 0.0

        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection_duration = max(0, (intersection_end - intersection_start).total_seconds())

        union_duration = max(0, (end1 - start1).total_seconds()) + max(0, (end2 - start2).total_seconds()) - intersection_duration

        if union_duration == 0:
            return 1.0 if intersection_duration > 0 else 0.0
        
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

        suppressed_indices = set()
        final_nms_segments = []

        for i in range(len(segments)):
            if i in suppressed_indices:
                continue

            current_segment = segments[i]
            
            for j in range(i + 1, len(segments)):
                if j in suppressed_indices:
                    continue

                next_segment = segments[j]

                if current_segment.get('level') == next_segment.get('level'):
                    iou = self._calculate_iou(current_segment, next_segment)
                    
                    # Simple and effective suppression logic:
                    # Suppress segment 'j' if:
                    # 1. IoU is above threshold AND
                    # 2. Current segment has higher confidence than next segment
                    if iou >= self.iou_threshold and current_segment.get('confidence', 0.0) > next_segment.get('confidence', 0.0):
                        logger.debug(f"NMS: Suppressing segment (index {j}, Level {next_segment.get('level')}) with confidence {next_segment.get('confidence'):.2f} due to overlap (IoU {iou:.2f}) with segment (index {i}, Confidence {current_segment.get('confidence'):.2f}).")
                        suppressed_indices.add(j)
        
        # Collect segments that were NOT suppressed
        for i in range(len(segments)):
            if i not in suppressed_indices:
                final_nms_segments.append(segments[i])
        
        return final_nms_segments

# --- Confidence Scoring ---

class ConfidenceScorer:
    """
    Computes confidence scores for wave segments and overall wave counts.
    Confidence is blended from various metrics.
    """
    def __init__(self, segment_weights: Dict[str, float], count_weights: Dict[str, float]):
        """
        Args:
            segment_weights (Dict[str, float]): Weights for scoring individual segments.
                                                e.g., {'rule_compliance': 0.6, 'amplitude_duration_norm': 0.2, ...}
            count_weights (Dict[str, float]): Weights for scoring overall wave counts.
                                              e.g., {'rule_compliance': 0.7, 'avg_segment_conf': 0.3}
        """
        self.segment_weights = self._normalize_weights(segment_weights)
        self.count_weights = self._normalize_weights(count_weights)
        logger.info(f"Initialized ConfidenceScorer with segment weights: {self.segment_weights}, count weights: {self.count_weights}")

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        total_weight = sum(weights.values())
        if not np.isclose(total_weight, 1.0) and total_weight > 0:
            logger.warning(f"Weights do not sum to 1.0 (sum={total_weight}). Normalizing.")
            return {k: v / total_weight for k, v in weights.items()}
        return weights
    
    def _normalize_amplitude_duration(self, segments: List[Dict], max_price_range: float, max_duration_seconds: float) -> List[Dict]:
        """
        Normalizes amplitude (price change) and duration for segments.
        """
        normalized_segments = []
        for segment in segments:
            segment = segment.copy()
            
            # Amplitude calculation: Ensure 'price_high' and 'price_low' are present or derived.
            # If only 'price' is available (e.g., from single point data), use it.
            price_high = segment.get('price_high', segment.get('price', 0))
            price_low = segment.get('price_low', segment.get('price', 0))
            amplitude = price_high - price_low
            
            norm_amplitude = amplitude / max_price_range if max_price_range > 0 else 0
            norm_amplitude = min(norm_amplitude, 1.0)

            duration_seconds = (segment['end'] - segment['start']).total_seconds()
            norm_duration = duration_seconds / max_duration_seconds if max_duration_seconds > 0 else 0
            norm_duration = min(norm_duration, 1.0)
            
            # Simple average for normalized amplitude and duration score
            segment['norm_amplitude_duration'] = (norm_amplitude + norm_duration) / 2.0
            normalized_segments.append(segment)
        return normalized_segments

    def _calculate_volatility_penalty(self, segments: List[Dict], volatility_metric: float) -> List[Dict]:
        """
        Applies a penalty based on market volatility. Higher volatility reduces confidence.
        Assumes volatility_metric is normalized (e.g., 0 to 1).
        """
        max_assumed_volatility = 0.5 # Assume a max volatility value that would halve confidence from this component
        # Score from 0 to 1, higher for lower volatility.
        penalty_score = max(0, 1 - (volatility_metric / max_assumed_volatility)) 
        
        for segment in segments:
            segment['volatility_penalty_score'] = penalty_score
        return segments
    
    def _calculate_chaos_metric(self, segments: List[Dict]) -> List[Dict]:
        """
        Placeholder for chaos metrics. Assigns a neutral score for now.
        """
        for segment in segments:
            segment['chaos_metric_score'] = 0.5 # Default neutral score
        return segments

    def score_segments(self, segments: List[Dict], price_data: pd.DataFrame, overall_volatility: float) -> List[Dict]:
        """
        Scores individual wave segments using defined weights.
        Args:
            segments: List of segment dictionaries (output from NMS).
            price_data: DataFrame with OHLCV data for context.
            overall_volatility: A float representing market volatility.
        Returns:
            List of segments with 'confidence' score added.
        """
        if not segments:
            return []

        # --- Pre-calculations for normalization ---
        max_price_range = price_data['high'].max() - price_data['low'].min() if not price_data.empty else 1.0
        if max_price_range <= 0: max_price_range = 1.0 # Avoid division by zero
        
        data_duration_seconds = (price_data.index.max() - price_data.index.min()).total_seconds() if not price_data.empty else 86400 * 180
        max_duration_seconds = max(data_duration_seconds, 86400 * 180) # Ensure at least 6 months in seconds

        # --- Apply normalization and scoring ---
        normalized_segments = self._normalize_amplitude_duration(segments, max_price_range, max_duration_seconds)
        volatility_penalized_segments = self._calculate_volatility_penalty(normalized_segments, overall_volatility)
        chaos_scored_segments = self._calculate_chaos_metric(volatility_penalized_segments)

        scored_segments = []
        for segment in chaos_scored_segments:
            segment_score = 0.0
            
            # Rule compliance score (comes from rule engine, default to 1.0 if not provided)
            rule_compliance = segment.get('rule_compliance', 1.0) 
            segment_score += self.segment_weights.get('rule_compliance', 0) * rule_compliance

            # Amplitude/Duration score
            segment_score += self.segment_weights.get('amplitude_duration_norm', 0) * segment.get('norm_amplitude_duration', 0.5)

            # Volatility penalty score
            segment_score += self.segment_weights.get('volatility_penalty', 0) * segment.get('volatility_penalty_score', 1.0)

            # Chaos metric score
            segment_score += self.segment_weights.get('chaos_metric', 0) * segment.get('chaos_metric_score', 0.5)
            
            segment['confidence'] = max(0.0, min(1.0, segment_score)) # Clamp confidence between 0 and 1
            scored_segments.append(segment)
        
        return scored_segments

    def score_wave_counts(self, wave_counts: List[Dict], scored_segments: List[Dict]) -> List[Dict]:
        """
        Scores overall wave count candidates by combining rule compliance and average segment confidence.
        Args:
            wave_counts: List of candidate wave count dictionaries. Expected format:
                         {'rank': int, 'description': str, 'segments': List[segment_id], 'rule_compliance_score': float}
            scored_segments: List of individual segments with their calculated confidence scores.
        Returns:
            List of wave counts, each with a 'confidence' score.
        """
        if not wave_counts or not scored_segments:
            return []

        segment_confidence_map = {seg['id']: seg['confidence'] for seg in scored_segments if 'id' in seg}
        scored_wave_counts = []

        for count in wave_counts:
            count_score_components = {}
            
            # Average segment confidence for this count
            segments_in_count = count.get('segments', [])
            valid_segment_confs = [segment_confidence_map[seg_id] for seg_id in segments_in_count if seg_id in segment_confidence_map]
            
            avg_segment_conf = np.mean(valid_segment_confs) if valid_segment_confs else 0.0
            count_score_components['avg_segment_conf'] = avg_segment_conf
            
            # Rule compliance for the entire pattern (obtained from rule engine)
            rule_compliance_count = count.get('rule_compliance_score', 1.0) # Default to 1.0
            count_score_components['rule_compliance'] = rule_compliance_count

            # Combine scores for the count using count_weights
            final_count_confidence = 0.0
            for key, weight in self.count_weights.items():
                if key in count_score_components:
                    final_count_confidence += weight * count_score_components[key]
            
            final_count_confidence = max(0.0, min(1.0, final_count_confidence))
            
            scored_wave_counts.append({
                'rank': count.get('rank'),
                'confidence': final_count_confidence,
                'description': count.get('description'),
                'wave_pattern': count.get('wave_pattern'), # Store the actual wave pattern structure if available
                'rule_compliance_score': rule_compliance_count # Also store the raw rule score
            })

        # Sort wave counts by confidence
        scored_wave_counts.sort(key=lambda x: x['confidence'], reverse=True)
        return scored_wave_counts

# --- Sample data generation for demonstration ---
def generate_sample_confidence_data(num_segments: int = 10, num_counts: int = 5) -> Tuple[List[Dict], List[Dict], pd.DataFrame, float]:
    """Generates mock data for testing ConfidenceScorer."""
    logger.info("Generating sample confidence data for NMS and ConfidenceScorer.")
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # Sample segments (output from NMS)
    segments = []
    for i in range(num_segments):
        # Simulate timestamps and durations
        start_offset_hours = (num_segments - i) * 2 # Make earlier segments longer/older
        start_time = now - datetime.timedelta(hours=start_offset_hours)
        # Simulate varying segment durations
        duration_minutes = 15 + i * 5 + np.random.randint(-5, 5)
        end_time = start_time + datetime.timedelta(minutes=duration_minutes)
        
        # Simulate values
        confidence = np.random.uniform(0.3, 0.9) # Raw confidence before scoring
        rule_compliance = np.random.choice([0.5, 0.8, 1.0], p=[0.2, 0.5, 0.3]) # Score from rule engine
        
        # Add price range for normalization
        price_low = 70000 + i * 10 + np.random.uniform(-10, 10)
        price_high = price_low + 50 + np.random.uniform(0, 30)

        segments.append({
            'id': f'seg_{i}', # Unique ID for segment
            'start': start_time,
            'end': end_time,
            'price_low': price_low,
            'price_high': price_high,
            'price': (price_low + price_high) / 2, # Mid-price for fallback
            'label': str(i+1), # Dummy label like '1', 'a'
            'type': 'impulse' if i % 2 == 0 else 'corrective',
            'level': 1, # Assuming level 1 for simplicity
            'rule_compliance': rule_compliance, # This would be assigned by the rule engine
            # 'norm_amplitude_duration', 'volatility_penalty_score', 'chaos_metric_score' will be added by scorer
        })

    # Mock price data for context (used in normalization)
    ts_index = pd.date_range(start=now - datetime.timedelta(hours=24), end=now, freq='1m', tz=datetime.timezone.utc)
    price_data = pd.DataFrame({
        'timestamp': ts_index,
        'open': 70000 + np.random.rand(len(ts_index))*500,
        'high': lambda df: df['open'] + np.random.rand(len(df))*100,
        'low': lambda df: df['open'] - np.random.rand(len(df))*100,
        'close': lambda df: df['open'] + np.random.rand(len(df))*50 - 25,
        'volume': 1000 + np.random.rand(len(df))*500
    })
    price_data['high'] = price_data.apply(lambda row: row['open'] + np.random.uniform(0, 100) if row['high'] < row['open'] else row['high'], axis=1)
    price_data['low'] = price_data.apply(lambda row: row['open'] - np.random.uniform(0, 100) if row['low'] > row['open'] else row['low'], axis=1)
    price_data['close'] = price_data.apply(lambda row: row['open'] + np.random.uniform(-50, 50) if row['close'] < row['low'] or row['close'] > row['high'] else row['close'], axis=1)
    
    price_data.set_index('timestamp', inplace=True)
    
    # Mock wave counts (output from rule engine)
    wave_counts = []
    for i in range(num_counts):
        wave_pattern_labels = ['1-2-3-4-5', 'a-b-c', 'W-X-Y', '1-2-3-4-5(ext)']
        # Assign a subset of generated segments to each wave count
        segments_for_count_ids = np.random.choice([seg['id'] for seg in segments], size=min(len(segments), 3+i), replace=False).tolist()
        
        wave_counts.append({
            'rank': i+1,
            'description': f'Candidate count {i+1}',
            'wave_pattern': { # Representing the structure of the count
                'label': np.random.choice(wave_pattern_labels),
                'segments': segments_for_count_ids # IDs of segments forming this count
            },
            'rule_compliance_score': np.random.choice([0.7, 0.9, 1.0], p=[0.3, 0.5, 0.2]) # Overall rule score for this count
        })
    
    overall_volatility = np.random.uniform(0.05, 0.3) # Example volatility score

    return segments, wave_counts, price_data, overall_volatility

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Load weights from a hypothetical config
    # In a real app, these would come from config.yaml
    config_weights = {
        'rule_compliance': 0.6,
        'amplitude_duration_norm': 0.2,
        'volatility_penalty': 0.1,
        'chaos_metric': 0.1,
    }
    
    # Segment scoring weights are the direct config weights
    segment_weights = {k: v for k, v in config_weights.items()}
    
    # Count scoring weights combine rule compliance and average segment confidence
    count_weights = {
        'rule_compliance': 0.7, # Higher weight for overall rule compliance of the count pattern
        'avg_segment_conf': 0.3, # Weight for the average confidence of its constituent segments
    }

    # Instantiate NMS and ConfidenceScorer
    nms_threshold = 0.6 # Example threshold
    nms_conf_weight = 0.8
    nms_module = NMS(iou_threshold=nms_threshold, confidence_weight=nms_conf_weight)
    
    scorer = ConfidenceScorer(segment_weights=segment_weights, count_weights=count_weights)

    # Generate mock data
    sample_segments, sample_wave_counts, sample_price_data, sample_volatility = generate_sample_confidence_data(num_segments=15, num_counts=7)

    # --- Test NMS ---
    print("\n--- Testing NMS ---")
    nms_applied_segments = nms_module.suppress(sample_segments)
    print(f"Original segments: {len(sample_segments)}, Segments after NMS: {len(nms_applied_segments)}")
    if len(sample_segments) > 0 and len(nms_applied_segments) < len(sample_segments):
        print("NMS successfully suppressed some overlapping segments.")
    elif len(sample_segments) == 0:
        print("No segments to test NMS.")
    else:
        print("NMS did not suppress any segments (may need tuning or data).")

    # --- Test Confidence Scoring ---
    print("\n--- Testing Confidence Scoring ---")
    
    # Score the segments that passed NMS
    scored_segments = scorer.score_segments(nms_applied_segments, sample_price_data, sample_volatility)
    print(f"\nScored {len(scored_segments)} segments. First 3 scored segments:")
    for seg in scored_segments[:3]:
        print(f"  ID: {seg.get('id')}, Level: {seg.get('level')}, Type: {seg.get('type')}, Confidence: {seg.get('confidence', 0.0):.3f}")

    # Score the wave counts using the scored segments
    scored_counts = scorer.score_wave_counts(sample_wave_counts, scored_segments)
    print(f"\nScored {len(scored_counts)} wave counts. Top 3 scored counts:")
    for count in scored_counts[:3]:
        print(f"  Rank: {count.get('rank')}, Description: {count.get('description')}, Confidence: {count.get('confidence', 0.0):.3f}, Rule Score: {count.get('rule_compliance_score', 1.0):.2f}")

    print("\nNMS and ConfidenceScorer modules created. Example usage demonstrated.")
    print("Dependencies: pandas, numpy")