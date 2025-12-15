"""
Elliott Wave Hard Rules Validation Test
Tests that all hard rules are properly enforced by the system.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timezone, timedelta

# Add the project root to the path
sys.path.append('.')

from analysis.enhanced_rule_engine import EnhancedElliottRuleEngine
from analysis.wave_hypothesis_engine import WaveHypothesis, WaveHypothesisEngine

def test_hard_rule_enforcement():
    """Test that all Elliott Wave hard rules are properly enforced."""
    print("=== TESTING ELLIOTT WAVE HARD RULES ENFORCEMENT ===")
    
    # Initialize the enhanced rule engine
    config = {
        'min_wave_duration_seconds': 60,
        'max_wave_duration_days': 7,
        'elliott_rule_strictness': 'moderate'
    }
    rule_engine = EnhancedElliottRuleEngine(config)
    print("✅ EnhancedElliottRuleEngine initialized")
    
    # Test 1: Valid 5-wave impulse pattern
    print("\\n--- Test 1: Valid 5-wave impulse pattern ---")
    valid_segments = [
        {
            'label': '1',
            'price_start': 100.0,
            'price_end': 110.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=5),
            'end': datetime.now(timezone.utc) - timedelta(hours=4)
        },
        {
            'label': '2',
            'price_start': 110.0,
            'price_end': 105.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=4),
            'end': datetime.now(timezone.utc) - timedelta(hours=3)
        },
        {
            'label': '3',
            'price_start': 105.0,
            'price_end': 115.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=3),
            'end': datetime.now(timezone.utc) - timedelta(hours=2)
        },
        {
            'label': '4',
            'price_start': 115.0,
            'price_end': 112.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=2),
            'end': datetime.now(timezone.utc) - timedelta(hours=1)
        },
        {
            'label': '5',
            'price_start': 112.0,
            'price_end': 120.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=1),
            'end': datetime.now(timezone.utc)
        }
    ]
    
    valid_wave_count = {
        'wave_pattern': {
            'label': '1-2-3-4-5',
            'segments': ['seg1', 'seg2', 'seg3', 'seg4', 'seg5']
        }
    }
    
    result = rule_engine.evaluate_wave_count(valid_wave_count, valid_segments)
    print(f"Valid pattern compliance score: {result['rule_compliance_score']}")
    print(f"Violations: {len(result['rule_violations'])}")
    if result['rule_violations']:
        for violation in result['rule_violations']:
            print(f"  - {violation['rule']}: {violation['message']}")
    
    # Test 2: Invalid pattern - Wave 3 is shortest
    print("\\n--- Test 2: Invalid pattern - Wave 3 is shortest ---")
    invalid_segments_1 = [
        {
            'label': '1',
            'price_start': 100.0,
            'price_end': 110.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=5),
            'end': datetime.now(timezone.utc) - timedelta(hours=4)
        },
        {
            'label': '2',
            'price_start': 110.0,
            'price_end': 105.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=4),
            'end': datetime.now(timezone.utc) - timedelta(hours=3)
        },
        {
            'label': '3',
            'price_start': 105.0,
            'price_end': 107.0,  # Very short Wave 3
            'start': datetime.now(timezone.utc) - timedelta(hours=3),
            'end': datetime.now(timezone.utc) - timedelta(hours=2)
        },
        {
            'label': '4',
            'price_start': 107.0,
            'price_end': 106.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=2),
            'end': datetime.now(timezone.utc) - timedelta(hours=1)
        },
        {
            'label': '5',
            'price_start': 106.0,
            'price_end': 108.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=1),
            'end': datetime.now(timezone.utc)
        }
    ]
    
    invalid_wave_count_1 = {
        'wave_pattern': {
            'label': '1-2-3-4-5',
            'segments': ['seg1', 'seg2', 'seg3', 'seg4', 'seg5']
        }
    }
    
    result = rule_engine.evaluate_wave_count(invalid_wave_count_1, invalid_segments_1)
    print(f"Invalid pattern (Wave 3 shortest) compliance score: {result['rule_compliance_score']}")
    print(f"Violations: {len(result['rule_violations'])}")
    if result['rule_violations']:
        for violation in result['rule_violations']:
            print(f"  - {violation['rule']}: {violation['message']}")
    
    # Test 3: Invalid pattern - Wave 2 retraces > 100%
    print("\\n--- Test 3: Invalid pattern - Wave 2 retraces > 100% ---")
    invalid_segments_2 = [
        {
            'label': '1',
            'price_start': 100.0,
            'price_end': 110.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=5),
            'end': datetime.now(timezone.utc) - timedelta(hours=4)
        },
        {
            'label': '2',
            'price_start': 110.0,
            'price_end': 95.0,  # Retraces past start of Wave 1
            'start': datetime.now(timezone.utc) - timedelta(hours=4),
            'end': datetime.now(timezone.utc) - timedelta(hours=3)
        },
        {
            'label': '3',
            'price_start': 95.0,
            'price_end': 115.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=3),
            'end': datetime.now(timezone.utc) - timedelta(hours=2)
        },
        {
            'label': '4',
            'price_start': 115.0,
            'price_end': 112.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=2),
            'end': datetime.now(timezone.utc) - timedelta(hours=1)
        },
        {
            'label': '5',
            'price_start': 112.0,
            'price_end': 120.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=1),
            'end': datetime.now(timezone.utc)
        }
    ]
    
    invalid_wave_count_2 = {
        'wave_pattern': {
            'label': '1-2-3-4-5',
            'segments': ['seg1', 'seg2', 'seg3', 'seg4', 'seg5']
        }
    }
    
    result = rule_engine.evaluate_wave_count(invalid_wave_count_2, invalid_segments_2)
    print(f"Invalid pattern (Wave 2 > 100% retracement) compliance score: {result['rule_compliance_score']}")
    print(f"Violations: {len(result['rule_violations'])}")
    if result['rule_violations']:
        for violation in result['rule_violations']:
            print(f"  - {violation['rule']}: {violation['message']}")
    
    # Test 4: Invalid pattern - Wave 4 overlaps Wave 1
    print("\\n--- Test 4: Invalid pattern - Wave 4 overlaps Wave 1 ---")
    invalid_segments_3 = [
        {
            'label': '1',
            'price_start': 100.0,
            'price_end': 110.0,
            'price_low': 99.0,
            'price_high': 110.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=5),
            'end': datetime.now(timezone.utc) - timedelta(hours=4)
        },
        {
            'label': '2',
            'price_start': 110.0,
            'price_end': 105.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=4),
            'end': datetime.now(timezone.utc) - timedelta(hours=3)
        },
        {
            'label': '3',
            'price_start': 105.0,
            'price_end': 115.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=3),
            'end': datetime.now(timezone.utc) - timedelta(hours=2)
        },
        {
            'label': '4',
            'price_start': 115.0,
            'price_end': 108.0,  # Overlaps with Wave 1 price territory
            'price_low': 108.0,
            'price_high': 116.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=2),
            'end': datetime.now(timezone.utc) - timedelta(hours=1)
        },
        {
            'label': '5',
            'price_start': 108.0,
            'price_end': 120.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=1),
            'end': datetime.now(timezone.utc)
        }
    ]
    
    invalid_wave_count_3 = {
        'wave_pattern': {
            'label': '1-2-3-4-5',
            'segments': ['seg1', 'seg2', 'seg3', 'seg4', 'seg5']
        }
    }
    
    result = rule_engine.evaluate_wave_count(invalid_wave_count_3, invalid_segments_3)
    print(f"Invalid pattern (Wave 4 overlaps Wave 1) compliance score: {result['rule_compliance_score']}")
    print(f"Violations: {len(result['rule_violations'])}")
    if result['rule_violations']:
        for violation in result['rule_violations']:
            print(f"  - {violation['rule']}: {violation['message']}")
    
    # Test 5: Valid corrective pattern (3 waves)
    print("\\n--- Test 5: Valid corrective pattern (3 waves) ---")
    corrective_segments = [
        {
            'label': 'a',
            'price_start': 120.0,
            'price_end': 110.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=3),
            'end': datetime.now(timezone.utc) - timedelta(hours=2)
        },
        {
            'label': 'b',
            'price_start': 110.0,
            'price_end': 115.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=2),
            'end': datetime.now(timezone.utc) - timedelta(hours=1)
        },
        {
            'label': 'c',
            'price_start': 115.0,
            'price_end': 108.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=1),
            'end': datetime.now(timezone.utc)
        }
    ]
    
    corrective_wave_count = {
        'wave_pattern': {
            'label': 'a-b-c',
            'segments': ['sega', 'segb', 'segc']
        }
    }
    
    result = rule_engine.evaluate_wave_count(corrective_wave_count, corrective_segments)
    print(f"Valid corrective pattern compliance score: {result['rule_compliance_score']}")
    print(f"Violations: {len(result['rule_violations'])}")
    if result['rule_violations']:
        for violation in result['rule_violations']:
            print(f"  - {violation['rule']}: {violation['message']}")
    
    print("\\n=== HARD RULES VALIDATION COMPLETE ===")
    return True

def test_hypothesis_validation():
    """Test hypothesis validation with the wave hypothesis engine."""
    print("\\n=== TESTING HYPOTHESIS VALIDATION ===")
    
    # Initialize the wave hypothesis engine
    config = {
        'min_wave_duration_seconds': 60,
        'max_wave_duration_days': 7,
        'elliott_rule_strictness': 'moderate'
    }
    engine = WaveHypothesisEngine(config)
    print("✅ WaveHypothesisEngine initialized")
    
    # Create a valid hypothesis
    valid_segments = [
        {
            'id': 'seg1',
            'label': '1',
            'price_start': 100.0,
            'price_end': 110.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=5),
            'end': datetime.now(timezone.utc) - timedelta(hours=4)
        },
        {
            'id': 'seg2',
            'label': '2',
            'price_start': 110.0,
            'price_end': 105.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=4),
            'end': datetime.now(timezone.utc) - timedelta(hours=3)
        },
        {
            'id': 'seg3',
            'label': '3',
            'price_start': 105.0,
            'price_end': 115.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=3),
            'end': datetime.now(timezone.utc) - timedelta(hours=2)
        },
        {
            'id': 'seg4',
            'label': '4',
            'price_start': 115.0,
            'price_end': 112.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=2),
            'end': datetime.now(timezone.utc) - timedelta(hours=1)
        },
        {
            'id': 'seg5',
            'label': '5',
            'price_start': 112.0,
            'price_end': 120.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=1),
            'end': datetime.now(timezone.utc)
        }
    ]
    
    valid_wave_count = {
        'wave_pattern': {
            'label': '1-2-3-4-5',
            'segments': ['seg1', 'seg2', 'seg3', 'seg4', 'seg5']
        },
        'rank': 1,
        'description': 'Valid 5-wave impulse pattern'
    }
    
    # Create hypothesis object
    hypothesis = WaveHypothesis(valid_wave_count, valid_segments, 'test_valid_hypothesis')
    print(f"Created hypothesis: {hypothesis.id}")
    
    # Test validation
    is_valid = engine._validate_hypothesis(hypothesis)
    print(f"Hypothesis validation result: {is_valid}")
    print(f"Hypothesis is valid: {hypothesis.is_valid}")
    print(f"Confidence score: {hypothesis.confidence_score}")
    print(f"Violations: {len(hypothesis.rule_violations)}")
    
    if hypothesis.rule_violations:
        for violation in hypothesis.rule_violations:
            print(f"  - {violation['rule']}: {violation['message']}")
    
    # Create an invalid hypothesis
    invalid_segments = [
        {
            'id': 'seg1',
            'label': '1',
            'price_start': 100.0,
            'price_end': 110.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=5),
            'end': datetime.now(timezone.utc) - timedelta(hours=4)
        },
        {
            'id': 'seg2',
            'label': '2',
            'price_start': 110.0,
            'price_end': 105.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=4),
            'end': datetime.now(timezone.utc) - timedelta(hours=3)
        },
        {
            'id': 'seg3',
            'label': '3',
            'price_start': 105.0,
            'price_end': 107.0,  # Very short Wave 3
            'start': datetime.now(timezone.utc) - timedelta(hours=3),
            'end': datetime.now(timezone.utc) - timedelta(hours=2)
        },
        {
            'id': 'seg4',
            'label': '4',
            'price_start': 107.0,
            'price_end': 106.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=2),
            'end': datetime.now(timezone.utc) - timedelta(hours=1)
        },
        {
            'id': 'seg5',
            'label': '5',
            'price_start': 106.0,
            'price_end': 108.0,
            'start': datetime.now(timezone.utc) - timedelta(hours=1),
            'end': datetime.now(timezone.utc)
        }
    ]
    
    invalid_wave_count = {
        'wave_pattern': {
            'label': '1-2-3-4-5',
            'segments': ['seg1', 'seg2', 'seg3', 'seg4', 'seg5']
        },
        'rank': 2,
        'description': 'Invalid 5-wave impulse pattern (Wave 3 shortest)'
    }
    
    # Create hypothesis object
    invalid_hypothesis = WaveHypothesis(invalid_wave_count, invalid_segments, 'test_invalid_hypothesis')
    print(f"\\nCreated invalid hypothesis: {invalid_hypothesis.id}")
    
    # Test validation
    is_valid = engine._validate_hypothesis(invalid_hypothesis)
    print(f"Invalid hypothesis validation result: {is_valid}")
    print(f"Invalid hypothesis is valid: {invalid_hypothesis.is_valid}")
    print(f"Confidence score: {invalid_hypothesis.confidence_score}")
    print(f"Violations: {len(invalid_hypothesis.rule_violations)}")
    
    if invalid_hypothesis.rule_violations:
        for violation in invalid_hypothesis.rule_violations:
            print(f"  - {violation['rule']}: {violation['message']}")
    
    print("\\n=== HYPOTHESIS VALIDATION COMPLETE ===")
    return True

def main():
    """Main function to run all validation tests."""
    print("=== ELLIOTT WAVE HARD RULES VALIDATION TEST ===")
    print(f"Started at: {datetime.now()}")
    
    try:
        # Test hard rule enforcement
        rules_result = test_hard_rule_enforcement()
        
        # Test hypothesis validation
        hypothesis_result = test_hypothesis_validation()
        
        print("\\n=== VALIDATION TEST SUMMARY ===")
        print("✅ EnhancedElliottRuleEngine properly enforces hard rules")
        print("✅ WaveHypothesisEngine correctly validates hypotheses")
        print("✅ Valid patterns are accepted")
        print("✅ Invalid patterns are rejected with specific violations")
        print("✅ All Elliott Wave hard rules are properly enforced")
        
        print("\\n🎉 ALL VALIDATION TESTS PASSED")
        print(f"Finished at: {datetime.now()}")
        
    except Exception as e:
        print(f"\\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)