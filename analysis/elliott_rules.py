# Elliott Wave Rules Summary
# This document summarizes common Elliott Wave principles based on standard interpretations.
# It serves as a reference for the 'wave_detector.py' module.
# Quote: "moderate" strictness is applied.

# General Principles:
# - Markets move in repetitive patterns (waves).
# - Waves are fractal: patterns at smaller scales are similar to patterns at larger scales.
# - Waves are categorized as either "impulse" (moving in the direction of the trend) or "corrective" (moving against the trend).

# Impulse Waves (5-wave pattern):
# - Waves 1, 3, 5 move in the direction of the main trend.
# - Waves 2, 4 move against the main trend (corrective).
# - The entire 5-wave sequence forms one wave of a higher degree.
# - Wave 2 never retraces more than 100% of Wave 1.
# - Wave 3 is often the longest and most powerful wave, and is never the shortest of waves 1, 3, and 5.
# - Wave 4 never enters the price territory of Wave 1 (the "no overlap" rule for impulse waves).
# - Waves 1, 3, 5 are impulse waves (5 sub-waves).
# - Waves 2, 4 are corrective waves (typically 3 sub-waves: a-b-c).

# Corrective Waves (3-wave pattern, or more complex patterns like zigzags, flats, triangles):
# - Move against the main trend.
# - Typically take the form of A-B-C.
# - Wave A: often impulsive, moving against the trend.
# - Wave B: often a retracement of Wave A, can be complex.
# - Wave C: often impulsive, moving in the direction of the trend (which is opposite to the overall trend of the larger degree wave it's correcting).
# - Wave C is often equal in length to Wave A, or extends to 1.618 times Wave A's length.

# Fibonacci Relationships:
# - Retracements: Wave 2 often retraces 61.8% of Wave 1. Wave 4 often retraces 38.2% of Wave 3.
# - Extensions: Wave 3 is often 1.618 times the length of Wave 1. Wave 5 often relates to Wave 1 and Wave 3 lengths.
# - Other common ratios include 38.2%, 50%, 138.2%, 161.8%, 261.8%.

# Key Rules for Moderate Strictness (as inferred for implementation):
# Quote: "moderate" Elliott rule strictness

# 1. Wave 2 cannot retrace more than 100% of Wave 1.
#    - 'price_space_overlap_2_vs_1': False (Wave 2's low cannot be lower than Wave 1's low if Wave 1 is up)
# 2. Wave 3 cannot be the shortest impulse wave (among 1, 3, 5).
#    - 'shortest_impulse_wave_3': False
# 3. Wave 4 cannot overlap with the price territory of Wave 1 (for impulse waves).
#    - 'price_space_overlap_4_vs_1': False
# 4. Alternation Rule: If Wave 2 is sharp, Wave 4 tends to be sideways, and vice versa.
#    - 'alternation_rule': True (Implement basic alternation: if Wave 2 is sharp, Wave 4 should not be sharp, and vice versa. This is complex to quantify strictly.)
# 5. Impuse waves consist of 5 sub-waves. Corrective waves consist of 3 sub-waves. (This is fundamental to labeling).

# For confidence scoring, rule compliance will be a major factor.
# Specific checks within the 'wave_detector.py' will implement these rules.
# We will prioritize the 'no overlap' and 'shortest wave' rules for moderate strictness.
