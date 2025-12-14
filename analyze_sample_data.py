#!/usr/bin/env python3

import sys
import os
import pandas as pd
import yaml

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.wave_detector import ElliottRuleEngine
from analysis.nms import NMS
from analysis.confidence import ConfidenceScorer

def load_sample_data(file_path):
    """Load sample data from CSV file"""
    print(f"Loading data from {file_path}")
    # Read the CSV file with header
    df = pd.read_csv(file_path)
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    print(f"Loaded {len(df)} data points")
    return df

def load_config():
    """Load configuration from config.yaml"""
    try:
        with open("config.yaml", 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("config.yaml not found. Using default configuration.")
        return {
            "min_wave_duration_seconds": 60,
            "max_wave_duration_days": 7,
            "elliott_rule_strictness": "moderate",
            "confidence_weights": {
                "rule_compliance": 0.6,
                "amplitude_duration_norm": 0.2,
                "volatility_penalty": 0.1,
                "chaos_metric": 0.1,
            }
        }

def main():
    print("Elliott Wave Analysis - Sample Data Tester")
    print("=" * 40)
    
    # Load configuration
    config = load_config()
    
    # Initialize analysis components
    rule_engine = ElliottRuleEngine(config)
    nms = NMS()
    confidence_scorer = ConfidenceScorer(
        weights=config.get('confidence_weights', {})
    )
    
    # Find sample data files
    data_dir = "data"
    sample_files = []
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.startswith("sample_") and file.endswith(".csv"):
                sample_files.append(os.path.join(data_dir, file))
    
    if not sample_files:
        print("No sample data files found in data/ directory")
        return
    
    print(f"Found {len(sample_files)} sample data files:")
    for i, file in enumerate(sample_files):
        print(f"  {i+1}. {os.path.basename(file)}")
    
    # Analyze each sample file
    for file_path in sample_files:
        print(f"\nAnalyzing {os.path.basename(file_path)}...")
        try:
            # Load data
            df = load_sample_data(file_path)
            
            # Print basic statistics
            print(f"  Data range: {df.index.min()} to {df.index.max()}")
            print(f"  Price range: {df['price'].min():.2f} to {df['price'].max():.2f}")
            print(f"  Volume range: {df['volume'].min():.0f} to {df['volume'].max():.0f}")
            
            # Test rule engine with sample data
            print("  Testing rule engine...")
            # Create a simple test segment to check rule engine functionality
            test_segment = {
                'start': df.index.min(),
                'end': df.index.max(),
                'start_price': float(df['price'].iloc[0]),
                'end_price': float(df['price'].iloc[-1])
            }
            
            # Check wave duration
            duration_score = rule_engine._check_wave_duration(test_segment)
            print(f"    Duration check score: {duration_score}")
            
            print("  Analysis completed successfully!")
            
        except Exception as e:
            print(f"  Error analyzing {file_path}: {str(e)}")
    
    print("\nSample data analysis complete!")

if __name__ == "__main__":
    main()