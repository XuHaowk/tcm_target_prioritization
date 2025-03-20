#!/usr/bin/env python
"""
Evaluation script for TCM target prioritization system.
"""
import os
import sys
import argparse
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.main import evaluate

def main():
    """Main function."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Evaluate TCM Target Prioritization Model")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--data", required=True, help="Path to test data file")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--output-dir", help="Output directory for visualizations")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Evaluate model
    evaluate(args.model, args.data, args.config, args.output_dir)

if __name__ == "__main__":
    main()
