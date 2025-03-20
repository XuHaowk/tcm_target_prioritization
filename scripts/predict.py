#!/usr/bin/env python
"""
Prediction script for TCM target prioritization system.
"""
import os
import sys
import argparse
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.main import predict

def main():
    """Main function."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Predict Targets for TCM Compound")
    parser.add_argument("--compound", required=True, help="Compound identifier")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--disease", help="Disease identifier")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top targets to return")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Predict targets
    predict(args.compound, args.model, args.config, args.disease, args.top_k)

if __name__ == "__main__":
    main()
