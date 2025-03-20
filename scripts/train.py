#!/usr/bin/env python
"""
Training script for TCM target prioritization system.
"""
import os
import sys
import argparse
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.main import train

def main():
    """Main function."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Train TCM Target Prioritization Model")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Train model
    train(args.config)

if __name__ == "__main__":
    main()
