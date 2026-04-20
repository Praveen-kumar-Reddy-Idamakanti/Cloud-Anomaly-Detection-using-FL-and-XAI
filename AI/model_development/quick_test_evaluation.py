#!/usr/bin/env python3
"""
Quick test to measure evaluation runtime with single seed
"""

import time
import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "model_development"))

from working_seeded_evaluation import WorkingSeededEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_runtime_test():
    """Test runtime with single seed"""
    logger.info("⏱️  Quick Runtime Test - Single Seed")
    logger.info("="*50)
    
    evaluator = WorkingSeededEvaluator()
    evaluator.seeds = [42]  # Test with single seed only
    
    start_time = time.time()
    result = evaluator.evaluate_with_seed(42)
    end_time = time.time()
    
    runtime = end_time - start_time
    logger.info(f"\n⏱️  Single seed runtime: {runtime:.2f} seconds")
    logger.info(f"📊 Estimated 3-seed runtime: {runtime*3:.2f} seconds ({runtime*3/60:.1f} minutes)")
    
    return runtime

if __name__ == "__main__":
    quick_runtime_test()
