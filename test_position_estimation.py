#!/usr/bin/env python3
"""
Test script for position estimation integration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from math_spirob.analyzer import load_experiment
from math_spirob.meta_analyzer import compute_metrics
from math_spirob.data_schema import ExperimentConfig

# Test with existing run
run_id = 'Run_003_Ramped_Cyl_Sz0.04-0.10-0.00_L0.30_T2.0'

record, lf = load_experiment(run_id)

# Temporarily enable position estimation
record.config.enable_position_estimation = True
record.config.position_estimator_segments = [0, 5, 10]
record.config.initial_positions = {
    0: [0.0, 0.0, 0.0],
    5: [0.15, 0.0, 0.0],
    10: [0.3, 0.0, 0.0]
}
record.config.initial_orientations = {
    0: [1.0, 0.0, 0.0, 0.0],
    5: [1.0, 0.0, 0.0, 0.0],
    10: [1.0, 0.0, 0.0, 0.0]
}

print(f"Testing position estimation for run {run_id}")
print(f"Sensors: {len(record.sensors)}")

metrics = compute_metrics(record, lf)

# Print position estimation metrics
pos_metrics = {k: v for k, v in metrics.items() if 'pos' in k or 'tip' in k or 'drift' in k}
print("Position estimation metrics:")
for k, v in sorted(pos_metrics.items()):
    print(f"  {k}: {v}")

print(f"Total metrics: {len(metrics)}")