#!/usr/bin/env python3
"""
Example script demonstrating contact force distribution analysis.

This script shows how to analyze and visualize contact force distributions
across SpiRob segments, including mean shares and peak usages.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import polars as pl
from math_spirob import (
    load_summary_parquet,
    plot_force_distribution,
    plot_force_peak_usage
)
import matplotlib
#matplotlib.use('Agg')  # Use non-interactive backend for headless environments

def main():
    print("SpiRob Contact Force Distribution Analysis")
    print("=" * 50)
    
    # Load summary data
    df = load_summary_parquet()
    print(f"Loaded summary data for {len(df)} runs")
    
    # Show available contact metrics
    contact_cols = [col for col in df.columns if 'contact' in col]
    share_mean_cols = [col for col in contact_cols if 'share_mean' in col]
    force_max_cols = [col for col in contact_cols if 'force_max' in col and not 'share' in col]
    max_segment_cols = [col for col in contact_cols if 'max_contact' in col]
    
    print(f"Found {len(share_mean_cols)} share mean metrics")
    print(f"Found {len(force_max_cols)} force max metrics")
    print(f"Max segment metrics: {max_segment_cols}")
    
    # Example: Analyze a specific run
    run_id = "Run_001_Ramped_Cyl_Sz0.02-0.10-0.00_L0.30_T2.0"
    run_data = df.filter(pl.col("run_id") == run_id)
    if run_data.is_empty():
        print(f"Run {run_id} not found")
        return
    
    print(f"\nAnalyzing run: {run_id}")
    
    # Get max segments
    max_force_seg = run_data.select("max_contact_force_segment").item()
    max_force_val = run_data.select("max_contact_force_value").item()
    max_share_seg = run_data.select("max_contact_share_segment").item()
    max_share_val = run_data.select("max_contact_share_value").item()
    
    print(f"Segment with highest peak force: {max_force_seg} (value: {max_force_val:.2f})")
    print(f"Segment with highest peak share: {max_share_seg} (value: {max_share_val:.3f})")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Mean force distribution
    plot_force_distribution(run_id)
    print(f"Saved mean force distribution plot: contact_distribution_{run_id}.png")
    
    # Peak force usage
    plot_force_peak_usage(run_id, metric='force')
    print(f"Saved peak force plot: peak_force_{run_id}.png")

    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()