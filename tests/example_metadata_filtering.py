#!/usr/bin/env python3
"""
Example script demonstrating metadata-based filtering for plots.

This script shows how to use the new filtering capabilities to plot
data from runs matching specific metadata criteria.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from math_spirob import (
    load_run_metadata, 
    list_available_meta_fields, 
    filter_runs,
    plot_time_series_filtered,
    plot_comparison_filtered
)
import matplotlib
#matplotlib.use('Agg')  # Use non-interactive backend for headless environments

def main():
    print("MuJoCo SpiRob - Metadata Filtering Example")
    print("=" * 50)
    
    # Load metadata
    meta_df = load_run_metadata()
    print(f"Loaded metadata for {len(meta_df)} runs")
    
    # Show available fields
    fields = list_available_meta_fields()
    print(f"Available metadata fields: {fields}")
    
    # Example 1: Filter by geometry type
    print("\nExample 1: Filter by geometry type 'cylinder'")
    cylinder_runs = filter_runs(meta_df, {'geom_type': 'cylinder'})
    print(f"Found {len(cylinder_runs)} cylinder runs: {cylinder_runs}...")
    
    # Example 2: Filter by L_target range
    print("\nExample 2: Filter by L_target >= 0.35")
    long_runs = filter_runs(meta_df, {'L_target_min': 0.36})
    print(f"Found {len(long_runs)} runs with L_target >= 0.35: {long_runs}")
    
    # Example 3: Multiple filters
    print("\nExample 3: Filter by geom_type and L_target")
    filtered_runs = filter_runs(meta_df, {'geom_type': 'cylinder', 'L_target_min': 0.35})
    print(f"Found {len(filtered_runs)} cylinder runs with L_target >= 0.35: {filtered_runs}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Time series plot for cylinder runs
    plot_time_series_filtered(
        {'geom_type': 'cylinder'}, 
        sensors=['body_seg_13_contact_force'], 
        axes=['X', 
              #'Y', 
              #'Z'
              ], 
        #save_path='example_cylinder_accelerometer.png'
    )
    plot_comparison_filtered(
        {'geom_type': 'cylinder'}, 
        sensor='body_seg_13_contact_force', 
        axis='X', 
        metric='mean', 
        #save_path='example_cylinder_acc_X_mean.png'
    )
    
    # Comparison plot for mean acceleration
    # plot_comparison_filtered(
    #     {'geom_type': 'cylinder', 'L_target_min': 0.36}, 
    #     sensor='acc_0', 
    #     axis='X', 
    #     metric='mean', 
    #     #save_path='example_cylinder_acc_X_mean.png'
    # )
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()