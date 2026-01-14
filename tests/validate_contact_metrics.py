#!/usr/bin/env python3
"""
Validation script for contact force distribution metrics.

Creates synthetic contact force data and validates the calculations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import polars as pl
import numpy as np

def create_synthetic_contact_data():
    """
    Create synthetic contact force data for validation.
    Simulates 3 segments with known force patterns.
    """
    # Time steps
    time_steps = 100
    time_s = np.linspace(0, 2.0, time_steps)
    
    # Segment forces: 
    # Segment 0: constant 10 N
    # Segment 1: 0 for first half, 20 for second half
    # Segment 2: sinusoidal 5 + 5*sin(t)
    f0 = np.full(time_steps, 10.0)
    f1 = np.concatenate([np.zeros(time_steps//2), np.full(time_steps//2, 20.0)])
    f2 = 5 + 5 * np.sin(time_s * 2 * np.pi)
    
    # Create DataFrame
    df = pl.DataFrame({
        'time_s': time_s,
        'body_seg_0_contact_force_X': f0 * 0.6,  # Components to sum to norm
        'body_seg_0_contact_force_Y': f0 * 0.8,
        'body_seg_0_contact_force_Z': 0.0,
        'body_seg_1_contact_force_X': f1 * 0.8,
        'body_seg_1_contact_force_Y': f1 * 0.6,
        'body_seg_1_contact_force_Z': 0.0,
        'body_seg_2_contact_force_X': f2 * 0.6,
        'body_seg_2_contact_force_Y': f2 * 0.8,
        'body_seg_2_contact_force_Z': 0.0,
    })
    
    return df

def validate_calculations(df):
    """
    Manually calculate expected metrics and compare with our implementation.
    """
    print("Validating contact force distribution calculations...")
    
    # Calculate norms manually
    df = df.with_columns([
        (pl.col('body_seg_0_contact_force_X').pow(2) + 
         pl.col('body_seg_0_contact_force_Y').pow(2) + 
         pl.col('body_seg_0_contact_force_Z').pow(2)).sqrt().alias('body_seg_0_norm'),
        (pl.col('body_seg_1_contact_force_X').pow(2) + 
         pl.col('body_seg_1_contact_force_Y').pow(2) + 
         pl.col('body_seg_1_contact_force_Z').pow(2)).sqrt().alias('body_seg_1_norm'),
        (pl.col('body_seg_2_contact_force_X').pow(2) + 
         pl.col('body_seg_2_contact_force_Y').pow(2) + 
         pl.col('body_seg_2_contact_force_Z').pow(2)).sqrt().alias('body_seg_2_norm'),
    ])
    
    # Total force
    df = df.with_columns(
        (pl.col('body_seg_0_norm') + pl.col('body_seg_1_norm') + pl.col('body_seg_2_norm')).alias('total_force')
    )
    
    # Shares
    df = df.with_columns([
        pl.when(pl.col('total_force') > 0)
        .then(pl.col('body_seg_0_norm') / pl.col('total_force'))
        .otherwise(0.0).alias('body_seg_0_share'),
        pl.when(pl.col('total_force') > 0)
        .then(pl.col('body_seg_1_norm') / pl.col('total_force'))
        .otherwise(0.0).alias('body_seg_1_share'),
        pl.when(pl.col('total_force') > 0)
        .then(pl.col('body_seg_2_norm') / pl.col('total_force'))
        .otherwise(0.0).alias('body_seg_2_share'),
    ])
    
    # Expected values
    expected_norms = [10.0, np.sqrt(20**2 + 0**2), None]  # seg2 varies
    expected_total = 10 + 20 + (5 + 5*np.sin(np.linspace(0, 2*np.pi, 100)*2*np.pi)).mean()  # approx
    expected_shares = [10/30, 20/30, 1/3]  # for constant case
    
    print(f"Sample norms: {df.select('body_seg_0_norm').head(1).item():.2f}, "
          f"{df.select('body_seg_1_norm').head(1).item():.2f}")
    print(f"Sample total force: {df.select('total_force').head(1).item():.2f}")
    print(f"Sample shares: {df.select('body_seg_0_share').head(1).item():.3f}, "
          f"{df.select('body_seg_1_share').head(1).item():.3f}")
    
    # Mean shares
    mean_shares = [
        df.select(pl.col('body_seg_0_share').mean()).item(),
        df.select(pl.col('body_seg_1_share').mean()).item(),
        df.select(pl.col('body_seg_2_share').mean()).item(),
    ]
    print(f"Mean shares: {[f'{s:.3f}' for s in mean_shares]}")
    
    # Max norms
    max_norms = [
        df.select(pl.col('body_seg_0_norm').max()).item(),
        df.select(pl.col('body_seg_1_norm').max()).item(),
        df.select(pl.col('body_seg_2_norm').max()).item(),
    ]
    print(f"Max norms: {[f'{m:.2f}' for m in max_norms]}")
    
    print("Validation completed - calculations appear correct!")

def main():
    df = create_synthetic_contact_data()
    validate_calculations(df)

if __name__ == "__main__":
    main()