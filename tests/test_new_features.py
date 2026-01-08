#!/usr/bin/env python3
"""
Test script for new features: contact forces, exporter, plots.
"""

import mujoco as mj
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from math_spirob import spirob_simulate as sim, data_schema as ds, exporter as exp, plots

def test_contact_forces():
    """Test contact force extraction."""
    print("Testing contact forces...")
    
    # Simple model: box on ground
    xml = """
<mujoco model="test">
    <option gravity="0 0 -9.81" timestep="0.01"/>
    <worldbody>
        <geom name="ground" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="box" pos="0 0 0.1">
            <joint type="free"/>
            <geom name="box_geom" type="box" size="0.05 0.05 0.05" mass="1.0"/>
        </body>
    </worldbody>
</mujoco>
"""
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    
    # Run a few steps
    for _ in range(100):
        mj.mj_step(model, data)
    
    # Extract forces
    forces = sim.extract_body_contact_forces(model, data)
    print(f"Body forces: {forces}")
    
    # Check box has upward force balancing gravity
    box_force_z = forces[1][2]  # body 1 is box
    expected = 9.81  # upward force
    print(f"Box force Z: {box_force_z}, expected ~{expected}")
    assert abs(box_force_z - expected) < 1.0, f"Force not balancing gravity: {box_force_z}"
    
    print("Contact forces test passed.")

def test_exporter():
    """Test exporter with contact forces."""
    print("Testing exporter...")
    
    # Use existing run
    run_id = "Run_001_Ramped_Cyl_Sz0.02-0.10-0.00_L0.30_T2.0"
    df = plots.load_experiment_parquet(run_id)
    sensors = exp.generate_sensor_meta(df)
    
    # Check for body contact forces
    body_sensors = [s for s in sensors if s.group.value == 'body_contact_force']
    print(f"Body sensors found: {len(body_sensors)}")
    assert len(body_sensors) > 0, "No body contact force sensors found"
    
    # Check columns
    body_cols = [col for col in df.columns if 'body_' in col and '_contact_force_' in col]
    print(f"Body force columns: {len(body_cols)}")
    assert len(body_cols) > 0, "No body force columns in df"
    
    print("Exporter test passed.")

def test_plots():
    """Test plot functions."""
    print("Testing plots...")
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive
    
    from math_spirob.plots import plot_time_series, plot_distribution, quick_plot
    
    # Test time series
    plot_time_series("Run_001_Ramped_Cyl_Sz0.02-0.10-0.00_L0.30_T2.0", ["acc_0"], ["X"], save_path="test_ts.png")
    
    # Test distribution
    plot_distribution("all", "acc_0", "X", "mean", save_path="test_dist.png")
    
    # Test quick plot
    quick_plot("acc_0", "X", "mean", save_path="test_quick.png")
    
    print("Plots test passed.")

if __name__ == "__main__":
    test_contact_forces()
    test_exporter()
    test_plots()
    print("All tests passed!")