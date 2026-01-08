import mujoco as mj
import numpy as np
import time

def create_box_on_ground_model():
    """Creates a simple MuJoCo model: box on ground plane."""
    xml = """
<mujoco model="box_on_ground">
    <option gravity="0 0 -9.81" timestep="0.01"/>
    <worldbody>
        <geom name="ground" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="box" pos="0 0 0.1">
            <joint type="free"/>
            <geom name="box_geom" type="box" size="0.05 0.05 0.05" mass="1.0" rgba="0.2 0.8 0.2 1"/>
        </body>
    </worldbody>
</mujoco>
"""
    return mj.MjModel.from_xml_string(xml)

def create_box_on_inclined_plane_model(angle_deg=30):
    """Creates a MuJoCo model: box on inclined plane."""
    angle_rad = np.deg2rad(angle_deg)
    xml = f"""
<mujoco model="box_on_incline">
    <option gravity="0 0 -9.81" timestep="0.01"/>
    <worldbody>
        <geom name="ground" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="plane" pos="0 0 0" euler="{angle_rad} 0 0">
            <geom name="plane_geom" type="box" size="0.5 0.01 0.5" rgba="0.5 0.5 0.5 1"/>
        </body>
        <body name="box" pos="0 0.2 0.2">
            <joint type="free"/>
            <geom name="box_geom" type="box" size="0.05 0.05 0.05" mass="1.0" rgba="0.2 0.8 0.2 1"/>
        </body>
    </worldbody>
</mujoco>
"""
    return mj.MjModel.from_xml_string(xml)

def log_contact_forces(model, data, step):
    """Logs detailed contact force information for all contacts."""
    print(f"\n--- Step {step} ---")
    print(f"Number of contacts: {data.ncon}")
    
    body_forces_v1 = {i: np.zeros(3) for i in range(model.nbody)}  # Using contact.frame @ force[:3]
    body_forces_v2 = {i: np.zeros(3) for i in range(model.nbody)}  # Using contact.frame.T @ force[:3]
    
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        
        force = np.zeros(6)
        mj.mj_contactForce(model, data, contact_id, force)
        
        geom1 = contact.geom1
        geom2 = contact.geom2
        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]
        
        geom1_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom1)
        geom2_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom2)
        body1_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body1)
        body2_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body2)
        
        rotation_matrix = contact.frame.reshape(3, 3)
        force_contact = force[:3]
        
        # Two possible transformations
        force_world_v1 = rotation_matrix @ force_contact  # Assuming contact.frame is contact -> world
        force_world_v2 = rotation_matrix.T @ force_contact  # Transpose
        
        print(f"Contact {contact_id}: geom1={geom1_name} (body={body1_name}), geom2={geom2_name} (body={body2_name})")
        print(f"  force_contact: {force_contact}")
        print(f"  contact.frame (reshaped):\n{rotation_matrix}")
        print(f"  force_world_v1 (frame @ force): {force_world_v1}")
        print(f"  force_world_v2 (frame.T @ force): {force_world_v2}")
        
        # Accumulate for bodies - assuming force_world_v1 is force on geom1
        body_forces_v1[body1] += force_world_v1
        body_forces_v1[body2] -= force_world_v1
        
        body_forces_v2[body1] += force_world_v2
        body_forces_v2[body2] -= force_world_v2
    
    print("Body forces V1 (frame @ force):")
    for body_id, force in body_forces_v1.items():
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        print(f"  {body_name}: {force}")
    
    print("Body forces V2 (frame.T @ force):")
    for body_id, force in body_forces_v2.items():
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        print(f"  {body_name}: {force}")
    
    return body_forces_v1, body_forces_v2

def check_physics(body_forces, model, test_name):
    """Checks if body forces are physically reasonable."""
    print(f"\n--- Physics Check for {test_name} ---")
    
    # Find box body (assuming it's the one with mass)
    box_id = None
    for i in range(model.nbody):
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if body_name == "box":
            box_id = i
            break
    
    if box_id is not None:
        force_on_box = body_forces[box_id]
        mass = 1.0  # From model
        gravity = 9.81
        expected_z = mass * gravity
        actual_z = force_on_box[2]
        print(f"Force on box z-component: {actual_z:.3f}, expected ≈ {expected_z:.3f}, diff: {abs(actual_z - expected_z):.3f}")
        
        # Check total force balance (should be near zero for static)
        total_force = np.zeros(3)
        for fid, f in body_forces.items():
            total_force += f
        print(f"Total force sum: {total_force}, magnitude: {np.linalg.norm(total_force):.6f}")
    else:
        print("Box body not found for physics check.")

def run_test(model, sim_time=2.0, log_steps=[50, 99], test_name="Test"):
    """Runs simulation and logs contact forces at specified steps."""
    data = mj.MjData(model)
    
    steps = int(sim_time / model.opt.timestep)
    
    for step in range(steps):
        mj.mj_step(model, data)
        
        if step in log_steps:
            body_forces_v1, body_forces_v2 = log_contact_forces(model, data, step)
            check_physics(body_forces_v1, model, f"{test_name} V1")
            check_physics(body_forces_v2, model, f"{test_name} V2")
    
    # Final log
    body_forces_v1, body_forces_v2 = log_contact_forces(model, data, steps)
    check_physics(body_forces_v1, model, f"{test_name} V1 Final")
    check_physics(body_forces_v2, model, f"{test_name} V2 Final")

def test_box_on_ground():
    """Test 1: Box on ground."""
    print("=== Test 1: Box on Ground ===")
    model = create_box_on_ground_model()
    run_test(model, sim_time=1.0, log_steps=[50, 99], test_name="Box on Ground")

def test_box_on_incline():
    """Test 2: Box on inclined plane."""
    print("=== Test 2: Box on Inclined Plane ===")
    model = create_box_on_inclined_plane_model(30)
    run_test(model, sim_time=1.0, log_steps=[50, 99], test_name="Box on Incline")

if __name__ == "__main__":
    print("""
Test Script for MuJoCo Contact Force Frame Investigation

This script tests the behavior of mj_contactForce and contact.frame in MuJoCo.
It runs two simple scenarios:
1. Box on ground: Verifies that contact forces balance gravity.
2. Box on inclined plane: Checks force components.

For each test, it logs:
- Raw contact forces in contact frame
- Two possible world frame transformations: frame @ force and frame.T @ force
- Accumulated body forces for both variants
- Physics checks: force balance and expected magnitudes

Interpretation:
- V1 (frame @ force) should show physically correct forces (box gets upward force ≈ m*g)
- V2 (frame.T @ force) will show inverted forces (box would fall)

Run with: python tests/test_contact_forces_frame.py
""")
    test_box_on_ground()
    test_box_on_incline()