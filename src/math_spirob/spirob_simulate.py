import mujoco as mj
import mujoco.viewer as viewer
import numpy as np
import polars as pl
from typing import Dict, Any, List, Tuple, Callable
import math
import itertools
import json
import time
import imageio
from .simple_segment_estimator import SimpleSegmentEstimator

# Definieren des Controller-Interface (Callback-Signatur)
# Ein Controller muss mj.MjModel, mj.MjData, die aktuelle Zeit (float) 
# und den Schrittindex (int) als Argumente akzeptieren.
ControllerFunc = Callable[[mj.MjModel, mj.MjData, float, int], None]

# --- Helper function for body contact forces ---

def extract_body_contact_forces(model: mj.MjModel, data: mj.MjData) -> Dict[int, np.ndarray]:
    """
    Extracts the total contact forces for each body in world frame.
    
    Returns a dict body_id -> np.array([Fx, Fy, Fz]) in world coordinates.
    
    Frame Convention:
    - mj_contactForce returns force[:3] in the contact frame, where force is the force applied to geom1 by geom2.
    - contact.frame is the 3x3 rotation matrix from contact frame to world frame.
    - Thus, F_world = contact.frame @ force[:3] gives the force on geom1 in world coordinates.
    - The force on geom2 is -F_world.
    - body_forces[body1] accumulates +F_world (force on body1), body_forces[body2] accumulates -F_world.
    """
    body_forces = {i: np.zeros(3, dtype=np.float64) for i in range(model.nbody)}
    
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        
        # Get force in contact frame (6D: Fx, Fy, Fz, Tx, Ty, Tz)
        force = np.zeros(6, dtype=np.float64)
        mj.mj_contactForce(model, data, contact_id, force)
        
        # Transform force from contact frame to world frame
        # contact.frame is rotation matrix: contact -> world
        if len(contact.frame) != 9:
            raise ValueError(f"contact.frame has unexpected length: {len(contact.frame)}")
        rotation_matrix = contact.frame.reshape(3, 3)
        force_world = rotation_matrix @ force[:3]  # F_world = R_contact_to_world @ F_contact
        #print(f"Contact ID {contact_id}: force_contact={force[:3]}, force_world={force_world}")
        #print(rotation_matrix)

        # Get body IDs
        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]
        
        # Apply forces: body1 gets +force_world, body2 gets -force_world
        body_forces[body1] += force_world
        body_forces[body2] -= force_world
    
    return body_forces

# --- 1. Helper-Funktionen (Datenverarbeitung) ---

def get_sliced_dict(data_dict: Dict[str, np.ndarray], final_length: int) -> Dict[str, np.ndarray]:
    """Schneidet alle Arrays im Dictionary auf die tatsächliche Länge (Synchronisation)."""
    return {name: arr[:final_length] for name, arr in data_dict.items()}

def create_single_polars_dataframe(
    sensor_groups: List[Tuple[str, Dict[str, np.ndarray]]], 
    time_data: np.ndarray, 
    final_length: int
) -> pl.DataFrame:
    """
    Erstellt ein einziges, breites Polars DataFrame aus allen Sensor-Gruppen.
    Verwendet die Spaltennamenskonvention: 'sensorname' für 1D und 'sensorname_X/Y/Z' für 3D.
    """
    
    all_columns: List[pl.Series] = [pl.Series("time_s", time_data)]
    
    for group_prefix, data_dict_global in sensor_groups:
        
        if not data_dict_global:
            print(f"Warnung: Gruppe '{group_prefix}' ist leer und wird ignoriert.")
            continue

        sliced_data_dict = get_sliced_dict(data_dict_global, final_length)
        
        for name, values_array in sliced_data_dict.items():
            
            # Special handling for body contact forces
            if group_prefix == "bodycontactfrc":
                sensor_name = f"body_{name}_contact_force"
            else:
                sensor_name = name
            
            # Bestimmung der Dimension (D) für diesen spezifischen Sensor
            current_D = values_array.shape[1] if values_array.ndim == 2 else 1
            
            if current_D == 3:
                # 3D: Benennung: sensorname_x/y/z für geom_pos und pos_estimate, sonst _X/Y/Z
                columns_prefix = sensor_name
                if group_prefix in ["geom_pos", "pos_estimate"]:
                    suffix = ["_x", "_y", "_z"]
                else:
                    suffix = ["_X", "_Y", "_Z"]
                all_columns.append(pl.Series(f"{columns_prefix}{suffix[0]}", values_array[:, 0]))
                all_columns.append(pl.Series(f"{columns_prefix}{suffix[1]}", values_array[:, 1]))
                all_columns.append(pl.Series(f"{columns_prefix}{suffix[2]}", values_array[:, 2]))
            
            elif current_D == 4:
                # 4D: Für Quaternions: sensorname_w/x/y/z, für Velocity: x/y/z/norm
                columns_prefix = sensor_name
                if group_prefix == "quat_estimate":
                    suffix = ["_w", "_x", "_y", "_z"]
                elif group_prefix == "vel_estimate":
                    suffix = ["_x", "_y", "_z", "_norm"]
                else:
                    suffix = ["_0", "_1", "_2", "_3"]
                all_columns.append(pl.Series(f"{columns_prefix}{suffix[0]}", values_array[:, 0]))
                all_columns.append(pl.Series(f"{columns_prefix}{suffix[1]}", values_array[:, 1]))
                all_columns.append(pl.Series(f"{columns_prefix}{suffix[2]}", values_array[:, 2]))
                all_columns.append(pl.Series(f"{columns_prefix}{suffix[3]}", values_array[:, 3]))
            
            elif current_D == 1:
                # 1D: Benennung: sensorname
                column_name = sensor_name
                all_columns.append(pl.Series(column_name, values_array.squeeze()))
            
            else:
                # Optionale Warnung für unbekannte Dimensionen
                print(f"Warnung: Sensor '{sensor_name}' in Gruppe '{group_prefix}' hat Dimension {current_D} und wird ignoriert.")
                
    return pl.DataFrame(all_columns)

# --- 2. Kernsimulationsfunktion ---

def initialize_data_structures(model: mj.MjModel, sim_time: float) -> Tuple[Dict, Dict, Dict, List, np.ndarray, int]:
    """Initialisiert alle Arrays und Metadaten vor der Simulation."""
    
    num_steps = int(sim_time / model.opt.timestep) + 1

    # Dictionaries für die Sensor-Zeitreihen
    acc_over_time, gyro_over_time, tendon_frc_over_time, tendon_pos_over_time, \
    tendon_vel_over_time, joint_pos_over_time, joint_vel_over_time = {}, {}, {}, {}, {}, {}, {}
    positions_over_time = {} # Für Geoms
    quaternions_over_time = {} # Für Geom Quaternions
    body_contact_force_over_time = {} # Für Body-Kontaktkräfte
    
    SENSOR_CONFIG = {
        mj.mjtSensor.mjSENS_ACCELEROMETER:    ('acc',    acc_over_time),
        mj.mjtSensor.mjSENS_GYRO:             ('gyro',   gyro_over_time),
        mj.mjtSensor.mjSENS_TENDONACTFRC:     ('tendon_frc', tendon_frc_over_time),
        mj.mjtSensor.mjSENS_TENDONPOS:        ('tendon_pos', tendon_pos_over_time),
        mj.mjtSensor.mjSENS_TENDONVEL:        ('tendon_vel', tendon_vel_over_time),
        mj.mjtSensor.mjSENS_JOINTPOS:         ('joint_pos',  joint_pos_over_time),
        mj.mjtSensor.mjSENS_JOINTVEL:         ('joint_vel',  joint_vel_over_time),
    }

    sensor_metadata = []
    
    # 1. Sensoren initialisieren
    for i in range(model.nsensor):
        sensor_type = model.sensor_type[i]
        
        if sensor_type in SENSOR_CONFIG:
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)
            _, data_dict = SENSOR_CONFIG[sensor_type]
            dim = model.sensor_dim[i] 
            
            time_series_array = np.zeros((num_steps, dim), dtype=np.float64)
            data_dict[name] = time_series_array 
            
            sensor_metadata.append({
                'name': name,
                'array': time_series_array, 
                'index': i                  
            })

    # 1b. Body-Kontaktkräfte initialisieren
    body_metadata = []
    for i in range(1, model.nbody):  # Skip worldbody (0)
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        force_array = np.zeros((num_steps, 3), dtype=np.float64)  # Fx, Fy, Fz
        body_contact_force_over_time[body_name] = force_array
        body_metadata.append({
            'name': body_name,
            'array': force_array,
            'id': i
        })

    # 2. Geoms initialisieren
    geom_metadata = []
    i = 0
    while True:
        name = f"g_{i}"
        geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
        if geom_id == -1:
            break
        
        pos_array = np.zeros((num_steps, 3), dtype=np.float64)
        positions_over_time[name] = pos_array
        
        geom_metadata.append({
            'name': name,
            'pos_array': pos_array,
            'id': geom_id
        })
        i += 1
        
    time_array = np.zeros(num_steps, dtype=np.float64)

    sensor_dicts = {
        "acc": acc_over_time, "gyro": gyro_over_time, "tendon_frc": tendon_frc_over_time, 
        "tendon_pos": tendon_pos_over_time, "tendon_vel": tendon_vel_over_time, 
        "joint_pos": joint_pos_over_time, "joint_vel": joint_vel_over_time, 
        "geom_pos": positions_over_time, "bodycontactfrc": body_contact_force_over_time
    }

    return sensor_dicts, sensor_metadata, geom_metadata, body_metadata, time_array, num_steps

def create_single_polars_dataframe(
    sensor_groups_config: List[Tuple[str, Dict[str, np.ndarray]]], 
    time_series_data: np.ndarray, 
    final_length: int
) -> pl.DataFrame:
    """
    Creates a single Polars DataFrame from sensor groups config.
    """
    df_dict = {"time_s": time_series_data[:final_length]}
    
    for group_name, sensor_dict in sensor_groups_config:
        for sensor_name, array in sensor_dict.items():
            array = array[:final_length]
            if group_name == "bodycontactfrc":
                df_dict[f"{sensor_name}_contact_force_X"] = array[:, 0]
                df_dict[f"{sensor_name}_contact_force_Y"] = array[:, 1]
                df_dict[f"{sensor_name}_contact_force_Z"] = array[:, 2]
            elif group_name in ["geom_pos", "geom_quat"]:
                # Special handling for geom
                geom_id = sensor_name.split('_')[-1]
                if group_name == "geom_pos":
                    base = f"geom_pos_{geom_id}"
                    df_dict[f"{base}_x"] = array[:, 0]
                    df_dict[f"{base}_y"] = array[:, 1]
                    df_dict[f"{base}_z"] = array[:, 2]
                elif group_name == "geom_quat":
                    base = f"geom_quat_{geom_id}"
                    df_dict[f"{base}_w"] = array[:, 0]
                    df_dict[f"{base}_x"] = array[:, 1]
                    df_dict[f"{base}_y"] = array[:, 2]
                    df_dict[f"{base}_z"] = array[:, 3]
            elif array.ndim == 1:
                # 1D sensor
                df_dict[sensor_name] = array
            elif array.ndim == 2:
                if array.shape[1] == 3:
                    # 3D vector
                    df_dict[f"{sensor_name}_X"] = array[:, 0]
                    df_dict[f"{sensor_name}_Y"] = array[:, 1]
                    df_dict[f"{sensor_name}_Z"] = array[:, 2]
                elif array.shape[1] == 4:
                    # Quaternion
                    df_dict[f"{sensor_name}_w"] = array[:, 0]
                    df_dict[f"{sensor_name}_x"] = array[:, 1]
                    df_dict[f"{sensor_name}_y"] = array[:, 2]
                    df_dict[f"{sensor_name}_z"] = array[:, 3]
                else:
                    # Other dimensions
                    for i in range(array.shape[1]):
                        df_dict[f"{sensor_name}_{i}"] = array[:, i]
    
    return pl.DataFrame(df_dict)

def run_simulation_and_get_dataframe(
    model: mj.MjModel, 
    data: mj.MjData, 
    sim_time: float, 
    controller: ControllerFunc,
    enable_viewer: bool,
    boost_viewer: float,
    include_geom_pos: bool = False,
    record_video: bool = False,
    video_fps: int = 30,
    video_resolution: tuple[int, int] = (640, 480),
    video_path: str = None,
    video_flip_vertical: bool = True,
    enable_position_estimation: bool = False,
    position_estimator_segments: list[int] = None
) -> pl.DataFrame:
    """
    Führt die Simulation aus, sammelt Daten und konvertiert sie in ein Polars DataFrame.
    """
    
    # Initialisiere alle Speicherstrukturen
    sensor_dicts, sensor_metadata, geom_metadata, body_metadata, time_array, num_steps = \
        initialize_data_structures(model, sim_time)
        
    # --- Position Estimation Initialisierung ---
    estimator = None
    pos_estimate_arrays = {}
    quat_estimate_arrays = {}
    vel_estimate_arrays = {}
    if enable_position_estimation and position_estimator_segments:
        # Hole Initialwerte aus MuJoCo
        initial_positions = {}
        initial_orientations = {}
        for seg_id in position_estimator_segments:
            geom_name = f'geom_{seg_id}'
            geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                initial_positions[seg_id] = data.geom_xpos[geom_id].tolist()
                initial_orientations[seg_id] = data.geom_xquat[geom_id].tolist()
        
        estimator = SimpleSegmentEstimator(
            segment_ids=position_estimator_segments,
            initial_positions=initial_positions,
            initial_orientations=initial_orientations
        )
        
        # Initialisiere Arrays für Schätzungen
        for seg_id in position_estimator_segments:
            pos_estimate_arrays[seg_id] = np.zeros((num_steps, 3))
            quat_estimate_arrays[seg_id] = np.zeros((num_steps, 4))
            vel_estimate_arrays[seg_id] = np.zeros((num_steps, 4))  # x,y,z,norm
        
    # --- Video Rendering Initialisierung ---
    frames = []
    width, height = video_resolution if record_video else (640, 480)
    mjv_scene = None
    mjv_camera = None
    mjv_option = None
    mjr_context = None
    rgb_buffer = None
    render_every = 1
    render_counter = 0
    if record_video:
        try:
            # Setze OpenGL-Plattform für headless Rendering
            import os
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
            
            # Initialisiere EGL-Kontext für headless Rendering
            import mujoco.egl
            width, height = video_resolution
            egl_context = mujoco.egl.GLContext(width, height)
            egl_context.make_current()
            
            # Offscreen Rendering Setup
            mjv_scene = mj.MjvScene(model, maxgeom=1000)
            mjv_camera = mj.MjvCamera()
            mjv_option = mj.MjvOption()
            
            # Kamera einstellen: Feste globale Kamera
            mjv_camera.type = mj.mjtCamera.mjCAMERA_FREE
            mjv_camera.lookat = np.array([0.0, 0.0, 0.1])  # Blickpunkt
            mjv_camera.distance = 1.0
            mjv_camera.azimuth = 90.0
            mjv_camera.elevation = -20.0
            
            # Adjust fovy to maintain consistent horizontal FOV across different aspect ratios
            aspect = width / height
            aspect_ref = 4/3  # Reference aspect ratio (640x480)
            fovy_ref = 45.0   # Reference fovy for reference aspect
            hfov_ref_rad = 2 * math.atan(math.tan(math.radians(fovy_ref)/2) * aspect_ref)
            fovy_rad = 2 * math.atan(math.tan(hfov_ref_rad/2) / aspect)
            try:
                mjv_camera.fovy = math.degrees(fovy_rad)
            except AttributeError:
                print(f"Warnung: Kamera FOV Anpassung nicht unterstützt (fovy nicht verfügbar), verwende Standard FOV")
            
            # Context für offscreen rendering
            mjr_context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)
            
            # Setze Buffer für offscreen rendering
            mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, mjr_context)
            
            # Resize offscreen buffer to match video resolution
            mj.mjr_resizeOffscreen(width, height, mjr_context)
            
            # Verify offscreen buffer size
            off_width = mjr_context.offWidth
            off_height = mjr_context.offHeight
            print(f"  Offscreen buffer size: {off_width}x{off_height}")
            if off_width < width or off_height < height:
                print(f"Warnung: Offscreen buffer ({off_width}x{off_height}) kleiner als Video-Auflösung ({width}x{height})")
            
            # Framebuffer für RGB
            rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Berechne Render-Intervall für korrekte FPS
            # render_every = max(1, round(1.0 / (video_fps * dt)))
            # Dies stellt sicher, dass Video-Länge ≈ Simulationszeit
            dt = model.opt.timestep
            render_every = max(1, int(round(1.0 / (video_fps * dt))))
            
            # Debug logging
            print(f"Video-Aufzeichnung initialisiert: {width}x{height} @ {video_fps} FPS (render every {render_every} steps)")
            print(f"  aspect: {aspect:.3f}")
            try:
                print(f"  fovy: {mjv_camera.fovy:.1f}°")
            except AttributeError:
                print("  fovy: nicht verfügbar (Standard verwendet)")
        except ImportError:
            print("Warnung: mujoco.egl nicht verfügbar. Video-Rendering im Headless-Modus nicht unterstützt.")
            record_video = False
        except Exception as e:
            print(f"Warnung: Video-Rendering konnte nicht initialisiert werden: {e}")
            record_video = False
    
    step_index = 0
    
    # Zustand t=0 speichern
    time_array[step_index] = data.time
    for meta in sensor_metadata:
        meta['array'][step_index] = data.sensor(meta['index']).data
    if include_geom_pos:
        for meta in geom_metadata:
            meta['pos_array'][step_index] = data.geom_xpos[meta['id']]
        
    steps_to_run = num_steps - 1

    if enable_viewer:
        with mj.viewer.launch_passive(model, data) as viewer:
            # Wir nutzen data.time für die Abbruchbedingung, 
            # damit wir genau sim_time Sekunden physikalischer Zeit simulieren
            while viewer.is_running() and data.time < sim_time:
                step_start = time.time()

                step_index += 1
                
                # --- CALL THE EXTERNAL CONTROLLER ---
                controller(model, data, data.time, step_index) 
                
                # --- Simulationsschritt ---
                mj.mj_step(model, data)
                
                # --- Position Estimation ---
                if estimator:
                    dt = model.opt.timestep
                    sensor_data = {}
                    for seg_id in position_estimator_segments:
                        acc_cols = [f'acc_{seg_id}_X', f'acc_{seg_id}_Y', f'acc_{seg_id}_Z']
                        gyro_cols = [f'gyro_{seg_id}_X', f'gyro_{seg_id}_Y', f'gyro_{seg_id}_Z']
                        acc_data = []
                        gyro_data = []
                        for col in acc_cols:
                            sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, col)
                            if sensor_id >= 0:
                                acc_data.append(data.sensor(sensor_id).data[0])
                        for col in gyro_cols:
                            sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, col)
                            if sensor_id >= 0:
                                gyro_data.append(data.sensor(sensor_id).data[0])
                        if acc_data and gyro_data:
                            sensor_data[seg_id] = {'acc': acc_data, 'gyro': gyro_data}
                    if sensor_data:
                        estimator.update_batch(sensor_data, dt)
                        states = estimator.get_all_states()
                        for seg_id in position_estimator_segments:
                            if seg_id in states:
                                state = states[seg_id]
                                pos_estimate_arrays[seg_id][step_index] = state.position
                                quat_estimate_arrays[seg_id][step_index] = state.orientation
                                vel_estimate_arrays[seg_id][step_index] = [state.velocity[0], state.velocity[1], state.velocity[2], np.linalg.norm(state.velocity)]
                
                # --- Datenspeicherung ---
                # Sicherheitscheck, damit wir nicht über das Array-Ende schreiben
                if step_index < len(time_array):
                    time_array[step_index] = data.time
                    for meta in sensor_metadata:
                        meta['array'][step_index] = data.sensor(meta['index']).data
                    if include_geom_pos:
                        for meta in geom_metadata:
                            meta['pos_array'][step_index] = data.geom_xpos[meta['id']]
                    
                    # Sammle Body-Kontaktkräfte
                    body_forces = extract_body_contact_forces(model, data)
                    for meta in body_metadata:
                        meta['array'][step_index] = body_forces[meta['id']]

                # GUI aktualisieren
                viewer.sync()

                # --- Video Frame aufzeichnen ---
                if record_video:
                    render_counter += 1
                    if render_counter % render_every == 0:
                        try:
                            mj.mjv_updateScene(model, data, mjv_option, None, mjv_camera, mj.mjtCatBit.mjCAT_ALL, mjv_scene)
                            mj.mjr_render(mj.MjrRect(0, 0, width, height), mjv_scene, mjr_context)
                            mj.mjr_readPixels(rgb_buffer, None, mj.MjrRect(0, 0, width, height), mjr_context)
                            frame = rgb_buffer.copy()
                            if video_flip_vertical:
                                frame = np.flipud(frame)  # MuJoCo rendert bottom-up, Videos brauchen top-down
                            frames.append(frame)
                        except Exception as e:
                            print(f"Warnung: Frame-Aufzeichnung fehlgeschlagen: {e}")

                # --- Zeitsteuerung mit Boost ---
                # Wir teilen den physikalischen Zeitschritt durch den Boost-Faktor
                target_step_duration = model.opt.timestep / boost_viewer
                elapsed_time = time.time() - step_start
                
                time_until_next_step = target_step_duration - elapsed_time
                
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    else:
        for _ in range(steps_to_run):
                
            step_index += 1
            
            # --- CALL THE EXTERNAL CONTROLLER ---
            controller(model, data, data.time, step_index) 
            
            # --- Simulationsschritt ---
            mj.mj_step(model, data)
            
            # --- Position Estimation ---
            if estimator:
                dt = model.opt.timestep
                sensor_data = {}
                for seg_id in position_estimator_segments:
                    acc_cols = [f'acc_{seg_id}_X', f'acc_{seg_id}_Y', f'acc_{seg_id}_Z']
                    gyro_cols = [f'gyro_{seg_id}_X', f'gyro_{seg_id}_Y', f'gyro_{seg_id}_Z']
                    acc_data = []
                    gyro_data = []
                    for col in acc_cols:
                        sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, col)
                        if sensor_id >= 0:
                            acc_data.append(data.sensor(sensor_id).data[0])
                    for col in gyro_cols:
                        sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, col)
                        if sensor_id >= 0:
                            gyro_data.append(data.sensor(sensor_id).data[0])
                    if acc_data and gyro_data:
                        sensor_data[seg_id] = {'acc': acc_data, 'gyro': gyro_data}
                if sensor_data:
                    estimator.update_batch(sensor_data, dt)
                    states = estimator.get_all_states()
                    for seg_id in position_estimator_segments:
                        if seg_id in states:
                            state = states[seg_id]
                            pos_estimate_arrays[seg_id][step_index] = state.position
                            quat_estimate_arrays[seg_id][step_index] = state.orientation
                            vel_estimate_arrays[seg_id][step_index] = [state.velocity[0], state.velocity[1], state.velocity[2], np.linalg.norm(state.velocity)]
            
            # --- Datenspeicherung ---
            time_array[step_index] = data.time
            for meta in sensor_metadata:
                meta['array'][step_index] = data.sensor(meta['index']).data
            if include_geom_pos:
                for meta in geom_metadata:
                    meta['pos_array'][step_index] = data.geom_xpos[meta['id']]
            
            # Sammle Body-Kontaktkräfte
            body_forces = extract_body_contact_forces(model, data)
            for meta in body_metadata:
                meta['array'][step_index] = body_forces[meta['id']]

            # --- Video Frame aufzeichnen ---
            if record_video:
                render_counter += 1
                if render_counter % render_every == 0:
                    try:
                        mj.mjv_updateScene(model, data, mjv_option, None, mjv_camera, mj.mjtCatBit.mjCAT_ALL, mjv_scene)
                        mj.mjr_render(mj.MjrRect(0, 0, width, height), mjv_scene, mjr_context)
                        mj.mjr_readPixels(rgb_buffer, None, mj.MjrRect(0, 0, width, height), mjr_context)
                        frame = rgb_buffer.copy()
                        if video_flip_vertical:
                            frame = np.flipud(frame)  # MuJoCo rendert bottom-up, Videos brauchen top-down
                        frames.append(frame)
                    except Exception as e:
                        print(f"Warnung: Frame-Aufzeichnung fehlgeschlagen: {e}")

    final_length = step_index + 1
    time_series_data = time_array[:final_length] 
    
    # --- 2. Polars Konvertierung ---
    
    SENSOR_GROUPS_CONFIG: List[Tuple[str, Dict[str, np.ndarray]]] = [
        ("acc", sensor_dicts["acc"]),
        ("gyro", sensor_dicts["gyro"]),
        ("tendon_frc", sensor_dicts["tendon_frc"]),
        ("tendon_pos", sensor_dicts["tendon_pos"]),
        ("tendon_vel", sensor_dicts["tendon_vel"]),
        ("joint_pos", sensor_dicts["joint_pos"]),
        ("joint_vel", sensor_dicts["joint_vel"]),
        ("bodycontactfrc", sensor_dicts["bodycontactfrc"]),
    ]
    
    if include_geom_pos:
         SENSOR_GROUPS_CONFIG.append(("geom_pos", sensor_dicts["geom_pos"]))
    
    # Add position estimates
    if enable_position_estimation and position_estimator_segments:
        pos_estimate_dict = {}
        quat_estimate_dict = {}
        vel_estimate_dict = {}
        for seg_id in position_estimator_segments:
            pos_estimate_dict[f'pos_estimate_{seg_id}'] = pos_estimate_arrays[seg_id][:final_length]
            quat_estimate_dict[f'quat_estimate_{seg_id}'] = quat_estimate_arrays[seg_id][:final_length]
            vel_estimate_dict[f'vel_estimate_{seg_id}'] = vel_estimate_arrays[seg_id][:final_length]
        SENSOR_GROUPS_CONFIG.append(("pos_estimate", pos_estimate_dict))
        SENSOR_GROUPS_CONFIG.append(("quat_estimate", quat_estimate_dict))
        SENSOR_GROUPS_CONFIG.append(("vel_estimate", vel_estimate_dict))

    final_wide_df = create_single_polars_dataframe(
        SENSOR_GROUPS_CONFIG, 
        time_series_data, 
        final_length
    )
    
    # --- Video speichern ---
    if record_video and frames and video_path:
        try:
            # Stelle sicher, dass das Verzeichnis existiert
            import os
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            
            print(f"Speichere Video nach {video_path}...")
            with imageio.get_writer(video_path, fps=video_fps, macro_block_size=None) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"Video erfolgreich gespeichert: {video_path}")
        except Exception as e:
            print(f"Fehler beim Speichern des Videos: {e}")
    
    return final_wide_df

# --- 3. Controller-Templates (Beispiele) ---

def static_controller(model: mj.MjModel, data: mj.MjData, current_time: float, step_index: int):
    """Setzt eine konstante Seilkraft (0.2) auf den ersten Aktuator."""
    # Beispiel: Nur den ersten Aktuator setzen

    #data.ctrl[0] = 0.2
    data.ctrl[1] = 0.3

def ramped_controller(model: mj.MjModel, data: mj.MjData, current_time: float, step_index: int):
    """Setzt eine linear ansteigende Kraft (0.0 bis 1.0 über 10 Sekunden) auf den ersten Aktuator."""
    max_time = 2.0
    max_force = -10.0
    force = (current_time / max_time) * max_force
    force = min(force, max_force)  # Begrenze auf max_force
    data.ctrl[0] = force

def sine_controller(model: mj.MjModel, data: mj.MjData, current_time: float, step_index: int):
    """Setzt eine sinusförmige Kraft (Amplitude 0.5, Frequenz 0.5 Hz) auf den ersten Aktuator."""

    amplitude = 0.5
    frequency = 2.0 * math.pi * 0.5 
    data.ctrl[0] = amplitude * np.sin(frequency * current_time)

# Diese Funktionen werden SPÄTER im Loop aufgerufen
def setup_cylinder(worldbody, pos, size, euler, **kwargs):
    body = worldbody.add_body(name="cylinder_obj", pos=pos)
    body.add_geom(
        name="cyl_geom",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        size=size,  # Erwartet [radius, half_length, unused]
        euler=euler,
        rgba=[0.2, 0.8, 0.5, 1],
        density=1000
    )

def setup_box(worldbody, pos, size, euler, **kwargs):
    body = worldbody.add_body(name="box_obj", pos=pos)
    body.add_geom(
        name="box_geom",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=size,  # Erwartet [x_half, y_half, z_half]
        euler=euler,
        rgba=[0.8, 0.2, 0.2, 1],
        density=1000
    )

def generate_grid_configs(variable_params: Dict[str, Any], fixed_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generiert eine Liste von Konfigurations-Dictionaries aus einem Grid."""
    
    # Trenne die Schlüssel und Werte, um sie itertools.product zu übergeben
    keys = list(variable_params.keys())
    values = [
        # Wenn der Wert ein Dictionary ist (wie bei 'controller'), verwende dessen Werte
        list(v.values()) if isinstance(v, dict) else v
        for v in variable_params.values()
    ]
    
    # Extrahieren der Controllernamen separat (für die ID-Generierung)
    controller_names = list(variable_params.get("controller", {}).keys())

    SIM_CONFIGS = []
    run_counter = 1
    
    # itertools.product erzeugt alle Kombinationen der Werte
    for combination in itertools.product(*values):
        
        # Erstelle ein Dict aus der aktuellen Kombination
        config = dict(zip(keys, combination))
        
        # Hinzufügen der festen Parameter
        config.update(fixed_params)
        
        # Wenn 'controller' im Grid ist, finde den passenden Namen für die ID
        ctrl_name = "Custom"
        if "controller" in keys:
            # Finde den Namen des Controllers anhand seines Funktions-/Objektwerts
            # (Dies ist etwas komplex, da man den Wert zurück auf den Schlüssel mappen muss)
            # Wir machen es einfacher, indem wir annehmen, dass 'controller' die letzte Variable ist:
            if isinstance(variable_params["controller"], dict):
                ctrl_value = config["controller"]
                
                # Finde den Namen, der zum Wert gehört
                ctrl_name = next((name for name, func in variable_params["controller"].items() if func == ctrl_value), "Unknown")
            
        # Generiere die eindeutige ID
        id_parts = [
            ctrl_name,
            f"L{config['L_target']:.2f}",
            f"T{config['sim_time']:.1f}",
            f"d{config['base_d']:.3f}",
            # ... füge weitere wichtige Parameter hinzu
        ]
        config["id"] = f"Run_{run_counter:03d}_{'_'.join(id_parts)}"
        
        SIM_CONFIGS.append(config)
        run_counter += 1
        
    return SIM_CONFIGS

def generate_hybrid_grid_configs(common_params, geom_scenarios, fixed_params):
    configs = []
    run_counter = 1

    # 1. Schritt: Erzeuge Grid für die gemeinsamen Parameter (L_target, controller, etc.)
    common_keys = list(common_params.keys())
    # Sonderbehandlung für Controller (wir wollen die Values, nicht Keys, aber Namen für ID)
    common_values = []
    for k, v in common_params.items():
        if k == "controller" and isinstance(v, dict):
            common_values.append(list(v.items())) # Speichert (Name, Funktion) Tupel
        else:
            common_values.append(v)

    # Iteriere über die Basis-Parameter
    for common_prod in itertools.product(*common_values):
        
        # Basis-Config Dictionary bauen
        base_config = {}
        ctrl_name_id = ""
        
        for i, key in enumerate(common_keys):
            val = common_prod[i]
            if key == "controller":
                # Tuple entpacken: (Name, Funktion)
                ctrl_name_id = val[0]
                base_config[key] = val[1]
            else:
                base_config[key] = val

        # 2. Schritt: Für JEDE Basis-Config, iteriere durch die Geometrie-Szenarien
        for scenario in geom_scenarios:
            geom_func = scenario["setup_func"]
            geom_name = scenario["obj_name"]
            
            # Hole die spezifischen Parameter für dieses Szenario (size, pos, euler)
            scen_params = scenario["params"]
            scen_keys = list(scen_params.keys())
            scen_values = list(scen_params.values())
            
            # Mini-Grid für dieses Szenario
            for geom_prod in itertools.product(*scen_values):
                
                # Kopiere Basis-Config, damit wir sie nicht überschreiben
                final_config = base_config.copy()
                final_config.update(fixed_params)
                
                # Füge Geometrie-Daten hinzu
                final_config["geom_func"] = geom_func
                
                # Geometrie-Parameter einzeln ins Config-Dict packen UND in ein 'geom_kwargs' Dict
                geom_kwargs = {}
                geom_id_parts = [geom_name]
                
                for i, key in enumerate(scen_keys):
                    val = geom_prod[i]
                    geom_kwargs[key] = val # Für den Funktionsaufruf später
                    
                    # ID Teil generieren (z.B. Size -> S0.1)
                    if key == "size":
                        s_str = "-".join([f"{x:.2f}" for x in val])
                        geom_id_parts.append(f"Sz{s_str}")
                    elif key == "pos":
                        # Optional, wenn Pos wichtig für ID ist
                        pass 

                final_config["geom_kwargs"] = geom_kwargs
                
                # ID erstellen
                id_parts = [
                    ctrl_name_id,
                    "_".join(geom_id_parts),
                    f"L{base_config['L_target']:.2f}",
                    f"T{base_config['sim_time']:.1f}"
                ]
                final_config["id"] = f"Run_{run_counter:03d}_{'_'.join(id_parts)}"
                
                configs.append(final_config)
                run_counter += 1
                
    return configs



def format_value_for_print(value: Any) -> str:
    """Konvertiert Listen/Arrays in einen kompakten, lesbaren String."""
    if isinstance(value, (list, tuple, np.ndarray)):
        # Runde Floats und konvertiere zu String: [0.10, 0.20]
        return "[" + ", ".join([f"{x:.2f}" for x in value]) + "]"
    
    if isinstance(value, float):
        return f"{value:.3f}"
        
    # Wenn es der String-Name der Funktion ist (z.B. 'setup_cylinder')
    if isinstance(value, str):
        return value.replace('setup_', '')
        
    return str(value)


def print_configs_formatted(config_list: List[Dict[str, Any]], preview_limit: int = 5, print_all: bool = False):
    """
    Gibt eine formatierte Vorschau der Konfigurationsliste auf der Konsole aus.
    Enthält nun Details zur variablen Geometrie.
    """
    
    print("\n" + "="*80)
    print(f"📄 VORSCHAU DER KONFIGURATIONEN ({len(config_list)} Läufe) 📄")
    print("="*80)

    if print_all:
        preview_limit = len(config_list)
    
    for i, config in enumerate(config_list):
        if i >= preview_limit:
            print(f"  ... und {len(config_list) - i} weitere Konfigurationen.")
            break
            
        # --- 1. Controller Name ---
        ctrl = config.get('controller')
        ctrl_name = ctrl.__name__ if hasattr(ctrl, '__name__') else str(ctrl)

        # --- 2. Geometrie-Informationen ---
        
        # Geometrie-Setup-Funktion
        geom_func = config.get('geom_func')
        geom_type_name = geom_func.__name__.replace('setup_', '') if hasattr(geom_func, '__name__') else "Unbekannt"
        
        # Geometrie-Parameter (pos, size, euler)
        geom_kwargs = config.get('geom_kwargs', {})
        
        geom_pos_str = format_value_for_print(geom_kwargs.get('pos', 'N/A'))
        geom_size_str = format_value_for_print(geom_kwargs.get('size', 'N/A'))
        geom_euler_str = format_value_for_print(geom_kwargs.get('euler', 'N/A'))
        
        # --- Ausgabe ---
        print(f"[{i+1}/{len(config_list)}] ID: {config['id']}")
        
        # Allgemeine Parameter
        print(f"  > Modell: L_target={config.get('L_target', 'N/A'):.2f}, base_d={config.get('base_d', 'N/A'):.3f}")
        print(f"  > Kontext: Time={config.get('sim_time', 'N/A'):.1f}, Controller={ctrl_name}")
        
        # Geometrie-Details
        print(f"  > OBJEKT ({geom_type_name.upper()}):")
        print(f"      Pos: {geom_pos_str}, Größe: {geom_size_str}, Euler: {geom_euler_str}")
        
    print("\n" + "="*80)

def save_configs_to_json(
    config_list: List[Dict[str, Any]], 
    filename: str = "simulation_configs.json", 
    indent: int = 4
):
    """
    Exportiert die Konfigurationsliste in eine JSON-Datei.
    Nicht-serialisierbare Objekte (Funktionen, NumPy-Arrays) werden in Strings 
    oder standardmäßige Python-Typen konvertiert.
    
    Args:
        config_list: Die Liste der Konfigurations-Dictionaries (SIM_CONFIGS).
        filename: Der Name der Exportdatei.
        indent: Die Anzahl der Leerzeichen für die JSON-Einrückung.
    """
    
    exportable_list = []
    
    for config in config_list:
        # Erstelle eine Kopie des Dictionarys, um das Original nicht zu verändern
        export_config = config.copy()
        
        # --- 1. Controller behandeln (Funktion/Objekt) ---
        if 'controller' in export_config:
            ctrl = export_config['controller']
            # Speichere den Namen der Funktion/Klasse als String
            ctrl_str = ctrl.__name__ if hasattr(ctrl, '__name__') else str(ctrl)
            export_config['controller_info_str'] = ctrl_str
            # Entferne das nicht-serialisierbare Objekt
            del export_config['controller']

        # --- 2. Geometrie-Funktion behandeln ---
        if 'geom_func' in export_config:
            geom_func = export_config['geom_func']
            # Speichere den Namen der Funktion als String
            geom_func_str = geom_func.__name__ if hasattr(geom_func, '__name__') else "Unbekannte Funktion"
            export_config['geom_func_info_str'] = geom_func_str
            # Entferne das nicht-serialisierbare Objekt
            del export_config['geom_func']

        # --- 3. Geometrie-Argumente (geom_kwargs) und andere Listen behandeln ---
        # Dies ist der kritische Schritt, um NumPy-Arrays in Listen zu konvertieren.
        
        # Hilfsfunktion zur rekursiven Konvertierung von NumPy-Typen
        def convert_to_serializable(item):
            if isinstance(item, (list, tuple, np.ndarray)):
                # Gehe rekursiv Listen/Arrays durch
                return [convert_to_serializable(x) for x in item]
            elif isinstance(item, dict):
                # Gehe rekursiv Dictionarys durch
                return {k: convert_to_serializable(v) for k, v in item.items()}
            elif isinstance(item, (np.float32, np.float64, np.generic)):
                # Konvertiere NumPy-Floats zu nativem Python-Float
                return float(item)
            elif isinstance(item, (np.int32, np.int64)):
                # Konvertiere NumPy-Integers zu nativem Python-Integer
                return int(item)
            else:
                return item

        # Wende die Konvertierung auf die geom_kwargs an (falls vorhanden)
        if 'geom_kwargs' in export_config:
            export_config['geom_kwargs'] = convert_to_serializable(export_config['geom_kwargs'])
            
        # Wende die Konvertierung auch auf andere Top-Level-Werte an, die Listen/Arrays sein könnten
        # (z.B. L_target, die aus np.array erstellt wurden)
        for key, value in export_config.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                export_config[key] = convert_to_serializable(value)
        
        
        # --- 4. Hinzufügen zur Exportliste ---
        exportable_list.append(export_config)

    # --- JSON-Export ---
    try:
        with open(filename, 'w') as f:
            json.dump(exportable_list, f, indent=indent)
        print(f"\n✅ ERFOLG: Konfigurationen erfolgreich exportiert nach: {filename}")
    except Exception as e:
        print(f"\n❌ FEHLER beim Exportieren nach JSON ({filename}): {e}")
        print("Stellen Sie sicher, dass keine nicht-serialisierbaren Typen (z.B. komplexe Objekte) übrig geblieben sind.")

def get_video_path(run_id: str, base_dir: str = "build/experiments") -> str:
    """Erzeugt den Pfad für das Video basierend auf run_id."""
    return f"{base_dir}/{run_id}/video.mp4"
