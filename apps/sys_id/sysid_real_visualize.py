#!/usr/bin/env python3
"""
SpiRob – Visualisierung der identifizierten Parameter mit echten Eingangsdaten.

Liest sysid_real_params.json und wendet die Parameter auf das Modell an.
Liest sys_id_combined.parquet, interpoliert die Kräfte und legt diese an.
Startet den passiven Viewer, um die Trajektorie abzuspielen.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco as mj
import mujoco.viewer as mj_viewer
import numpy as np
import polars as pl
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

import math_spirob.spirob_generator as sg

# ── Modell-Geometrie (feststehend - analog zu sysid_real.py) ────────
L_TARGET = 0.44
BASE_D = 0.1
TIP_D = 0.03
DELTA_THETA_DEG = 30.0

# ====================================================================
#  Hilfsfunktionen
# ====================================================================

def make_model() -> mj.MjModel:
    xml_path = Path(__file__).resolve().parent / "spiral_chain_wo_cylinder.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"XML_Datei nicht gefunden: {xml_path}")
        
    spec = mj.MjSpec.from_file(str(xml_path))
    
    #cylinder = spec.worldbody.add_body(name="cylinder", pos=[-0.11, 0.00, 0.11+0.053])
    #cylinder.add_geom(
    #    name="cyl_geom",
    #    type=mj.mjtGeom.mjGEOM_CYLINDER,
    #    size=[0.05, 0.15, 0.05],
    #    euler=[90, 0, 0],
    #    rgba=[0.2, 0.8, 0.5, 1],
    #    density=1000,
    #)
    
    return spec.compile()

def set_params(model: mj.MjModel,
               stiffness: list[float],
               damping: list[float],
               tendon_stiffness: list[float]) -> None:
    """Setzt individuelle physikalische Parameter für Joints und Tendons."""
    for i in range(model.njnt):
        model.jnt_stiffness[i] = stiffness[i]
        dof = model.jnt_dofadr[i]
        model.dof_damping[dof] = damping[i]
    for i in range(model.ntendon):
        model.tendon_stiffness[i] = tendon_stiffness[i]

def smooth_data(data: np.ndarray, fps: float, cutoff: float = 5.0) -> np.ndarray:
    """Applies a Butterworth low-pass filter to remote tracking noise."""
    nyq = 0.5 * fps
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        return data
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    if len(data) <= 15:
        return data
    return filtfilt(b, a, data)

# ====================================================================
#  Hauptmethode
# ====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Visualisierung Real-SysID")
    ap.add_argument("--test-timeout", type=float, default=None, 
                    help="Simulation nach X Sekunden abbrechen (für automatische Tests)")
    ap.add_argument("--invert-angles", action="store_true", 
                    help="Spiegelt die Start-Gelenkwinkel der Trajektorie (* -1)")
    ap.add_argument("--swap-forces", action="store_true", 
                    help="Vertauscht Tendon 0 und Tendon 1 (Kraft 0 auf Sehne 1, etc.)")
    args = ap.parse_args()

    # Ordnerstruktur
    base_dir = Path(__file__).resolve().parent
    build_dir = base_dir / "build"
    params_path = build_dir / "sysid_real_linear_profile_params.json"
    data_path = build_dir / "sys_id_auto_GX010070.parquet"

    if not params_path.exists():
        print(f"ERROR: Parameter-Datei nicht gefunden: {params_path}")
        return
    if not data_path.exists():
        print(f"ERROR: Daten-Datei nicht gefunden: {data_path}")
        return

    # 1. Parameter laden
    print("=" * 55)
    print(f"Lade Parameter aus {params_path.name}...")
    with open(params_path, "r") as f:
        params = json.load(f)
    
    stiffness = params["stiffness"]
    damping = params["damping"]
    tendon_stiffness = params["tendon_stiffness"]

    # 2. Daten laden und interpolieren
    print(f"Lade Realdaten aus {data_path.name}...")
    df = pl.read_parquet(data_path)
    
    joint_cols = [f"joint_{i}_deg" for i in range(1, 14)]
    df = df.drop_nulls(subset=["global_timestamp_s", "meas_force_0_N", "meas_force_1_N"] + joint_cols)
    
    if len(df) == 0:
        print("ERROR: Keine gültigen Datenreihen gefunden.")
        return

    raw_time = df["global_timestamp_s"].to_numpy()
    raw_time = raw_time - raw_time[0]
    sim_time_max = raw_time[-1]
    
    fps = 1.0 / np.median(np.diff(raw_time))
    
    f0 = df["meas_force_0_N"].to_numpy()
    f1 = df["meas_force_1_N"].to_numpy()

    qpos_data = np.zeros((len(df), 13))
    for i, col in enumerate(joint_cols):
        # Winkel optional invertieren via Flag:
        if args.invert_angles:
            rads = np.deg2rad(-1.0 * df[col].to_numpy())
        else:
            rads = np.deg2rad(df[col].to_numpy())
        qpos_data[:, i] = smooth_data(rads, fps, cutoff=10.0)
    
    start_qpos_measured = qpos_data[0][::-1]
    print(f"Start-Gelenkwinkel (gemessen): {np.rad2deg(start_qpos_measured)}")

    if args.swap_forces:
        force_interp_0 = interp1d(raw_time, f1, kind='linear', fill_value=(f1[0], f1[-1]), bounds_error=False)
        force_interp_1 = interp1d(raw_time, f0, kind='linear', fill_value=(f0[0], f0[-1]), bounds_error=False)
    else:
        force_interp_0 = interp1d(raw_time, f0, kind='linear', fill_value=(f0[0], f0[-1]), bounds_error=False)
        force_interp_1 = interp1d(raw_time, f1, kind='linear', fill_value=(f1[0], f1[-1]), bounds_error=False)

    print(f"  Datensätze: {len(qpos_data)}, FPS: {fps:.1f}, Dauer: {sim_time_max:.2f}s")
    print("=" * 55)

    # 3. Modell erstellen und Parameter setzen
    model = make_model()
    model.opt.timestep = 0.004
    model.opt.iterations = 20
    set_params(model, stiffness, damping, tendon_stiffness)
    data = mj.MjData(model)

    def controller(m: mj.MjModel, d: mj.MjData, t: float) -> None:
        """Nutzt die interpolierten Kräfte."""
        d.ctrl[0] = -force_interp_0(t)
        d.ctrl[1] = -force_interp_1(t)

    print("Physikalisches Start-Gleichgewicht (Settling) im Viewer anzeigen...")
    
    # 4. Simulation / Viewer Loop
    program_start = time.time()
    
    with mj_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            time.sleep(1.0) # 5.1s ist recht lang, ggf. anpassen
            
            # 1. Kompletter Reset aller internen Zustände (löscht alte Geschwindigkeiten & setzt time=0)
            mj.mj_resetData(model, data)
            
            # 2. Startpose und Startkräfte setzen
            data.qpos[:] = start_qpos_measured
            print(f"Setzte Start-Gelenkwinkel: {np.rad2deg(data.qpos[:13])}")
            
            data.ctrl[0] = -force_interp_0(0.0)
            data.ctrl[1] = -force_interp_1(0.0)
            print(f"Setzte start_forces: {data.ctrl[0]:.2f} N, {data.ctrl[1]:.2f} N")
            
            # 3. Berechnen der Vorwärtskinematik und der abhängigen Matrizen VOR dem ersten Schritt
            mj.mj_forward(model, data)
            
            # --- 4. Kurze Einschwingphase (Settle-Phase) ---
            # Lässt den Roboter seine exakte Ruheposition für die anliegenden Kräfte finden
            print("  Phase: Einschwingen...")
            for _ in range(1000): # ca. 100 Schritte (z.B. 0.2s bei dt=0.002)
                step_start = time.time()
                mj.mj_step(model, data)
                # Optional: Geschwindigkeiten nullen, um Schwingungen zu dämpfen
                data.qvel[:] = 0.0 
                viewer.sync()
                # Echtzeitsynchronisation
                dt_left = model.opt.timestep - (time.time() - step_start)
                if dt_left > 0:
                    time.sleep(dt_left)

            
            # Zeit nach dem Einschwingen wieder auf 0 setzen für die Trajektorie
            data.time = 0.0 
            
            # --- Phase: Trajektorie abfahren ---
            print("  Phase: Aufgenommene Trajektorie fahrt ab...")
            while viewer.is_running() and data.time < sim_time_max:
                step_start = time.time()
                
                # Controller wird nun garantiert bei data.time = 0.0 gestartet
                controller(model, data, data.time)
                mj.mj_step(model, data)
                viewer.sync()
                
                # Echtzeitsynchronisation
                dt_left = model.opt.timestep - (time.time() - step_start)
                if dt_left > 0:
                    time.sleep(dt_left)

    print("Visualisierung abgeschlossen.")

if __name__ == "__main__":
    main()
