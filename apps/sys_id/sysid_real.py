#!/usr/bin/env python3
"""
SpiRob – Real Data System Identification (Multi-Parameter)
"""

import argparse
import time
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np
import polars as pl
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from scipy.signal import butter, filtfilt

import math_spirob.spirob_generator as sg

# ── Model Geometry (fixed) ──────────────────────────────────────────
L_TARGET = 0.44
BASE_D = 0.1
TIP_D = 0.03
DELTA_THETA_DEG = 30.0

# ── Initialization Values for Optimization ──────────────────────────
INIT_BASE_STIFFNESS = 0.05
INIT_BASE_DAMPING = 0.05
INIT_BASE_TENDON_STIFFNESS = 10.0

# ── Optimization Bounds ──────────────────────────────────────────────
BOUNDS_STIFFNESS = (0.01, 100.0)
BOUNDS_DAMPING = (0.01, 100.0)
BOUNDS_TENDON_STIFFNESS = (1, 100.0)

# ── Optimization Cost Function Selector ──────────────────────────────
# 'sysid_multi'   : Weighted position + 0.5 * weighted velocity (from sysid_multi.py)
# 'position_only' : Standard uniform position MSE
ACTIVE_COST_FUNCTION = "sysid_multi"

# ====================================================================
#  Helper Functions
# ====================================================================

def make_model() -> mj.MjModel:
    xml_path = Path(__file__).resolve().parent / "spiral_chain_wo_cylinder.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    return mj.MjModel.from_xml_path(str(xml_path))

def set_params(model: mj.MjModel,
               stiffness: np.ndarray,
               damping: np.ndarray,
               tendon_stiffness: np.ndarray) -> None:
    """Sets individual physical parameters (arrays) for all joints and tendons."""
    for i in range(model.njnt):
        model.jnt_stiffness[i] = stiffness[i]
        dof = model.jnt_dofadr[i]
        model.dof_damping[dof] = damping[i]
    for i in range(model.ntendon):
        model.tendon_stiffness[i] = tendon_stiffness[i]

def decode_params(x_scaled: np.ndarray, njnt: int, ntendon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts 3 parameter arrays from the flat scalar vector."""
    stiffness = x_scaled[0:njnt]
    damping = x_scaled[njnt:2*njnt]
    tendon_stiffness = x_scaled[2*njnt:2*njnt+ntendon]
    return stiffness, damping, tendon_stiffness

# ====================================================================
#  Multiprocessing Globals & Profiling
# ====================================================================
import atexit

_LOCAL_MODEL = None
_LOCAL_DATA = None
_FORCE_INTERP_0 = None
_FORCE_INTERP_1 = None
_SIM_TIMESTEPS = None
_GT_QPOS = None
_GT_QVEL = None

# Profiling variables per worker
_PROF_EVALS = 0
_PROF_TIME_SETUP = 0.0
_PROF_TIME_SIM = 0.0
_PROF_TIME_MSE = 0.0

def _print_worker_stats():
    if _PROF_EVALS > 0:
        total = _PROF_TIME_SETUP + _PROF_TIME_SIM + _PROF_TIME_MSE
        print(f"  [Worker Profiling] Evals: {_PROF_EVALS} | Setup: {_PROF_TIME_SETUP:.2f}s | Sim: {_PROF_TIME_SIM:.2f}s | MSE: {_PROF_TIME_MSE:.2f}s | Total: {total:.2f}s")
        
atexit.register(_print_worker_stats)

def get_local_model_and_data() -> tuple[mj.MjModel, mj.MjData]:
    global _LOCAL_MODEL, _LOCAL_DATA
    if _LOCAL_MODEL is None:
        # Model & MjData created ONLY ONCE per worker core to save time!
        _LOCAL_MODEL = make_model()
        _LOCAL_DATA = mj.MjData(_LOCAL_MODEL)
    return _LOCAL_MODEL, _LOCAL_DATA

def controller(model: mj.MjModel, data: mj.MjData, t: float) -> None:
    """Uses real interpolated measured force for tendons."""
    if _FORCE_INTERP_0 is not None and _FORCE_INTERP_1 is not None:
        data.ctrl[0] = -_FORCE_INTERP_0(t) # Tendon actuators pull when negative
        data.ctrl[1] = -_FORCE_INTERP_1(t)

def simulate_and_sample(model: mj.MjModel, 
                        data: mj.MjData,
                        sim_time: float, 
                        gt_qpos_init: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates the model and samples the position (qpos) and velocity (qvel) exactly at the recorded 
    real-world timestamps for an exact match.
    """
    global _PROF_TIME_SIM
    t_start = time.perf_counter()
    
    mj.mj_resetData(model, data)
    model.opt.timestep = 0.02
    
    # Set the initial state from exactly frame 0 of our real world target
    data.qpos[:] = gt_qpos_init
    data.qvel[:] = np.zeros(model.nv)
    mj.mj_forward(model, data)
    
    # the timestep target logic
    dt = model.opt.timestep
    total_steps = int(sim_time / dt)
    
    qpos_history = []
    qvel_history = []

    for step in range(total_steps + 1):
        qpos_history.append(data.qpos.copy())
        qvel_history.append(data.qvel.copy())
        
        controller(model, data, data.time)
        mj.mj_step(model, data)

    _PROF_TIME_SIM += (time.perf_counter() - t_start)
    return np.array(qpos_history), np.array(qvel_history)

def global_objective(x: np.ndarray, scale: np.ndarray, sim_time: float, njnt: int, ntendon: int) -> float:
    model, data = get_local_model_and_data()
    return cost_function(x, scale, sim_time, model, data, njnt, ntendon)

def cost_function(params_norm: np.ndarray,
                  scale: np.ndarray,
                  sim_time: float,
                  model: mj.MjModel,
                  data: mj.MjData,
                  njnt: int,
                  ntendon: int) -> float:
    global _PROF_EVALS, _PROF_TIME_SETUP, _PROF_TIME_MSE, _PROF_TIME_SIM
    _PROF_EVALS += 1
    
    if _PROF_EVALS % 1000 == 0:
        total = _PROF_TIME_SETUP + _PROF_TIME_SIM + _PROF_TIME_MSE
        print(f"  [Worker Profiling | ~1000 Evals] Setup: {_PROF_TIME_SETUP:.2f}s ({_PROF_TIME_SETUP/total*100:.1f}%) | "
              f"Sim: {_PROF_TIME_SIM:.2f}s ({_PROF_TIME_SIM/total*100:.1f}%) | "
              f"MSE: {_PROF_TIME_MSE:.2f}s ({_PROF_TIME_MSE/total*100:.1f}%)")
              
    t_start = time.perf_counter()
    
    phys = params_norm * scale
    
    if np.any(phys <= 0):
        _PROF_TIME_SETUP += (time.perf_counter() - t_start)
        return 1e6

    stiff, damp, t_stiff = decode_params(phys, njnt, ntendon)

    try:
        set_params(model, stiff, damp, t_stiff)
        _PROF_TIME_SETUP += (time.perf_counter() - t_start)
        
        est_qpos, est_qvel = simulate_and_sample(model, data, sim_time, _GT_QPOS[0])
        
        t_mse = time.perf_counter()
    except Exception:
        _PROF_TIME_SETUP += (time.perf_counter() - t_start)
        return 1e6

    # Evaluation
    len_gt = len(_GT_QPOS)
    len_est = len(est_qpos)
    
    if len_gt == 0 or len_est == 0:
        _PROF_TIME_MSE += (time.perf_counter() - t_mse)
        return 1e6
        
    # Entferne gleichmäßig Datenpunkte aus dem Array mit höherer Samplingrate
    if len_gt > len_est:
        idx = np.round(np.linspace(0, len_gt - 1, len_est)).astype(int)
        gt_pos_match = _GT_QPOS[idx]
        gt_vel_match = _GT_QVEL[idx]
        est_pos_match = est_qpos
        est_vel_match = est_qvel
        n = len_est
    elif len_est > len_gt:
        idx = np.round(np.linspace(0, len_est - 1, len_gt)).astype(int)
        est_pos_match = est_qpos[idx]
        est_vel_match = est_qvel[idx]
        gt_pos_match = _GT_QPOS
        gt_vel_match = _GT_QVEL
        n = len_gt
    else:
        gt_pos_match = _GT_QPOS
        gt_vel_match = _GT_QVEL
        est_pos_match = est_qpos
        est_vel_match = est_qvel
        n = len_gt

    # 1. Umrechnung in Grad
    err_pos_deg = np.rad2deg(gt_pos_match - est_pos_match)
    err_vel_deg = np.rad2deg(gt_vel_match - est_vel_match)
        
    if ACTIVE_COST_FUNCTION == "sysid_multi":
        w = np.linspace(1.5, 0.5, n).reshape(-1, 1)
        # 2. RMSE statt MSE
        rmse_pos = np.sqrt(np.mean(w * err_pos_deg ** 2))
        rmse_vel = np.sqrt(np.mean(w * err_vel_deg ** 2))
        cost = rmse_pos + 0.1 * rmse_vel
    elif ACTIVE_COST_FUNCTION == "position_only":
        cost = np.sqrt(np.mean(err_pos_deg ** 2))
    else:
        cost = np.sqrt(np.mean(err_pos_deg ** 2)) # fallback
        
    _PROF_TIME_MSE += (time.perf_counter() - t_mse)
    return cost

# ====================================================================
#  Data Preparation
# ====================================================================

def smooth_data(data: np.ndarray, fps: float, cutoff: float = 5.0) -> np.ndarray:
    """Applies a Butterworth low-pass filter to remote tracking noise."""
    nyq = 0.5 * fps
    normal_cutoff = cutoff / nyq
    # Fallback if cutoff is too high for the fps
    if normal_cutoff >= 1.0:
        return data
        
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    # Handle short data series
    if len(data) <= 15:
        return data
        
    return filtfilt(b, a, data)

def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-Parameter System ID on Real Data")
    ap.add_argument("--sim-time", type=float, default=-1.0, help="Max sim time against data (-1 to use full trace length)")
    ap.add_argument("--maxiter", type=int, default=50) 
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=10, help="Parallel processes count")
    ap.add_argument("--force-cutoff", type=float, default=5.0, help="Cutoff freq for force smoothing (Hz)")
    args = ap.parse_args()

    data_path = Path(__file__).resolve().parent / "build" / "sys_id_auto_gx10069.parquet"
    if not data_path.exists():
        print(f"ERROR: File not found at {data_path}")
        return

    print("=" * 55)
    print("Lade reale Daten...")
    df = pl.read_parquet(data_path)
    
    # Drop rows with nulls in crucial columns
    joint_cols = [f"joint_{i}_deg" for i in range(1, 14)]
    df = df.drop_nulls(subset=["global_timestamp_s", "meas_force_0_N", "meas_force_1_N"] + joint_cols)
    
    if len(df) == 0:
        print("ERROR: No valid rows remaining after null dropping.")
        return

    # Reset time to start at 0
    raw_time = df["global_timestamp_s"].to_numpy()
    raw_time = raw_time - raw_time[0]
    
    fps = 1.0 / np.median(np.diff(raw_time))
    
    # Extract forces
    f0_raw = df["meas_force_0_N"].to_numpy()
    f1_raw = df["meas_force_1_N"].to_numpy()
    
    # Glätten der gemessenen Seilkräfte mit angegebener Cutoff-Frequenz
    f0 = smooth_data(f0_raw, fps, cutoff=args.force_cutoff)
    f1 = smooth_data(f1_raw, fps, cutoff=args.force_cutoff)

    # Convert angular data (joint_X_deg -> radians)
    qpos_data = np.zeros((len(df), 13))
    for i, col in enumerate(joint_cols):
        # We invert real world angles or just convert if matching mujoco polarity directly
        # For now assume degrees map perfectly to radians
        rads = np.deg2rad(df[col].to_numpy())
        qpos_data[:, i] = smooth_data(rads, fps, cutoff=10.0) # Light smoothing on positions

    print(f"  Datensätze: {len(qpos_data)}, FPS={fps:.1f}, Dauer={raw_time[-1]:.2f}s")
    
    # Truncate to sim_time
    max_t = raw_time[-1] if args.sim_time <= 0 else min(args.sim_time, raw_time[-1])
    mask = raw_time <= max_t
    t_target = raw_time[mask]
    qpos_target = qpos_data[mask]
    f0_target = f0[mask]
    f1_target = f1[mask]
    
    def remove_outliers_mad(data_1d: np.ndarray, thresh: float = 3.5) -> np.ndarray:
        arr = data_1d.copy()
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad < 1e-6: mad = 1e-6
        z_scores = 0.6745 * np.abs(arr - median) / mad
        for idx in np.where(z_scores > thresh)[0]:
            if 0 < idx < len(arr) - 1:
                arr[idx] = (arr[idx-1] + arr[idx+1]) / 2.0
            elif idx == 0:
                arr[idx] = arr[1]
            else:
                arr[idx] = arr[-2]
        return arr

    # Calculate velocity approximation from real position data using finite differences
    qvel_target = np.zeros_like(qpos_target)
    if len(t_target) > 1:
        for i in range(qpos_target.shape[1]):
            raw_vel = np.gradient(qpos_target[:, i], t_target)
            
            # 1. Ausreißer entfernen (Peaks durch Mittelwert der Nachbarn ersetzen)
            cleaned_vel = remove_outliers_mad(raw_vel, thresh=3.5)
            
            # 2. Mit Butterworth tiefpass-filtern
            # cutoff relativ aggressiv (z.B. 2.0 Hz), da aus Ableitung noch viel Rauschen entsteht
            smooth_vel = smooth_data(cleaned_vel, fps=fps, cutoff=2.0)
            
            qvel_target[:, i] = smooth_vel

    qvel_out_path = Path(__file__).resolve().parent / "build" / "sysid_real_qvel.parquet"
    qvel_data = {"global_timestamp_s": t_target}
    for i in range(qvel_target.shape[1]):
        qvel_data[f"joint_{i + 1}_vel_rad_s"] = qvel_target[:, i]
    pl.DataFrame(qvel_data).write_parquet(qvel_out_path)
    print(f"  [Info] Numerische Geschwindigkeiten exportiert: {qvel_out_path.name}")

    # Setting global values for multiprocessing memory efficiency
    global _GT_QPOS, _GT_QVEL, _SIM_TIMESTEPS, _FORCE_INTERP_0, _FORCE_INTERP_1
    _GT_QPOS = qpos_target
    _GT_QVEL = qvel_target
    _SIM_TIMESTEPS = t_target
    _FORCE_INTERP_0 = interp1d(t_target, f0_target, kind='linear', fill_value=(f0_target[0], f0_target[-1]), bounds_error=False)
    _FORCE_INTERP_1 = interp1d(t_target, f1_target, kind='linear', fill_value=(f1_target[0], f1_target[-1]), bounds_error=False)

    njnt = 13
    ntendon = 2

    # ── 2. Vorbereitung Optimierung ─────────────────────────────────
    scale = np.concatenate([
        np.full(njnt, INIT_BASE_STIFFNESS),
        np.full(njnt, INIT_BASE_DAMPING),
        np.full(ntendon, INIT_BASE_TENDON_STIFFNESS)
    ])

    norm_bounds = []
    for _ in range(njnt): norm_bounds.append((BOUNDS_STIFFNESS[0]/INIT_BASE_STIFFNESS, BOUNDS_STIFFNESS[1]/INIT_BASE_STIFFNESS))
    for _ in range(njnt): norm_bounds.append((BOUNDS_DAMPING[0]/INIT_BASE_DAMPING, BOUNDS_DAMPING[1]/INIT_BASE_DAMPING))
    for _ in range(ntendon): norm_bounds.append((BOUNDS_TENDON_STIFFNESS[0]/INIT_BASE_TENDON_STIFFNESS, BOUNDS_TENDON_STIFFNESS[1]/INIT_BASE_TENDON_STIFFNESS))

    x0 = np.ones(len(scale))

    print("\n" + "=" * 55)
    print(f"Starte Optimierung über {max_t:.2f}s Datentrajektorie")
    print(f"Parameter: {len(scale)} (13x Stiffness, 13x Damping, 2x Tendon Stiffness)")
    print("-" * 55)

    t0 = time.time()
    
    best_x = x0.copy()
    def on_step(xk, convergence=None):
        nonlocal best_x
        best_x = xk.copy()

    try:
        # Note: Use workers=args.workers. In DE, args.workers overrides default
        result = differential_evolution(
            global_objective,
            args=(scale, max_t, njnt, ntendon),
            bounds=norm_bounds,
            x0=x0,
            maxiter=args.maxiter,
            tol=args.tol,
            seed=42,
            polish=True,
            init="sobol",
            popsize=args.workers,
            recombination=0.9,
            workers=args.workers,
            disp=True,
            callback=on_step,
        )
        final_x = result.x
        final_cost = result.fun
        status_msg = result.message
        nfev = result.nfev
    except KeyboardInterrupt:
        print("\n\n[!] Optimierung durch Benutzer abgebrochen (KeyboardInterrupt).")
        print("Berechne Kosten für den bisher besten gefundenen Parametersatz...")
        final_x = best_x
        final_cost = global_objective(final_x, scale, max_t, njnt, ntendon)
        status_msg = "User aborted (KeyboardInterrupt)"
        nfev = "N/A"
        
    elapsed = time.time() - t0

    # ── 4. Ergebnis ────────────────────────────────────────────────
    phys = final_x * scale
    stiff_res, damp_res, t_stiff_res = decode_params(phys, njnt, ntendon)
    
    print("\n" + "=" * 55 + "\nERGEBNIS")
    print(f"  Status  : {status_msg}\n  Cost    : {final_cost:.10e}\n  Aufrufe : {nfev}\n  Dauer   : {elapsed:.1f} s\n")
    
    print("\n  Detaillierte Parameter-Ergebnisse (PRO GELENK / SEHNE):")
    print(f"  {'Index':>5s} | {'Stiffness identified':>25s} | {'Damping identified':>25s}")
    print("  " + "-" * 60)
    for i in range(njnt):
        print(f"  {i:5d} | {stiff_res[i]:25.4f} | {damp_res[i]:25.4f}")

    if ntendon > 0:
        print(f"\n  {'Index':>5s} | {'Tendon Stiff identified':>25s}")
        print("  " + "-" * 35)
        for i in range(ntendon):
            print(f"  {i:5d} | {t_stiff_res[i]:25.2f}")

    print("=" * 55)
    
    # Save the parameters to a file to use in other scripts
    import json
    out_dict = {
        "stiffness": stiff_res.tolist(),
        "damping": damp_res.tolist(),
        "tendon_stiffness": t_stiff_res.tolist(),
        "cost": float(final_cost)
    }
    
    out_file = Path(__file__).resolve().parent / "build" / "sysid_real_params.json"
    with open(out_file, "w") as f:
        json.dump(out_dict, f, indent=4)
        
    print(f"Parameter saved to: {out_file}")

if __name__ == "__main__":
    main()
