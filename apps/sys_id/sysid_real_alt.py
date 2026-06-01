#!/usr/bin/env python3
"""
SpiRob – Real Data System Identification (Alternating Parameter Blocks)
Optimizes Stiffness/Tendon-Stiffness first, then Damping, iteratively.
"""

import argparse
import time
from pathlib import Path
import json

import mujoco as mj
import numpy as np
import polars as pl
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from scipy.signal import butter, filtfilt

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
        _LOCAL_MODEL = make_model()
        _LOCAL_DATA = mj.MjData(_LOCAL_MODEL)
    return _LOCAL_MODEL, _LOCAL_DATA

def controller(model: mj.MjModel, data: mj.MjData, t: float) -> None:
    if _FORCE_INTERP_0 is not None and _FORCE_INTERP_1 is not None:
        data.ctrl[0] = -_FORCE_INTERP_0(t)
        data.ctrl[1] = -_FORCE_INTERP_1(t)

def simulate_and_sample(model: mj.MjModel, 
                        data: mj.MjData,
                        sim_time: float, 
                        gt_qpos_init: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    global _PROF_TIME_SIM
    t_start = time.perf_counter()
    
    mj.mj_resetData(model, data)
    model.opt.timestep = 0.02
    
    data.qpos[:] = gt_qpos_init
    data.qvel[:] = np.zeros(model.nv)
    mj.mj_forward(model, data)
    
    dt = model.opt.timestep
    total_steps = int(sim_time / dt)
    
    sample_idx = 0
    num_samples = len(_SIM_TIMESTEPS)
    
    sampled_qpos = np.zeros((num_samples, model.njnt))
    sampled_qvel = np.zeros((num_samples, model.nv))

    for step in range(total_steps + 1):
        while sample_idx < num_samples and _SIM_TIMESTEPS[sample_idx] <= data.time:
            sampled_qpos[sample_idx, :] = data.qpos.copy()
            sampled_qvel[sample_idx, :] = data.qvel.copy()
            sample_idx += 1
            
        if sample_idx >= num_samples:
            break
            
        controller(model, data, data.time)
        mj.mj_step(model, data)

    _PROF_TIME_SIM += (time.perf_counter() - t_start)
    return sampled_qpos, sampled_qvel

def cost_function_raw(stiff: np.ndarray,
                      damp: np.ndarray,
                      t_stiff: np.ndarray,
                      sim_time: float,
                      model: mj.MjModel,
                      data: mj.MjData) -> float:
    global _PROF_EVALS, _PROF_TIME_SETUP, _PROF_TIME_MSE, _PROF_TIME_SIM
    _PROF_EVALS += 1
    
    t_start = time.perf_counter()
    if np.any(stiff <= 0) or np.any(damp <= 0) or np.any(t_stiff <= 0):
        _PROF_TIME_SETUP += (time.perf_counter() - t_start)
        return 1e6

    try:
        set_params(model, stiff, damp, t_stiff)
        _PROF_TIME_SETUP += (time.perf_counter() - t_start)
        
        est_qpos, est_qvel = simulate_and_sample(model, data, sim_time, _GT_QPOS[0])
        
        t_mse = time.perf_counter()
    except Exception:
        _PROF_TIME_SETUP += (time.perf_counter() - t_start)
        return 1e6

    n = min(len(_GT_QPOS), len(est_qpos))
    if n == 0:
        _PROF_TIME_MSE += (time.perf_counter() - t_mse)
        return 1e6
        
    if ACTIVE_COST_FUNCTION == "sysid_multi":
        w = np.linspace(1.5, 0.5, n).reshape(-1, 1)
        mse_pos = np.mean(w * (_GT_QPOS[:n] - est_qpos[:n]) ** 2)
        mse_vel = np.mean(w * (_GT_QVEL[:n] - est_qvel[:n]) ** 2)
        cost = mse_pos + 0.5 * mse_vel
    else:
        cost = np.mean((_GT_QPOS[:n] - est_qpos[:n]) ** 2)
        
    _PROF_TIME_MSE += (time.perf_counter() - t_mse)
    return cost

def global_objective_stiff(x_b1: np.ndarray, curr_damp_norm: np.ndarray, sim_time: float, njnt: int, ntendon: int) -> float:
    model, data = get_local_model_and_data()
    stiff_scale = np.full(njnt, INIT_BASE_STIFFNESS)
    tstiff_scale = np.full(ntendon, INIT_BASE_TENDON_STIFFNESS)
    damp_scale = np.full(njnt, INIT_BASE_DAMPING)
    
    stiff = x_b1[:njnt] * stiff_scale
    t_stiff = x_b1[njnt:] * tstiff_scale
    damp = curr_damp_norm * damp_scale
    return cost_function_raw(stiff, damp, t_stiff, sim_time, model, data)

def global_objective_damp(x_b2: np.ndarray, curr_b1_norm: np.ndarray, sim_time: float, njnt: int, ntendon: int) -> float:
    model, data = get_local_model_and_data()
    stiff_scale = np.full(njnt, INIT_BASE_STIFFNESS)
    tstiff_scale = np.full(ntendon, INIT_BASE_TENDON_STIFFNESS)
    damp_scale = np.full(njnt, INIT_BASE_DAMPING)
    
    stiff = curr_b1_norm[:njnt] * stiff_scale
    t_stiff = curr_b1_norm[njnt:] * tstiff_scale
    damp = x_b2 * damp_scale
    return cost_function_raw(stiff, damp, t_stiff, sim_time, model, data)

# ====================================================================
#  Data Preparation
# ====================================================================

def smooth_data(data: np.ndarray, fps: float, cutoff: float = 5.0) -> np.ndarray:
    nyq = 0.5 * fps
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        return data
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    if len(data) <= 15:
        return data
    return filtfilt(b, a, data)

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

def main() -> None:
    ap = argparse.ArgumentParser(description="Alternating System ID on Real Data")
    ap.add_argument("--sim-time", type=float, default=-1.0, help="Max sim time (-1 for full)")
    ap.add_argument("--inner-maxiter", type=int, default=10, help="DE maxiter per alternating step") 
    ap.add_argument("--outer-maxiter", type=int, default=5, help="Number of alternating loops")
    ap.add_argument("--outer-tol", type=float, default=0.001, help="Tolerance for outer loop stop")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=10, help="Parallel processes count")
    args = ap.parse_args()

    data_path = Path(__file__).resolve().parent / "build" / "sys_id_auto_gx10069.parquet"
    if not data_path.exists():
        print(f"ERROR: File not found at {data_path}")
        return

    print("=" * 55)
    print("Lade reale Daten...")
    df = pl.read_parquet(data_path)
    
    joint_cols = [f"joint_{i}_deg" for i in range(1, 14)]
    df = df.drop_nulls(subset=["global_timestamp_s", "meas_force_0_N", "meas_force_1_N"] + joint_cols)
    if len(df) == 0:
        print("ERROR: No valid rows remaining after null dropping.")
        return

    raw_time = df["global_timestamp_s"].to_numpy()
    raw_time = raw_time - raw_time[0]
    fps = 1.0 / np.median(np.diff(raw_time))
    
    f0 = df["meas_force_0_N"].to_numpy()
    f1 = df["meas_force_1_N"].to_numpy()
    
    qpos_data = np.zeros((len(df), 13))
    for i, col in enumerate(joint_cols):
        rads = np.deg2rad(df[col].to_numpy())
        qpos_data[:, i] = smooth_data(rads, fps, cutoff=10.0)

    print(f"  Datensätze: {len(qpos_data)}, FPS={fps:.1f}, Dauer={raw_time[-1]:.2f}s")
    
    max_t = raw_time[-1] if args.sim_time <= 0 else min(args.sim_time, raw_time[-1])
    mask = raw_time <= max_t
    t_target = raw_time[mask]
    qpos_target = qpos_data[mask]
    f0_target = f0[mask]
    f1_target = f1[mask]

    qvel_target = np.zeros_like(qpos_target)
    if len(t_target) > 1:
        for i in range(qpos_target.shape[1]):
            raw_vel = np.gradient(qpos_target[:, i], t_target)
            cleaned_vel = remove_outliers_mad(raw_vel, thresh=3.5)
            qvel_target[:, i] = smooth_data(cleaned_vel, fps=fps, cutoff=2.0)

    global _GT_QPOS, _GT_QVEL, _SIM_TIMESTEPS, _FORCE_INTERP_0, _FORCE_INTERP_1
    _GT_QPOS = qpos_target
    _GT_QVEL = qvel_target
    _SIM_TIMESTEPS = t_target
    _FORCE_INTERP_0 = interp1d(t_target, f0_target, kind='linear', fill_value=(f0_target[0], f0_target[-1]), bounds_error=False)
    _FORCE_INTERP_1 = interp1d(t_target, f1_target, kind='linear', fill_value=(f1_target[0], f1_target[-1]), bounds_error=False)

    njnt = 13
    ntendon = 2

    # Bounds for Block 1 (Stiffness & Tendon Stiffness)
    bounds_b1 = []
    for _ in range(njnt): bounds_b1.append((BOUNDS_STIFFNESS[0]/INIT_BASE_STIFFNESS, BOUNDS_STIFFNESS[1]/INIT_BASE_STIFFNESS))
    for _ in range(ntendon): bounds_b1.append((BOUNDS_TENDON_STIFFNESS[0]/INIT_BASE_TENDON_STIFFNESS, BOUNDS_TENDON_STIFFNESS[1]/INIT_BASE_TENDON_STIFFNESS))
    
    # Bounds for Block 2 (Damping)
    bounds_b2 = []
    for _ in range(njnt): bounds_b2.append((BOUNDS_DAMPING[0]/INIT_BASE_DAMPING, BOUNDS_DAMPING[1]/INIT_BASE_DAMPING))

    curr_b1 = np.ones(njnt + ntendon)
    curr_b2 = np.ones(njnt)
    
    print("\n" + "=" * 55)
    print(f"Starte Alternierende Optimierung über {max_t:.2f}s")
    print(f"B1 (Stiff): {len(curr_b1)} Param | B2 (Damp): {len(curr_b2)} Param")
    print(f"Outer Iterations: {args.outer_maxiter} | Inner Max Iter Loop: {args.inner_maxiter}")
    print("-" * 55)

    t0 = time.time()
    best_overall_cost = float('inf')
    
    try:
        for outer in range(args.outer_maxiter):
            print(f"\n>>>> OUTER LOOP {outer + 1} / {args.outer_maxiter} <<<<")
            
            # --- 1. Optimize Stiffness (B1) ---
            print("  -> Optimiere STIFFNESS (B1) - halte Damping konstant...")
            res_b1 = differential_evolution(
                global_objective_stiff,
                args=(curr_b2, max_t, njnt, ntendon),
                bounds=bounds_b1,
                x0=curr_b1,
                maxiter=args.inner_maxiter,
                tol=args.tol,
                seed=42,
                polish=False, # Keep false inside loop to save time
                popsize=args.workers,
                workers=args.workers,
                disp=True,
            )
            curr_b1 = res_b1.x
            print(f"     [B1 Step] Cost nach Stiffness: {res_b1.fun:.6e}")
            
            # --- 2. Optimize Damping (B2) ---
            print("  -> Optimiere DAMPING (B2) - halte Stiffness konstant...")
            res_b2 = differential_evolution(
                global_objective_damp,
                args=(curr_b1, max_t, njnt, ntendon),
                bounds=bounds_b2,
                x0=curr_b2,
                maxiter=args.inner_maxiter,
                tol=args.tol,
                seed=42+outer, # differ seed slightly
                polish=True if outer == args.outer_maxiter - 1 else False, # polish on last step
                popsize=args.workers,
                workers=args.workers,
                disp=True,
            )
            curr_b2 = res_b2.x
            print(f"     [B2 Step] Cost nach Damping: {res_b2.fun:.6e}")
            
            # Check convergence
            cost = res_b2.fun
            if best_overall_cost - cost < args.outer_tol:
                print(f"\n[INFO] Konvergenz erreicht! (Kosten-Verbesserung < {args.outer_tol})")
                best_overall_cost = cost
                break
            best_overall_cost = cost

    except KeyboardInterrupt:
        print("\n\n[!] Optimierung durch Benutzer abgebrochen (KeyboardInterrupt).")

    elapsed = time.time() - t0
    
    stiff_scale = np.full(njnt, INIT_BASE_STIFFNESS)
    tstiff_scale = np.full(ntendon, INIT_BASE_TENDON_STIFFNESS)
    damp_scale = np.full(njnt, INIT_BASE_DAMPING)
    
    stiff_res = curr_b1[:njnt] * stiff_scale
    t_stiff_res = curr_b1[njnt:] * tstiff_scale
    damp_res = curr_b2 * damp_scale

    print("\n" + "=" * 55 + "\nERGEBNIS")
    print(f"  Cost    : {best_overall_cost:.10e}\n  Dauer   : {elapsed:.1f} s\n")
    
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
    
    out_dict = {
        "stiffness": stiff_res.tolist(),
        "damping": damp_res.tolist(),
        "tendon_stiffness": t_stiff_res.tolist(),
        "cost": float(best_overall_cost)
    }
    
    out_file = Path(__file__).resolve().parent / "build" / "sysid_real_alt_params.json"
    with open(out_file, "w") as f:
        json.dump(out_dict, f, indent=4)
        
    print(f"Parameter saved to: {out_file}")

if __name__ == "__main__":
    main()