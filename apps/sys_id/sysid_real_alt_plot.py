#!/usr/bin/env python3
"""
SpiRob – Real Data System Identification (Alternating Parameter Blocks) + Convergence Plot
Optimizes Stiffness/Tendon-Stiffness first, then Damping, iteratively.
Tracks the progression of the best parameter values and saves a plot.
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
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

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
BOUNDS_STIFFNESS = (0.001, 100)   #joint 12 hat 0.41nm/rad stiffness, also 1.0 is großzügig
BOUNDS_DAMPING = (0.01, 100.0)
BOUNDS_TENDON_STIFFNESS = (1, 1000.0)

# ── Optimization Cost Function Selector ──────────────────────────────
ACTIVE_COST_FUNCTION = "sysid_multi"

FIXED_TENDON_STIFFNESS = 500.0
FIXED_JOINT_DAMPING = 1.0

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
    model.opt.timestep = 0.004
    model.opt.iterations = 20
    
    data.qpos[:] = gt_qpos_init
    data.qvel[:] = np.zeros(model.nv)
    mj.mj_forward(model, data)

    # Kurze Einschwingphase: lässt das System unter den anliegenden Kräften zur Ruhe kommen
    for _ in range(1000):
        mj.mj_step(model, data)
        data.qvel[:] = 0.0

    # Trajektorie ab hier wieder bei t=0 starten lassen
    data.time = 0.0
    
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

def cost_function_raw(stiff: np.ndarray,
                      damp: np.ndarray,
                      t_stiff: np.ndarray,
                      sim_time: float,
                      model: mj.MjModel,
                      data: mj.MjData) -> float:
    global _PROF_EVALS, _PROF_TIME_SETUP, _PROF_TIME_MSE, _PROF_TIME_SIM, _EVAL_COUNTER
    try:
        _EVAL_COUNTER += 1
    except NameError:
        _EVAL_COUNTER = 1
    
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

    # LÖSUNG: 
    # 1. Umrechnung des Fehlers in Grad (vergrößert den Wert um Faktor ~57)
    err_pos_deg = np.rad2deg(gt_pos_match - est_pos_match)
    err_vel_deg = np.rad2deg(gt_vel_match - est_vel_match)
        
    if ACTIVE_COST_FUNCTION == "sysid_multi":
        w = np.linspace(1.1, 0.8, n).reshape(-1, 1)
        # 2. RMSE (Wurzel aus MSE) statt reinem MSE. 
        # Das verhindert, dass Werte < 1 beim Quadrieren verschwinden.
        rmse_pos = np.sqrt(np.sum(w * err_pos_deg ** 2))
        rmse_vel = np.sqrt(np.sum(w * err_vel_deg ** 2))
        
        # Man kann die Gewichtung der Geschwindigkeit anpassen, falls sie dominiert
        cost = rmse_pos + 0.05 * rmse_vel 
    else:
        # Auch hier RMSE
        cost = np.sqrt(np.mean(err_pos_deg ** 2))
        
    if _EVAL_COUNTER % 100 == 0:
        import os
        import json
        from pathlib import Path
        
        intermediate_dir = Path(__file__).resolve().parent / "build" / "intermediate_plots"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Loggen der Formate
        log_info = {
            "eval_counter": _EVAL_COUNTER,
            "worker_pid": os.getpid(),
            "raw_gt_shape": list(_GT_QPOS.shape) if _GT_QPOS is not None else None,
            "raw_est_shape": list(est_qpos.shape) if est_qpos is not None else None,
            "match_gt_shape": list(gt_pos_match.shape),
            "match_est_shape": list(est_pos_match.shape),
            "data_type_gt": str(gt_pos_match.dtype),
            "data_type_est": str(est_pos_match.dtype)
        }
        log_file = intermediate_dir / f"data_format_log_W{os.getpid()}_{_EVAL_COUNTER}.json"
        try:
            with open(log_file, "w") as f:
                json.dump(log_info, f, indent=4)
        except Exception as e:
            print(f"Fehler beim Loggen des Zwischenschritts {_EVAL_COUNTER}: {e}")
        
        # 2. Plotten
        try:
            fig_val, axs_val = plt.subplots(3, 1, figsize=(12, 15), sharey=True)
            colors = plt.cm.tab20(np.linspace(0, 1, gt_pos_match.shape[1]))
            
            for i in range(gt_pos_match.shape[1]):
                axs_val[0].plot(gt_pos_match[:, i], label=f"J{i}", color=colors[i])
            axs_val[0].set_title(f"Interim {_EVAL_COUNTER} (W{os.getpid()}): Ground Truth (Matched)")
            axs_val[0].set_ylabel("Winkel (rad)")
            axs_val[0].grid(True, linestyle="--", alpha=0.5)
            
            for i in range(est_pos_match.shape[1]):
                axs_val[1].plot(est_pos_match[:, i], label=f"J{i}", color=colors[i])
            axs_val[1].set_title(f"Interim {_EVAL_COUNTER} (W{os.getpid()}): Simuliert (Matched)")
            axs_val[1].set_ylabel("Winkel (rad)")
            axs_val[1].grid(True, linestyle="--", alpha=0.5)
            
            for i in range(est_qpos.shape[1]):
                axs_val[2].plot(est_qpos[:, i], label=f"J{i}", color=colors[i])
            axs_val[2].set_title(f"Interim {_EVAL_COUNTER} (W{os.getpid()}): Simuliert (RAW vor Matching)")
            axs_val[2].set_xlabel("Samples")
            axs_val[2].set_ylabel("Winkel (rad)")
            axs_val[2].grid(True, linestyle="--", alpha=0.5)
            
            axs_val[0].legend(loc='center left', bbox_to_anchor=(1.0, 1.1))
            plt.tight_layout()
            
            plot_path = intermediate_dir / f"interim_plot_W{os.getpid()}_{_EVAL_COUNTER:05d}.png"
            fig_val.savefig(plot_path, dpi=150)
            plt.close(fig_val)
        except Exception as e:
            print(f"Fehler beim Zwischenplot {_EVAL_COUNTER}: {e}")

    _PROF_TIME_MSE += (time.perf_counter() - t_mse)
    return cost

def global_objective_stiff(x_b1: np.ndarray, sim_time: float, njnt: int, ntendon: int) -> float:
    model, data = get_local_model_and_data()
    stiff_scale = np.full(njnt, INIT_BASE_STIFFNESS)
    
    stiff = x_b1 * stiff_scale
    t_stiff = np.full(ntendon, FIXED_TENDON_STIFFNESS)
    damp = np.full(njnt, FIXED_JOINT_DAMPING)
    return cost_function_raw(stiff, damp, t_stiff, sim_time, model, data)

def global_objective_damp(x_b2: np.ndarray, curr_b1_norm: np.ndarray, sim_time: float, njnt: int, ntendon: int) -> float:
    model, data = get_local_model_and_data()
    stiff_scale = np.full(njnt, INIT_BASE_STIFFNESS)
    damp_scale = np.full(njnt, INIT_BASE_DAMPING)
    
    stiff = curr_b1_norm * stiff_scale
    t_stiff = np.full(ntendon, FIXED_TENDON_STIFFNESS)
    damp = x_b2 * damp_scale
    return cost_function_raw(stiff, damp, t_stiff, sim_time, model, data)

# ====================================================================
#  Data Preparation
# ====================================================================

def smooth_data(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    if len(data) < window_size or window_size < 2:
        return data
    return uniform_filter1d(data, size=window_size)

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
    ap = argparse.ArgumentParser(description="Alternating System ID on Real Data with Plot")
    ap.add_argument("--sim-time", type=float, default=10.0, help="Max sim time (-1 for full)")
    ap.add_argument("--inner-maxiter", type=int, default=10, help="DE maxiter per alternating step") 
    ap.add_argument("--outer-maxiter", type=int, default=5, help="Number of alternating loops")
    ap.add_argument("--outer-tol", type=float, default=0.001, help="Tolerance for outer loop stop")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=10, help="Parallel processes count")
    ap.add_argument("--force-window", type=int, default=15, help="Window size for moving average filter (frames)")
    ap.add_argument("--data-smooth-window", type=int, default=5, help="Window size for smoothing joint data (frames)")
    args = ap.parse_args()

    data_path = Path(__file__).resolve().parent / "build" / "sys_id_auto_GX010070.parquet"
    if not data_path.exists():
        data_path = Path(__file__).resolve().parent / "build" / "sys_id_auto.parquet" # Fallback if specific file doesn't exist
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
    
    f0_raw = df["meas_force_0_N"].to_numpy()
    f1_raw = df["meas_force_1_N"].to_numpy()
    
    # Glätten der gemessenen Seilkräfte mit Mittelwertfilter
    f0 = smooth_data(f0_raw, window_size=args.force_window)
    f1 = smooth_data(f1_raw, window_size=args.force_window)
    
    qpos_data = np.zeros((len(df), 13))
    for i, col in enumerate(joint_cols):
        rads = np.deg2rad(df[col].to_numpy())
        qpos_data[:, i] = smooth_data(rads, window_size=args.data_smooth_window)

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
            qvel_target[:, i] = smooth_data(cleaned_vel, window_size=args.data_smooth_window)

    # Plot smoothed input data (positions and velocities)
    try:
        fig_smooth, axs_smooth = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        colors_smooth = plt.cm.tab20(np.linspace(0, 1, qpos_target.shape[1]))
        for i in range(qpos_target.shape[1]):
            axs_smooth[0].plot(t_target, qpos_target[:, i], color=colors_smooth[i])
        axs_smooth[0].set_title("Smoothed Position Data (qpos_target)")
        axs_smooth[0].set_ylabel("Position (rad)")
        axs_smooth[0].grid(True, linestyle="--", alpha=0.5)
        
        for i in range(qvel_target.shape[1]):
            axs_smooth[1].plot(t_target, qvel_target[:, i], color=colors_smooth[i])
        axs_smooth[1].set_title("Smoothed Velocity Data (qvel_target)")
        axs_smooth[1].set_ylabel("Velocity (rad/s)")
        axs_smooth[1].set_xlabel("Time (s)")
        axs_smooth[1].grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        smooth_plot_path = Path(__file__).resolve().parent / "build" / "sysid_real_alt_smoothed_data.png"
        fig_smooth.savefig(smooth_plot_path, dpi=150)
        plt.close(fig_smooth)
        print(f"  [Info] Smoothed Daten Plot gespeichert unter: {smooth_plot_path}")
    except Exception as e:
        print(f"  [Warn] Plot für smoothed Daten fehlgeschlagen: {e}")

    global _GT_QPOS, _GT_QVEL, _SIM_TIMESTEPS, _FORCE_INTERP_0, _FORCE_INTERP_1
    _GT_QPOS = qpos_target
    _GT_QVEL = qvel_target
    _SIM_TIMESTEPS = t_target
    _FORCE_INTERP_0 = interp1d(t_target, f0_target, kind='linear', fill_value=(f0_target[0], f0_target[-1]), bounds_error=False)
    _FORCE_INTERP_1 = interp1d(t_target, f1_target, kind='linear', fill_value=(f1_target[0], f1_target[-1]), bounds_error=False)

    njnt = 13
    ntendon = 2

    # Bounds for Block 1 (Stiffness only)
    bounds_b1 = []
    for _ in range(njnt): bounds_b1.append((BOUNDS_STIFFNESS[0]/INIT_BASE_STIFFNESS, BOUNDS_STIFFNESS[1]/INIT_BASE_STIFFNESS))
    
    curr_b1 = np.ones(njnt)
    # Damping is now fixed, dummy track to not break existing plots
    curr_b2 = np.full(njnt, FIXED_JOINT_DAMPING / INIT_BASE_DAMPING)
    
    print("\n" + "=" * 55)
    print(f"Starte Optimierung über {max_t:.2f}s")
    print(f"Nur STIFFNESS wird optimiert (Damping ist FIX auf {FIXED_JOINT_DAMPING})")
    print(f"Max Iter Loop: {args.inner_maxiter}")
    print("-" * 55)

    
    # ── Convergence History for Plotting ──
    history_b1 = []
    history_b2 = []
    cost_history = []

    def callback_b1(xk, convergence=None):
        history_b1.append(xk.copy())
        history_b2.append(curr_b2.copy())
        cost_history.append(global_objective_stiff(xk, max_t, njnt, ntendon))

    t0 = time.time()
    best_overall_cost = float('inf')
    
    try:
        print("  -> Optimiere STIFFNESS - Damping konstant...")
        res_b1 = differential_evolution(
            global_objective_stiff,
            args=(max_t, njnt, ntendon),
            bounds=bounds_b1,
            x0=curr_b1,
            maxiter=args.inner_maxiter,
            tol=args.tol,
            seed=42,
            polish=True,
            popsize=args.workers,
            workers=args.workers,
            disp=True,
            callback=callback_b1
        )
        curr_b1 = res_b1.x
        best_overall_cost = res_b1.fun
        print(f"     [Result] Final Cost: {best_overall_cost:.6e}")

    except KeyboardInterrupt:
        print("\n\n[!] Optimierung durch Benutzer abgebrochen (KeyboardInterrupt).")

    elapsed = time.time() - t0
    
    stiff_scale = np.full(njnt, INIT_BASE_STIFFNESS)
    damp_scale = np.full(njnt, INIT_BASE_DAMPING)
    
    stiff_res = curr_b1 * stiff_scale
    t_stiff_res = np.full(ntendon, FIXED_TENDON_STIFFNESS)
    damp_res = curr_b2 * damp_scale

    # ── Plotting Convergence ──
    try:
        if len(cost_history) > 0:
            fig_cost, ax_cost = plt.subplots(figsize=(10, 6))
            iterations_cost = np.arange(len(cost_history))
            ax_cost.semilogy(iterations_cost, cost_history, marker="o", linewidth=2, color="steelblue")
            ax_cost.set_title("Kosten-Konvergenz über die Iterationen")
            ax_cost.set_xlabel("Iteration")
            ax_cost.set_ylabel("Kosten (log-Skala)")
            ax_cost.grid(True, which="both", linestyle="--", alpha=0.5)
            plt.tight_layout()

            cost_plot_path = Path(__file__).resolve().parent / "build" / "sysid_real_alt_cost_convergence.png"
            fig_cost.savefig(cost_plot_path, dpi=150)
            plt.close(fig_cost)
            print(f"\n  [Info] Kosten-Konvergenzplot gespeichert unter: {cost_plot_path}")

        if len(history_b1) > 0:
            hist_b1 = np.array(history_b1)
            hist_b2 = np.array(history_b2)
            
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            iterations = np.arange(len(hist_b1))
            
            for i in range(njnt):
                axs[0].plot(iterations, hist_b1[:, i] * INIT_BASE_STIFFNESS, alpha=0.7, label=f"J{i}")
            axs[0].set_title("Verlauf: Joint Stiffness")
            axs[0].set_ylabel("Stiffness")
            axs[0].grid(True, linestyle="--", alpha=0.5)
            
            for i in range(njnt):
                axs[1].plot(iterations, hist_b2[:, i] * INIT_BASE_DAMPING, alpha=0.7, label=f"J{i}")
            axs[1].set_title("Verlauf: Joint Damping")
            axs[1].set_ylabel("Damping")
            axs[1].grid(True, linestyle="--", alpha=0.5)
            
            # Tendon plot has been removed since it's constant
            axs[2].set_visible(False)
            
            plt.tight_layout()
            plot_path = Path(__file__).resolve().parent / "build" / "sysid_real_alt_convergence.png"
            fig.savefig(plot_path, dpi=150)
            print(f"\n  [Info] Konvergenz-Plot gespeichert unter: {plot_path}")
            
            # --- 2D Parameter Space Plot per Joint ---
            param_plots_dir = Path(__file__).resolve().parent / "build" / "sysid_real_alt_param_space_plots"
            param_plots_dir.mkdir(parents=True, exist_ok=True)
            
            stiff_vals = hist_b1[:, :njnt] * INIT_BASE_STIFFNESS
            damp_vals = hist_b2[:, :njnt] * INIT_BASE_DAMPING
            
            cmap = plt.get_cmap("tab20")
            for i in range(njnt):
                fig2 = plt.figure(figsize=(8, 6))
                ax2 = fig2.add_subplot(111)
                color = cmap(i % 20)
                
                # Line showing the evolution
                ax2.plot(stiff_vals[:, i], damp_vals[:, i], color=color, alpha=0.6, linewidth=1.5, label=f"Verlauf Joint {i}")
                # Scatter for dots
                ax2.scatter(stiff_vals[:, i], damp_vals[:, i], color=color, s=15, alpha=0.8)
                # Mark Start (circle) and End (star)
                ax2.scatter(stiff_vals[0, i], damp_vals[0, i], color="blue", marker='o', s=80, edgecolor='black', zorder=5, label="Start")
                ax2.scatter(stiff_vals[-1, i], damp_vals[-1, i], color="red", marker='*', s=200, edgecolor='black', zorder=5, label="Ende")
                
                ax2.set_title(f"Parameterraum Joint {i}: Stiffness vs. Damping")
                ax2.set_xlabel("Joint Stiffness")
                ax2.set_ylabel("Joint Damping")
                ax2.grid(True, linestyle="--", alpha=0.5)
                ax2.legend()
                
                fig2.tight_layout()
                space_plot_path = param_plots_dir / f"joint_{i:02d}_space.png"
                fig2.savefig(space_plot_path, dpi=150)
                plt.close(fig2)
                
            print(f"  [Info] Parameterraum-Plots je Joint gespeichert im Ordner:\n         {param_plots_dir}")
            
        # Optional: memory cleanup for pyplot
        plt.close('all')
            
    except Exception as e:
        print(f"\n  [Warn] Plot konnte nicht erstellt werden: {e}")

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

    # ── Final Validation Simulation & Plot ──
    try:
        print("\n  [Info] Starte finale Validierungssimulation...")
        val_model, val_data = get_local_model_and_data()
        set_params(val_model, stiff_res, damp_res, t_stiff_res)
        est_qpos, _ = simulate_and_sample(val_model, val_data, max_t, _GT_QPOS[0])
        
        fig_val, axs_val = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
        colors_val = plt.cm.tab20(np.linspace(0, 1, njnt))
        
        # Subplot 1: Ground Truth
        for i in range(njnt):
            axs_val[0].plot(_GT_QPOS[:, i], label=f"J{i}", color=colors_val[i])
        axs_val[0].set_title("Ground Truth Trajektorien (Realität)")
        axs_val[0].set_ylabel("Winkel (rad)")
        axs_val[0].grid(True, linestyle="--", alpha=0.5)
        
        # Subplot 2: Simulation
        for i in range(njnt):
            axs_val[1].plot(est_qpos[:, i], label=f"J{i}", color=colors_val[i])
        axs_val[1].set_title("Simulierte Trajektorien (Identifizierte Parameter)")
        axs_val[1].set_xlabel("Samples")
        axs_val[1].set_ylabel("Winkel (rad)")
        axs_val[1].grid(True, linestyle="--", alpha=0.5)
        
        # Legende außerhalb platzieren
        axs_val[1].legend(loc='center left', bbox_to_anchor=(1.0, 1.1))
        
        plt.tight_layout()
        val_plot_path = Path(__file__).resolve().parent / "build" / "sysid_real_alt_validation.png"
        fig_val.savefig(val_plot_path, dpi=150)
        print(f"  [Info] Validierungsplot gespeichert unter: {val_plot_path}")
        plt.close(fig_val)
    except Exception as e:
        print(f"\n  [Warn] Validierungsplot konnte nicht erstellt werden: {e}")

if __name__ == "__main__":
    main()