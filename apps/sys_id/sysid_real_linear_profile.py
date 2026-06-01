#!/usr/bin/env python3
"""
SpiRob - Real Data System Identification with Linear Stiffness Profile

Fixed reference stiffness at the lowest joint (joint 12 / index 12) and a
single optimized linear drop toward the other joints.
The script reuses the real-data preprocessing and plots from sysid_real_alt_plot.
"""

import argparse
import atexit
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import polars as pl
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.optimize import differential_evolution

# ── Model Geometry (fixed) ──────────────────────────────────────────
L_TARGET = 0.44
BASE_D = 0.1
TIP_D = 0.03
DELTA_THETA_DEG = 30.0

# ── Linear Stiffness Profile ───────────────────────────────────────
REFERENCE_JOINT_INDEX = 12
REFERENCE_STIFFNESS = 0.42  # Nm/rad, manually measured

# The optimizer searches a normalized drop fraction in [0, 0.98].
# Actual drop per joint is fraction * MAX_LINEAR_DROP_PER_JOINT.
MAX_LINEAR_DROP_PER_JOINT = REFERENCE_STIFFNESS / REFERENCE_JOINT_INDEX

# ── Initialization Values for Optimization ──────────────────────────
INIT_BASE_DAMPING = 0.05
INIT_BASE_TENDON_STIFFNESS = 10.0

# ── Optimization Bounds ──────────────────────────────────────────────
BOUNDS_DROP_FRACTION = (0.0, 0.98)

# ── Optimization Cost Function Selector ──────────────────────────────
ACTIVE_COST_FUNCTION = "sysid_multi"

FIXED_JOINT_DAMPING = 0.05
FIXED_TENDON_STIFFNESS = 500.0

# ====================================================================
#  Helper Functions
# ====================================================================

def make_model() -> mj.MjModel:
    xml_path = Path(__file__).resolve().parent / "spiral_chain_wo_cylinder.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    return mj.MjModel.from_xml_path(str(xml_path))


def set_params(model: mj.MjModel, stiffness: np.ndarray, damping: np.ndarray, tendon_stiffness: np.ndarray) -> None:
    for i in range(model.njnt):
        model.jnt_stiffness[i] = stiffness[i]
        dof = model.jnt_dofadr[i]
        model.dof_damping[dof] = damping[i]
    for i in range(model.ntendon):
        model.tendon_stiffness[i] = tendon_stiffness[i]


def build_linear_stiffness_profile(drop_per_joint: float, njnt: int) -> np.ndarray:
    joint_ids = np.arange(njnt)
    distances_from_reference = REFERENCE_JOINT_INDEX - joint_ids
    stiffness = REFERENCE_STIFFNESS - drop_per_joint * distances_from_reference
    return stiffness


# ====================================================================
#  Multiprocessing Globals & Profiling
# ====================================================================

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


def _print_worker_stats() -> None:
    if _PROF_EVALS > 0:
        total = _PROF_TIME_SETUP + _PROF_TIME_SIM + _PROF_TIME_MSE
        print(
            f"  [Worker Profiling] Evals: {_PROF_EVALS} | Setup: {_PROF_TIME_SETUP:.2f}s | "
            f"Sim: {_PROF_TIME_SIM:.2f}s | MSE: {_PROF_TIME_MSE:.2f}s | Total: {total:.2f}s"
        )


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


def simulate_and_sample(
    model: mj.MjModel,
    data: mj.MjData,
    sim_time: float,
    gt_qpos_init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    global _PROF_TIME_SIM
    t_start = time.perf_counter()

    mj.mj_resetData(model, data)
    model.opt.timestep = 0.004
    model.opt.iterations = 20

    data.qpos[:] = gt_qpos_init
    data.qvel[:] = np.zeros(model.nv)
    mj.mj_forward(model, data)

    dt = model.opt.timestep
    total_steps = int(sim_time / dt)

    qpos_history = []
    qvel_history = []

    for _ in range(total_steps + 1):
        qpos_history.append(data.qpos.copy())
        qvel_history.append(data.qvel.copy())
        controller(model, data, data.time)
        mj.mj_step(model, data)

    _PROF_TIME_SIM += time.perf_counter() - t_start
    return np.array(qpos_history), np.array(qvel_history)


def cost_function_raw(
    stiffness: np.ndarray,
    damp: np.ndarray,
    t_stiff: np.ndarray,
    sim_time: float,
    model: mj.MjModel,
    data: mj.MjData,
) -> float:
    global _PROF_EVALS, _PROF_TIME_SETUP, _PROF_TIME_MSE, _PROF_TIME_SIM, _EVAL_COUNTER

    try:
        _EVAL_COUNTER += 1
    except NameError:
        _EVAL_COUNTER = 1

    _PROF_EVALS += 1

    t_start = time.perf_counter()
    if np.any(stiffness <= 0) or np.any(damp <= 0) or np.any(t_stiff <= 0):
        _PROF_TIME_SETUP += time.perf_counter() - t_start
        return 1e6

    try:
        set_params(model, stiffness, damp, t_stiff)
        _PROF_TIME_SETUP += time.perf_counter() - t_start

        est_qpos, est_qvel = simulate_and_sample(model, data, sim_time, _GT_QPOS[0])
        t_mse = time.perf_counter()
    except Exception:
        _PROF_TIME_SETUP += time.perf_counter() - t_start
        return 1e6

    len_gt = len(_GT_QPOS)
    len_est = len(est_qpos)

    if len_gt == 0 or len_est == 0:
        _PROF_TIME_MSE += time.perf_counter() - t_mse
        return 1e6

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

    err_pos_deg = np.rad2deg(gt_pos_match - est_pos_match)
    err_vel_deg = np.rad2deg(gt_vel_match - est_vel_match)

    if ACTIVE_COST_FUNCTION == "sysid_multi":
        w = np.linspace(1.1, 0.8, n).reshape(-1, 1)
        rmse_pos = np.sqrt(np.sum(w * err_pos_deg ** 2))
        rmse_vel = np.sqrt(np.sum(w * err_vel_deg ** 2))
        cost = rmse_pos + 0.05 * rmse_vel
    else:
        cost = np.sqrt(np.mean(err_pos_deg ** 2))

    if _EVAL_COUNTER % 100 == 0:
        import os

        intermediate_dir = Path(__file__).resolve().parent / "build" / "intermediate_plots"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        log_info = {
            "eval_counter": _EVAL_COUNTER,
            "worker_pid": os.getpid(),
            "raw_gt_shape": list(_GT_QPOS.shape) if _GT_QPOS is not None else None,
            "raw_est_shape": list(est_qpos.shape) if est_qpos is not None else None,
            "match_gt_shape": list(gt_pos_match.shape),
            "match_est_shape": list(est_pos_match.shape),
            "data_type_gt": str(gt_pos_match.dtype),
            "data_type_est": str(est_pos_match.dtype),
        }
        log_file = intermediate_dir / f"data_format_log_W{os.getpid()}_{_EVAL_COUNTER}.json"
        try:
            with open(log_file, "w") as f:
                json.dump(log_info, f, indent=4)
        except Exception as exc:
            print(f"Fehler beim Loggen des Zwischenschritts {_EVAL_COUNTER}: {exc}")

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

            axs_val[0].legend(loc="center left", bbox_to_anchor=(1.0, 1.1))
            plt.tight_layout()

            plot_path = intermediate_dir / f"interim_plot_W{os.getpid()}_{_EVAL_COUNTER:05d}.png"
            fig_val.savefig(plot_path, dpi=150)
            plt.close(fig_val)
        except Exception as exc:
            print(f"Fehler beim Zwischenplot {_EVAL_COUNTER}: {exc}")

    _PROF_TIME_MSE += time.perf_counter() - t_mse
    return cost


def global_objective_drop(
    x_norm: np.ndarray,
    sim_time: float,
    njnt: int,
    ntendon: int,
) -> float:
    model, data = get_local_model_and_data()
    drop_per_joint = float(x_norm[0]) * MAX_LINEAR_DROP_PER_JOINT
    stiffness = build_linear_stiffness_profile(drop_per_joint, njnt)
    damping = np.full(njnt, FIXED_JOINT_DAMPING)
    t_stiff = np.full(ntendon, FIXED_TENDON_STIFFNESS)
    return cost_function_raw(stiffness, damping, t_stiff, sim_time, model, data)


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
    if mad < 1e-6:
        mad = 1e-6
    z_scores = 0.6745 * np.abs(arr - median) / mad
    for idx in np.where(z_scores > thresh)[0]:
        if 0 < idx < len(arr) - 1:
            arr[idx] = (arr[idx - 1] + arr[idx + 1]) / 2.0
        elif idx == 0:
            arr[idx] = arr[1]
        else:
            arr[idx] = arr[-2]
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description="Linear-profile System ID on Real Data")
    ap.add_argument("--sim-time", type=float, default=10.0, help="Max sim time (-1 for full)")
    ap.add_argument("--inner-maxiter", type=int, default=10, help="DE maxiter per refinement step")
    ap.add_argument("--outer-maxiter", type=int, default=5, help="Number of refinement loops")
    ap.add_argument("--outer-tol", type=float, default=0.001, help="Tolerance for outer loop stop")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=10, help="Parallel processes count")
    ap.add_argument("--force-window", type=int, default=15, help="Window size for moving average filter (frames)")
    ap.add_argument("--data-smooth-window", type=int, default=5, help="Window size for smoothing joint data (frames)")
    args = ap.parse_args()

    data_path = Path(__file__).resolve().parent / "build" / "sys_id_auto_GX010070.parquet"
    if not data_path.exists():
        data_path = Path(__file__).resolve().parent / "build" / "sys_id_auto.parquet"
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
        smooth_plot_path = Path(__file__).resolve().parent / "build" / "sysid_real_linear_profile_smoothed_data.png"
        fig_smooth.savefig(smooth_plot_path, dpi=150)
        plt.close(fig_smooth)
        print(f"  [Info] Smoothed Daten Plot gespeichert unter: {smooth_plot_path}")
    except Exception as exc:
        print(f"  [Warn] Plot für smoothed Daten fehlgeschlagen: {exc}")

    global _GT_QPOS, _GT_QVEL, _SIM_TIMESTEPS, _FORCE_INTERP_0, _FORCE_INTERP_1
    _GT_QPOS = qpos_target
    _GT_QVEL = qvel_target
    _SIM_TIMESTEPS = t_target
    _FORCE_INTERP_0 = interp1d(t_target, f0_target, kind="linear", fill_value=(f0_target[0], f0_target[-1]), bounds_error=False)
    _FORCE_INTERP_1 = interp1d(t_target, f1_target, kind="linear", fill_value=(f1_target[0], f1_target[-1]), bounds_error=False)

    njnt = 13
    ntendon = 2

    x0 = np.array([0.35])
    norm_bounds = [BOUNDS_DROP_FRACTION]

    print("\n" + "=" * 55)
    print(f"Starte Optimierung über {max_t:.2f}s")
    print(f"Stiffness-Profil: joint 12 fix auf {REFERENCE_STIFFNESS:.2f} Nm/rad")
    print("Optimiert wird nur die lineare Abnahme pro Gelenk")
    print(f"Max Iter Loop: {args.inner_maxiter}")
    print("-" * 55)

    history_x = []
    history_cost = []

    def callback_refine(xk: np.ndarray, convergence: float | None = None) -> None:
        history_x.append(xk.copy())

    t0 = time.time()
    best_x = x0.copy()
    best_cost = float("inf")
    status_msg = "not started"

    try:
        for outer_idx in range(args.outer_maxiter):
            print(f"  -> Refinement {outer_idx + 1}/{args.outer_maxiter}")
            result = differential_evolution(
                global_objective_drop,
                args=(max_t, njnt, ntendon),
                bounds=norm_bounds,
                x0=best_x,
                maxiter=args.inner_maxiter,
                tol=args.tol,
                seed=42,
                polish=True,
                init="sobol",
                popsize=max(5, args.workers),
                workers=args.workers,
                disp=True,
                callback=callback_refine,
            )

            current_cost = float(result.fun)
            history_cost.append(current_cost)
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = result.x.copy()

            status_msg = str(result.message)
            if len(history_cost) > 1 and abs(history_cost[-2] - history_cost[-1]) < args.outer_tol:
                print("  [Info] Outer-loop stop: improvement below outer-tol")
                break
    except KeyboardInterrupt:
        print("\n\n[!] Optimierung durch Benutzer abgebrochen (KeyboardInterrupt).")
        best_cost = global_objective_drop(best_x, max_t, njnt, ntendon)
        status_msg = "User aborted (KeyboardInterrupt)"

    elapsed = time.time() - t0

    drop_per_joint = float(best_x[0]) * MAX_LINEAR_DROP_PER_JOINT
    stiff_res = build_linear_stiffness_profile(drop_per_joint, njnt)
    damp_res = np.full(njnt, FIXED_JOINT_DAMPING)
    t_stiff_res = np.full(ntendon, FIXED_TENDON_STIFFNESS)

    print("\n" + "=" * 55 + "\nERGEBNIS")
    print(f"  Status  : {status_msg}\n  Cost    : {best_cost:.10e}\n  Dauer   : {elapsed:.1f} s\n")
    print(f"  Optimierte lineare Abnahme pro Gelenk: {drop_per_joint:.6f} Nm/rad")
    print(f"  Joint 12 Referenzstiffness: {REFERENCE_STIFFNESS:.3f} Nm/rad")

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
        "drop_per_joint": drop_per_joint,
        "reference_joint_index": REFERENCE_JOINT_INDEX,
        "reference_stiffness": REFERENCE_STIFFNESS,
        "cost": float(best_cost),
    }
    out_file = Path(__file__).resolve().parent / "build" / "sysid_real_linear_profile_params.json"
    with open(out_file, "w") as f:
        json.dump(out_dict, f, indent=4)
    print(f"Parameter saved to: {out_file}")

    try:
        if len(history_x) > 0:
            hist_x = np.array(history_x).reshape(-1)
            profile_history = [build_linear_stiffness_profile(float(x) * MAX_LINEAR_DROP_PER_JOINT, njnt) for x in hist_x]
            profile_history = np.array(profile_history)

            fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=False)
            iterations = np.arange(len(hist_x))

            axs[0].plot(iterations, hist_x * MAX_LINEAR_DROP_PER_JOINT, marker="o")
            axs[0].set_title("Verlauf: Lineare Abnahme pro Gelenk")
            axs[0].set_ylabel("Drop pro Gelenk (Nm/rad)")
            axs[0].grid(True, linestyle="--", alpha=0.5)

            for i in range(min(len(profile_history), 10)):
                axs[1].plot(np.arange(njnt), profile_history[i], alpha=0.4)
            axs[1].plot(np.arange(njnt), stiff_res, color="black", linewidth=2.5, label="Final")
            axs[1].set_title("Evolution des linearen Stiffness-Profils")
            axs[1].set_xlabel("Joint index")
            axs[1].set_ylabel("Stiffness (Nm/rad)")
            axs[1].grid(True, linestyle="--", alpha=0.5)
            axs[1].legend()

            plt.tight_layout()
            plot_path = Path(__file__).resolve().parent / "build" / "sysid_real_linear_profile_convergence.png"
            fig.savefig(plot_path, dpi=150)
            print(f"\n  [Info] Konvergenz-Plot gespeichert unter: {plot_path}")
            plt.close(fig)

            profile_dir = Path(__file__).resolve().parent / "build" / "sysid_real_linear_profile_plots"
            profile_dir.mkdir(parents=True, exist_ok=True)
            fig2 = plt.figure(figsize=(8, 6))
            ax2 = fig2.add_subplot(111)
            colors = plt.cm.viridis(np.linspace(0, 1, len(profile_history)))
            for i, prof in enumerate(profile_history):
                ax2.plot(np.arange(njnt), prof, color=colors[i], alpha=0.7)
            ax2.plot(np.arange(njnt), stiff_res, color="red", linewidth=2.5, label="Final")
            ax2.axvline(REFERENCE_JOINT_INDEX, color="gray", linestyle="--", alpha=0.6)
            ax2.set_title("Lineare Stiffness-Profile über die Optimierung")
            ax2.set_xlabel("Joint index")
            ax2.set_ylabel("Stiffness (Nm/rad)")
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend()
            fig2.tight_layout()
            profile_plot_path = profile_dir / "stiffness_profile_evolution.png"
            fig2.savefig(profile_plot_path, dpi=150)
            plt.close(fig2)
            print(f"  [Info] Profil-Plots gespeichert im Ordner:\n         {profile_dir}")

        plt.close("all")
    except Exception as exc:
        print(f"\n  [Warn] Plot konnte nicht erstellt werden: {exc}")

    try:
        print("\n  [Info] Starte finale Validierungssimulation...")
        val_model, val_data = get_local_model_and_data()
        set_params(val_model, stiff_res, damp_res, t_stiff_res)
        est_qpos, _ = simulate_and_sample(val_model, val_data, max_t, _GT_QPOS[0])

        fig_val, axs_val = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
        colors_val = plt.cm.tab20(np.linspace(0, 1, njnt))

        for i in range(njnt):
            axs_val[0].plot(_GT_QPOS[:, i], label=f"J{i}", color=colors_val[i])
        axs_val[0].set_title("Ground Truth Trajektorien (Realität)")
        axs_val[0].set_ylabel("Winkel (rad)")
        axs_val[0].grid(True, linestyle="--", alpha=0.5)

        for i in range(njnt):
            axs_val[1].plot(est_qpos[:, i], label=f"J{i}", color=colors_val[i])
        axs_val[1].set_title("Simulierte Trajektorien (Linearer Stiffness-Ansatz)")
        axs_val[1].set_xlabel("Samples")
        axs_val[1].set_ylabel("Winkel (rad)")
        axs_val[1].grid(True, linestyle="--", alpha=0.5)
        axs_val[1].legend(loc="center left", bbox_to_anchor=(1.0, 1.1))

        plt.tight_layout()
        val_plot_path = Path(__file__).resolve().parent / "build" / "sysid_real_linear_profile_validation.png"
        fig_val.savefig(val_plot_path, dpi=150)
        print(f"  [Info] Validierungsplot gespeichert unter: {val_plot_path}")
        plt.close(fig_val)
    except Exception as exc:
        print(f"\n  [Warn] Validierungsplot konnte nicht erstellt werden: {exc}")


if __name__ == "__main__":
    main()