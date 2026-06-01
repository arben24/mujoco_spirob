#!/usr/bin/env python3
"""
Refactored Real-Data System ID (modular)

This module separates: data loading/preprocessing, simulation+cost,
and plotting into independent functions so plotting can be extended
without touching core functionality.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import mujoco as mj
import mujoco.viewer as mj_viewer
import numpy as np
import polars as pl
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from scipy.ndimage import uniform_filter1d

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Constants (kept small/defaults for quick tests)
INIT_BASE_STIFFNESS = 0.05
INIT_BASE_DAMPING = 0.05
INIT_BASE_TENDON_STIFFNESS = 10.0
BOUNDS_STIFFNESS = (0.001, 100)
BOUNDS_DAMPING = (0.001, 100)

# Globals filled during data prep
_GT_QPOS: Optional[np.ndarray] = None
_GT_QVEL: Optional[np.ndarray] = None
_SIM_TIMESTEPS: Optional[np.ndarray] = None
_FORCE_INTERP_0 = None
_FORCE_INTERP_1 = None
_RECORD_DT: float = 0.0
# Actual simulation duration (may be shorter than requested sim_time if real data is shorter)
_ACTUAL_SIM_TIME: float = 0.0
# Real system joint indexing can be opposite to MuJoCo model indexing.
# If True: reverse real joint columns once during preprocessing to match simulation order.
_REVERSE_REAL_JOINT_ORDER: bool = True
_VIEWER_ENABLED: bool = False
_VIEWER_INTERVAL: int = 100
# first-call print flags
_FIRST_SIM_PRINTED = False
_FIRST_COST_PRINTED = False
_FIRST_COST_MATCH_PRINTED = False
_EVAL_COUNTER = 0


def _maybe_reverse_joint_order(arr_2d: np.ndarray, reverse_order: bool) -> np.ndarray:
    """Reverse joint axis (columns) if requested.

    Centralized place for real<->simulation joint index mapping.
    """
    if not reverse_order:
        return arr_2d
    return arr_2d[:, ::-1]


def _save_signal_comparison_plot(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    y_filtered: np.ndarray,
    title: str,
    y_label: str,
    out_path: Path,
    legend_label: str,
) -> None:
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_raw, y_raw, color="lightgray", linewidth=1.0, alpha=0.8, label="raw")
    ax.plot(x_raw[: len(y_filtered)], y_filtered, color="steelblue", linewidth=1.6, label="filtered")
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def detect_and_fill_outliers(data_1d: np.ndarray, thresh: float = 3.5) -> tuple[np.ndarray, np.ndarray]:
    """Detect outliers using MAD-based z-score and fill them by interpolation.

    Returns (filled_array, mask_outliers) where mask_outliers is boolean array True for outliers.
    """
    arr = data_1d.copy()
    if arr.size == 0:
        return arr, np.zeros_like(arr, dtype=bool)

    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    if mad < 1e-12:
        # If MAD is zero (flat signal), treat no outliers
        return arr, np.zeros_like(arr, dtype=bool)

    z = 0.6745 * np.abs(arr - median) / mad
    mask = z > thresh

    if not np.any(mask):
        return arr, mask

    # Indices of valid (non-outlier) points
    idx = np.arange(arr.size)
    valid_idx = idx[~mask]
    valid_vals = arr[~mask]

    if valid_idx.size == 0:
        # all outliers: fallback to median
        filled = np.full_like(arr, median)
        return filled, mask

    if valid_idx.size == 1:
        # single valid point: fill with that value
        filled = np.full_like(arr, valid_vals[0])
        return filled, mask

    # interpolate across valid points; np.interp handles multiple consecutive outliers
    filled = arr.copy()
    filled[mask] = np.interp(idx[mask], valid_idx, valid_vals)
    return filled, mask


def detect_and_fill_outliers_simple(data_1d: np.ndarray, abs_thresh: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Simple outlier detection: mark a sample as outlier if it deviates from
    the average of its immediate neighbors by more than `abs_thresh`.

    This is intended for isolated spikes; multiple consecutive outliers are
    filled by interpolation across valid points (same strategy as MAD-based).
    Returns (filled_array, mask_outliers).
    """
    arr = data_1d.copy()
    n = arr.size
    if n == 0:
        return arr, np.zeros_like(arr, dtype=bool)

    mask = np.zeros(n, dtype=bool)
    if n == 1:
        return arr, mask

    # interior points: compare to mean of neighbors
    for i in range(1, n - 1):
        neighbor_mean = 0.5 * (arr[i - 1] + arr[i + 1])
        if abs(arr[i] - neighbor_mean) > abs_thresh:
            mask[i] = True

    # endpoints: compare to the single neighbor
    if abs(arr[0] - arr[1]) > abs_thresh:
        mask[0] = True
    if abs(arr[-1] - arr[-2]) > abs_thresh:
        mask[-1] = True

    if not np.any(mask):
        return arr, mask

    idx = np.arange(n)
    valid_idx = idx[~mask]
    valid_vals = arr[~mask]

    if valid_idx.size == 0:
        filled = np.full_like(arr, np.median(arr))
        return filled, mask
    if valid_idx.size == 1:
        filled = np.full_like(arr, valid_vals[0])
        return filled, mask

    filled = arr.copy()
    filled[mask] = np.interp(idx[mask], valid_idx, valid_vals)
    return filled, mask


def downsample_uniform(
    data_array: np.ndarray,
    time_array: np.ndarray,
    record_dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample data uniformly by recording every N-th sample based on record_dt.

    This ensures data points are removed evenly across the entire dataset
    (not just truncated at the end). Uses the actual sampling rate of
    the data to compute record_every.

    Args:
        data_array: shape (N, ...) — array to downsample.
        time_array: shape (N,) — timestamps corresponding to data.
        record_dt: target recording interval (e.g. 0.1 s).

    Returns:
        (downsampled_data, downsampled_time) with shape (M, ...) where M << N.
    """
    if len(time_array) < 2 or record_dt <= 0:
        return data_array, time_array

    # Compute mean sampling period from actual time differences
    dt_actual = np.mean(np.diff(time_array))
    if dt_actual <= 0:
        return data_array, time_array

    # How many samples to skip between recordings
    record_every = max(1, int(round(record_dt / dt_actual)))

    # Uniformly select indices: 0, record_every, 2*record_every, ...
    indices = np.arange(0, len(time_array), record_every)

    downsampled_data = data_array[indices]
    downsampled_time = time_array[indices]

    return downsampled_data, downsampled_time


def save_preprocessing_plots(
    raw_time: np.ndarray,
    raw_force_0: np.ndarray,
    raw_force_1: np.ndarray,
    filtered_force_0: np.ndarray,
    filtered_force_1: np.ndarray,
    raw_qpos: np.ndarray,
    filtered_qpos: np.ndarray,
    raw_qvel: np.ndarray,
    filtered_qvel: np.ndarray,
    out_dir: Path,
) -> None:
    if plt is None:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    axs[0].plot(raw_time, raw_force_0, color="lightgray", linewidth=1.0, alpha=0.8, label="force_0 raw")
    axs[0].plot(raw_time[: len(filtered_force_0)], filtered_force_0, color="steelblue", linewidth=1.6, label="force_0 filtered")
    axs[0].plot(raw_time, raw_force_1, color="silver", linewidth=1.0, alpha=0.7, label="force_1 raw")
    axs[0].plot(raw_time[: len(filtered_force_1)], filtered_force_1, color="darkorange", linewidth=1.6, label="force_1 filtered")
    axs[0].set_title("Forces before/after filtering")
    axs[0].set_ylabel("force (N)")
    axs[0].grid(True, linestyle="--", alpha=0.4)
    axs[0].legend(loc="best")

    for i in range(raw_qpos.shape[1]):
        axs[1].plot(raw_time, raw_qpos[:, i], color="lightgray", linewidth=0.9, alpha=0.5)
        axs[1].plot(raw_time[: len(filtered_qpos)], filtered_qpos[:, i], linewidth=1.3)
    axs[1].set_title("Joint positions before/after filtering")
    axs[1].set_ylabel("qpos (rad)")
    axs[1].grid(True, linestyle="--", alpha=0.4)

    for i in range(raw_qvel.shape[1]):
        axs[2].plot(raw_time, raw_qvel[:, i], color="lightgray", linewidth=0.9, alpha=0.5)
        axs[2].plot(raw_time[: len(filtered_qvel)], filtered_qvel[:, i], linewidth=1.3)
    axs[2].set_title("Joint velocities before/after filtering")
    axs[2].set_xlabel("time (s)")
    axs[2].set_ylabel("qvel (rad/s)")
    axs[2].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / "preprocessing_overview.png", dpi=150)
    plt.close(fig)

    _save_signal_comparison_plot(
        raw_time,
        raw_force_0,
        filtered_force_0,
        "Force 0 before/after filtering",
        "force (N)",
        out_dir / "force_0_comparison.png",
        "force 0",
    )
    _save_signal_comparison_plot(
        raw_time,
        raw_force_1,
        filtered_force_1,
        "Force 1 before/after filtering",
        "force (N)",
        out_dir / "force_1_comparison.png",
        "force 1",
    )
    _save_signal_comparison_plot(
        raw_time,
        raw_qpos[:, 0],
        filtered_qpos[:, 0],
        "qpos[0] before/after filtering",
        "qpos (rad)",
        out_dir / "qpos_0_comparison.png",
        "qpos 0",
    )
    _save_signal_comparison_plot(
        raw_time,
        raw_qvel[:, 0],
        filtered_qvel[:, 0],
        "qvel[0] before/after filtering",
        "qvel (rad/s)",
        out_dir / "qvel_0_comparison.png",
        "qvel 0",
    )
    # Also save per-joint comparison plots so each joint can be inspected individually
    nj = raw_qpos.shape[1]
    for j in range(nj):
        _save_signal_comparison_plot(
            raw_time,
            raw_qpos[:, j],
            filtered_qpos[:, j],
            f"qpos[{j}] before/after filtering",
            "qpos (rad)",
            out_dir / f"qpos_{j}_comparison.png",
            f"qpos {j}",
        )
        _save_signal_comparison_plot(
            raw_time,
            raw_qvel[:, j],
            filtered_qvel[:, j],
            f"qvel[{j}] before/after filtering",
            "qvel (rad/s)",
            out_dir / f"qvel_{j}_comparison.png",
            f"qvel {j}",
        )


def make_model(xml_file: Optional[Path] = None) -> mj.MjModel:
    if xml_file is None:
        xml_file = Path(__file__).resolve().parent / "spiral_chain_wo_cylinder.xml"
    return mj.MjModel.from_xml_path(str(xml_file))


def set_params(model: mj.MjModel, stiffness: np.ndarray, damping: np.ndarray, tendon_stiffness: np.ndarray) -> None:
    for i in range(model.njnt):
        model.jnt_stiffness[i] = float(stiffness[i])
        dof = int(model.jnt_dofadr[i])
        model.dof_damping[dof] = float(damping[i])
    for i in range(model.ntendon):
        model.tendon_stiffness[i] = float(tendon_stiffness[i])


def get_local_model_and_data(xml_file: Optional[Path] = None):
    # Create model+data on demand; safe for multiprocessing workers
    model = make_model(xml_file)
    data = mj.MjData(model)
    return model, data


def should_preview_simulation(eval_index: int, preview_enabled: bool, preview_interval: int) -> bool:
    """Return True when the current evaluation should be shown in the passive viewer.

    Previewing is intentionally opt-in. The first evaluation is always shown,
    then every `preview_interval` evaluations.
    """
    if not preview_enabled:
        return False
    if eval_index <= 1:
        return True
    if preview_interval <= 0:
        return False
    return eval_index % preview_interval == 0


def preview_simulation_passive(
    model: mj.MjModel,
    qpos_init: np.ndarray,
    sim_time: float,
    record_dt: float = 0.0,
    title: str = "Simulation preview",
) -> None:
    """Show one replay pass in a passive MuJoCo viewer in real time.

    This is separate from the optimizer logic on purpose: when the viewer is
    disabled, this function is never called and does nothing to the run time.
    """
    if not _VIEWER_ENABLED:
        return
    if mj_viewer is None:
        return
    if model.nu < 2:
        return

    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    data.qpos[:] = qpos_init
    data.qvel[:] = np.zeros(model.nv)
    mj.mj_forward(model, data)

    dt = model.opt.timestep
    effective_sim_time = _ACTUAL_SIM_TIME if _ACTUAL_SIM_TIME > 0 else sim_time
    total_steps = max(1, int(effective_sim_time / dt))
    print(f"[viewer] {title}: real-time replay for {effective_sim_time:.3f}s at dt={dt:.4f}s")

    with mj_viewer.launch_passive(model, data) as viewer:
        step_index = 0
        while viewer.is_running() and step_index <= total_steps:
            step_start = time.time()
            apply_measured_forces(data, data.time)
            mj.mj_step(model, data)

            viewer.sync()
            dt_left = dt - (time.time() - step_start)
            if dt_left > 0:
                time.sleep(dt_left)

            step_index += 1


def apply_measured_forces(data: mj.MjData, t: float) -> tuple[float, float]:
    """Write the measured force profile into `data.ctrl` at simulation time `t`.

    `t` is MuJoCo's current simulation time. We use it only as the lookup key
    for the interpolated force signals; the controller itself is just replaying
    measured inputs.
    """
    if _FORCE_INTERP_0 is None or _FORCE_INTERP_1 is None:
        return 0.0, 0.0

    force_0 = float(_FORCE_INTERP_0(t))
    force_1 = float(_FORCE_INTERP_1(t))
    data.ctrl[0] = -force_0
    data.ctrl[1] = -force_1
    return force_0, force_1


def simulate_and_sample(model: mj.MjModel, data: mj.MjData, sim_time: float, qpos_init: np.ndarray, record_dt: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mj.mj_resetData(model, data)
    model.opt.timestep = 0.004
    model.opt.iterations = 20

    if model.nu < 2:
        raise RuntimeError(f"Expected at least 2 actuators for force replay, got model.nu={model.nu}")

    data.qpos[:] = qpos_init
    data.qvel[:] = np.zeros(model.nv)
    mj.mj_forward(model, data)

    # Settling phase: let the system calm down under the applied forces.
    # Apply the same interpolated force profile during settling so the system
    # reaches a force-induced equilibrium before we reset time and replay.
    settling_steps = 1000
    for _ in range(settling_steps):
        apply_measured_forces(data, data.time)
        mj.mj_step(model, data)
        data.qvel[:] = 0.0

    # After settling, restart trajectory time so the main simulation replays the
    # measured force profile from t=0 against a settled initial state.
    data.time = 0.0
    dt = model.opt.timestep
    # Use actual_sim_time if available (from preprocessing); fallback to sim_time
    effective_sim_time = _ACTUAL_SIM_TIME if _ACTUAL_SIM_TIME > 0 else sim_time
    total_steps = max(1, int(effective_sim_time / dt))

    # Determine recording interval: if record_dt > 0, record every N steps; else record all
    record_every = max(1, int(round(record_dt / dt))) if record_dt > 0 else 1

    qpos_hist = []
    qvel_hist = []
    applied_f0_hist = []
    applied_f1_hist = []
    for step in range(total_steps + 1):
        if step % record_every == 0:
            qpos_hist.append(data.qpos.copy())
            qvel_hist.append(data.qvel.copy())
            # record the force values applied at this simulation time (positive N)
            force_0, force_1 = apply_measured_forces(data, data.time)
            applied_f0_hist.append(force_0)
            applied_f1_hist.append(force_1)
        else:
            apply_measured_forces(data, data.time)
        mj.mj_step(model, data)

    qpos_arr = np.array(qpos_hist)
    qvel_arr = np.array(qvel_hist)
    applied_f0_arr = np.array(applied_f0_hist)
    applied_f1_arr = np.array(applied_f1_hist)

    # construct recorded time array for returned samples
    sim_steps = np.arange(0, total_steps + 1)
    record_indices = sim_steps[::record_every]
    sim_times = record_indices * dt

    # Print shapes once to help debugging and verification
    global _FIRST_SIM_PRINTED
    if not _FIRST_SIM_PRINTED:
        try:
            print(f"[simulate] produced est_qpos shape={qpos_arr.shape}, est_qvel shape={qvel_arr.shape}, record_dt={record_dt}, total_steps={total_steps}, record_every={record_every}, effective_sim_time={effective_sim_time:.4f}s")
        except Exception:
            pass
        _FIRST_SIM_PRINTED = True

    return qpos_arr, qvel_arr, applied_f0_arr, applied_f1_arr, sim_times


def load_and_preprocess(path: Path, sim_time: float, force_window: int = 60, data_smooth_window: int = 60, outlier_thresh: float = 0.0005, simple_outlier: bool = False, outlier_abs_thresh: float = 0.1, record_dt: float = 0.0):
    global _GT_QPOS, _GT_QVEL, _SIM_TIMESTEPS, _FORCE_INTERP_0, _FORCE_INTERP_1
    if not path.exists():
        raise FileNotFoundError(path)

    df = pl.read_parquet(path)
    joint_cols = [f"joint_{i}_deg" for i in range(1, 14)]
    df = df.drop_nulls(subset=["global_timestamp_s", "meas_force_0_N", "meas_force_1_N"] + joint_cols)
    if len(df) == 0:
        raise RuntimeError("No valid rows after dropping nulls")

    raw_time = df["global_timestamp_s"].to_numpy()
    raw_time = raw_time - raw_time[0]

    f0_raw = df["meas_force_0_N"].to_numpy()
    f1_raw = df["meas_force_1_N"].to_numpy()
    # Outlier detection: choose simple neighbor-based or MAD-based
    if simple_outlier:
        f0_filled, f0_mask = detect_and_fill_outliers_simple(f0_raw, abs_thresh=outlier_abs_thresh)
        f1_filled, f1_mask = detect_and_fill_outliers_simple(f1_raw, abs_thresh=outlier_abs_thresh)
    else:
        f0_filled, f0_mask = detect_and_fill_outliers(f0_raw, thresh=outlier_thresh)
        f1_filled, f1_mask = detect_and_fill_outliers(f1_raw, thresh=outlier_thresh)
    f0 = uniform_filter1d(f0_filled, size=force_window)
    f1 = uniform_filter1d(f1_filled, size=force_window)

    qpos_raw = np.zeros((len(df), 13))
    for i, col in enumerate(joint_cols):
        rads = np.deg2rad(df[col].to_numpy())
        qpos_raw[:, i] = rads

    # Apply real->simulation joint index mapping exactly once in preprocessing.
    qpos_raw_before_map = qpos_raw.copy()
    qpos_raw = _maybe_reverse_joint_order(qpos_raw, _REVERSE_REAL_JOINT_ORDER)
    print(f"Joint order mapping: reverse_real_joint_order={_REVERSE_REAL_JOINT_ORDER}")
    if len(qpos_raw) > 0:
        print(
            "[joint_map] sample t0 first3 before->after: "
            f"{np.array2string(qpos_raw_before_map[0, :3], precision=4)} -> "
            f"{np.array2string(qpos_raw[0, :3], precision=4)}"
        )

    # Per-joint outlier detection/filling (simple or MAD)
    qpos_filled = np.zeros_like(qpos_raw)
    qpos_masks = np.zeros_like(qpos_raw, dtype=bool)
    for i in range(qpos_raw.shape[1]):
        if simple_outlier:
            filled, mask = detect_and_fill_outliers_simple(qpos_raw[:, i], abs_thresh=outlier_abs_thresh)
        else:
            filled, mask = detect_and_fill_outliers(qpos_raw[:, i], thresh=outlier_thresh)
        qpos_filled[:, i] = filled
        qpos_masks[:, i] = mask

    # then smooth the filled signals
    qpos = np.zeros_like(qpos_raw)
    for i in range(qpos_raw.shape[1]):
        qpos[:, i] = uniform_filter1d(qpos_filled[:, i], size=data_smooth_window)

    qvel_raw = np.zeros_like(qpos_raw)
    if len(raw_time) > 1:
        for i in range(qpos_raw.shape[1]):
            # compute from raw/fill-equals-raw positions for inspection
            qvel_raw[:, i] = np.gradient(qpos_filled[:, i], raw_time)

    max_t = raw_time[-1] if sim_time <= 0 else min(sim_time, raw_time[-1])
    mask = raw_time <= max_t
    t_target = raw_time[mask]
    qpos_target = qpos[mask]
    f0_target = f0[mask]
    f1_target = f1[mask]

    # Store the actual simulation duration for use in cost function and validation
    global _ACTUAL_SIM_TIME
    _ACTUAL_SIM_TIME = max_t

    qvel_target = np.zeros_like(qpos_target)
    if len(t_target) > 1:
        for i in range(qpos_target.shape[1]):
            raw_vel = np.gradient(qpos[mask, i], t_target)
            # also smooth velocities
            qvel_target[:, i] = uniform_filter1d(raw_vel, size=data_smooth_window)

    # ── Uniform downsampling (if record_dt > 0) ────────────────────────────────────────
    # This removes data points evenly across the entire dataset, not just truncating at end.
    # Critical: ensures real-data samples align with simulation samples (both use same record_dt).
    if record_dt > 0:
        # Use one shared index selection for all signals to preserve strict alignment.
        if len(t_target) > 1:
            dt_actual = np.mean(np.diff(t_target))
            if dt_actual > 0:
                record_every = max(1, int(round(record_dt / dt_actual)))
                sel_idx = np.arange(0, len(t_target), record_every)
            else:
                sel_idx = np.arange(len(t_target))
        else:
            sel_idx = np.arange(len(t_target))

        t_target = t_target[sel_idx]
        qpos_target = qpos_target[sel_idx]
        qvel_target = qvel_target[sel_idx]
        f0_target = f0_target[sel_idx]
        f1_target = f1_target[sel_idx]

        # Ensure final lengths match what the simulation will record.
        # Compute expected simulated sample count from sim_time and record_dt
        if sim_time > 0:
            expected_sim_samples = int(np.floor(sim_time / record_dt)) + 1
        else:
            expected_sim_samples = len(t_target)

        if expected_sim_samples != len(t_target):
            final_n = min(expected_sim_samples, len(t_target))
            sel_idx = np.round(np.linspace(0, len(t_target) - 1, final_n)).astype(int)
            t_target = t_target[sel_idx]
            qpos_target = qpos_target[sel_idx]
            qvel_target = qvel_target[sel_idx]
            f0_target = f0_target[sel_idx]
            f1_target = f1_target[sel_idx]

    save_preprocessing_plots(
        raw_time,
        f0_raw,
        f1_raw,
        f0,
        f1,
        qpos_raw,
        qpos,
        qvel_raw,
        qvel_target,
        Path(__file__).resolve().parent / "build" / "preprocessing_plots",
    )

    _GT_QPOS = qpos_target
    _GT_QVEL = qvel_target
    _SIM_TIMESTEPS = t_target
    _FORCE_INTERP_0 = interp1d(t_target, f0_target, kind='linear', fill_value=(f0_target[0], f0_target[-1]), bounds_error=False)
    _FORCE_INTERP_1 = interp1d(t_target, f1_target, kind='linear', fill_value=(f1_target[0], f1_target[-1]), bounds_error=False)

    return {
        "t": t_target,
        "qpos": qpos_target,
        "qvel": qvel_target,
        "f0": f0_target,
        "f1": f1_target,
    }


def cost_function_raw(stiff: np.ndarray, damp: np.ndarray, t_stiff: np.ndarray, sim_time: float, xml_file: Optional[Path]) -> float:
    # Build model+data inside worker/process
    model, data = get_local_model_and_data(xml_file)
    if np.any(stiff <= 0) or np.any(damp <= 0) or np.any(t_stiff <= 0):
        return 1e6
    global _EVAL_COUNTER
    _EVAL_COUNTER += 1
    try:
        set_params(model, stiff, damp, t_stiff)
        est_qpos, est_qvel, _, _, _ = simulate_and_sample(model, data, sim_time, _GT_QPOS[0], record_dt=_RECORD_DT)
    except Exception:
        return 1e6

    # Print shapes used in cost computation once for debugging
    global _FIRST_COST_PRINTED
    if not _FIRST_COST_PRINTED:
        try:
            print(f"[cost_function] _GT_QPOS shape={getattr(_GT_QPOS, 'shape', None)}, est_qpos shape={getattr(est_qpos, 'shape', None)}, _GT_QVEL shape={getattr(_GT_QVEL, 'shape', None)}, est_qvel shape={getattr(est_qvel, 'shape', None)}")
        except Exception:
            pass
        _FIRST_COST_PRINTED = True

    len_gt = len(_GT_QPOS)
    len_est = len(est_qpos)
    if len_gt == 0 or len_est == 0:
        return 1e6

    match_mode = "equal"
    if len_gt > len_est:
        idx = np.round(np.linspace(0, len_gt - 1, len_est)).astype(int)
        gt_pos_match = _GT_QPOS[idx]
        gt_vel_match = _GT_QVEL[idx]
        est_pos_match = est_qpos
        est_vel_match = est_qvel
        match_mode = "resample_gt_to_est"
    elif len_est > len_gt:
        idx = np.round(np.linspace(0, len_est - 1, len_gt)).astype(int)
        est_pos_match = est_qpos[idx]
        est_vel_match = est_qvel[idx]
        gt_pos_match = _GT_QPOS
        gt_vel_match = _GT_QVEL
        match_mode = "resample_est_to_gt"
    else:
        gt_pos_match = _GT_QPOS
        gt_vel_match = _GT_QVEL
        est_pos_match = est_qpos
        est_vel_match = est_qvel

    # Safety check: cost should always be computed on equal-sized arrays.
    if gt_pos_match.shape != est_pos_match.shape or gt_vel_match.shape != est_vel_match.shape:
        return 1e6

    # One-time debug print that shows the actual arrays entering cost computation.
    global _FIRST_COST_MATCH_PRINTED
    if not _FIRST_COST_MATCH_PRINTED:
        print(
            "[cost_match] "
            f"mode={match_mode}, "
            f"gt_pos_match shape={gt_pos_match.shape}, est_pos_match shape={est_pos_match.shape}, "
            f"gt_vel_match shape={gt_vel_match.shape}, est_vel_match shape={est_vel_match.shape}"
        )
        try:
            gt_span = np.rad2deg(np.max(gt_pos_match, axis=0) - np.min(gt_pos_match, axis=0))
            est_span = np.rad2deg(np.max(est_pos_match, axis=0) - np.min(est_pos_match, axis=0))
            print(
                "[motion_span_deg] "
                f"gt mean={float(np.mean(gt_span)):.3f}, est mean={float(np.mean(est_span)):.3f}, "
                f"gt max={float(np.max(gt_span)):.3f}, est max={float(np.max(est_span)):.3f}"
            )
        except Exception:
            pass
        _FIRST_COST_MATCH_PRINTED = True

    err_pos_deg = np.rad2deg(gt_pos_match - est_pos_match)
    err_vel_deg = np.rad2deg(gt_vel_match - est_vel_match)
    n = len(err_pos_deg)
    w = np.linspace(1.1, 0.8, n).reshape(-1, 1)
    rmse_pos = np.sqrt(np.mean(w * err_pos_deg ** 2))
    rmse_vel = np.sqrt(np.mean(w * err_vel_deg ** 2))
    cost = float(rmse_pos + 0.5 * rmse_vel)

    if should_preview_simulation(_EVAL_COUNTER, _VIEWER_ENABLED, _VIEWER_INTERVAL):
        try:
            preview_simulation_passive(model, _GT_QPOS[0], sim_time, record_dt=_RECORD_DT, title=f"evaluation {_EVAL_COUNTER}")
        except Exception as exc:
            print(f"[viewer] preview skipped: {exc}")
    return cost


def write_cost_function_debug_parquet(
    gt_pos_match: np.ndarray,
    est_pos_match: np.ndarray,
    gt_vel_match: np.ndarray,
    est_vel_match: np.ndarray,
    out_path: Path,
) -> None:
    err_pos_deg_debug = np.rad2deg(gt_pos_match - est_pos_match)
    n_debug = len(err_pos_deg_debug)
    w_debug = np.linspace(1.1, 0.8, n_debug)

    data_dict = {"step": np.arange(n_debug), "weight": w_debug}
    nj = gt_pos_match.shape[1]
    for j in range(nj):
        data_dict[f"gt_pos_deg_j{j}"] = np.rad2deg(gt_pos_match[:, j])
        data_dict[f"est_pos_deg_j{j}"] = np.rad2deg(est_pos_match[:, j])
        data_dict[f"gt_vel_deg_j{j}"] = np.rad2deg(gt_vel_match[:, j])
        data_dict[f"est_vel_deg_j{j}"] = np.rad2deg(est_vel_match[:, j])
        data_dict[f"err_deg_j{j}"] = err_pos_deg_debug[:, j]
        data_dict[f"weighted_sq_err_j{j}"] = w_debug * (err_pos_deg_debug[:, j] ** 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data_dict).write_parquet(out_path)


def global_objective_stiff_damp(
    x_b1: np.ndarray,
    sim_time: float,
    njnt: int,
    ntendon: int,
    xml_file: Optional[Path] = None,
) -> float:
    stiff_scale = np.full(njnt, INIT_BASE_STIFFNESS)
    damp_scale = np.full(njnt, INIT_BASE_DAMPING)
    stiff = x_b1[:njnt] * stiff_scale
    damp = x_b1[njnt:] * damp_scale
    t_stiff = np.full(ntendon, 500.0)
    return cost_function_raw(stiff, damp, t_stiff, sim_time, xml_file)


def optimize_stiffness_damping(
    sim_time: float,
    njnt: int,
    ntendon: int,
    xml_file: Optional[Path],
    workers: int = 1,
    maxiter: int = 10,
    tol: float = 0.01,
    pop_mult: float = 5.0,
):
    bounds_stiff = [
        (BOUNDS_STIFFNESS[0] / INIT_BASE_STIFFNESS, BOUNDS_STIFFNESS[1] / INIT_BASE_STIFFNESS)
        for _ in range(njnt)
    ]
    bounds_damp = [
        (BOUNDS_DAMPING[0] / INIT_BASE_DAMPING, BOUNDS_DAMPING[1] / INIT_BASE_DAMPING)
        for _ in range(njnt)
    ]
    bounds = bounds_stiff + bounds_damp
    curr_b1 = np.ones(2 * njnt)
    history = []
    gen_counter = 0

    def cb(xk, convergence=None):
        nonlocal gen_counter
        gen_counter += 1
        history.append(xk.copy())
        best_cost = global_objective_stiff_damp(xk, sim_time, njnt, ntendon, xml_file)
        print(f"[gen {gen_counter:03d}/{maxiter:03d}] cost={best_cost:.6e} convergence={convergence:.3e}")

    popsize_int = max(1, int(round(pop_mult)))
    total_population = popsize_int * len(bounds)
    if total_population < max(1, workers):
        popsize_int = int(np.ceil(workers / max(1, len(bounds))))
        total_population = popsize_int * len(bounds)

    print(
        f"[DE setup] pop_mult={pop_mult} -> popsize={popsize_int} (per-dim), "
        f"total_population={total_population}, workers={workers}, nvars={len(bounds)}"
    )

    res = differential_evolution(
        global_objective_stiff_damp,
        args=(sim_time, njnt, ntendon, xml_file),
        bounds=bounds,
        x0=curr_b1,
        maxiter=maxiter,
        tol=tol,
        seed=42,
        polish=True,
        popsize=popsize_int,
        workers=workers,
        disp=False,
        callback=cb,
    )

    stiff_phys = res.x[:njnt] * np.full(njnt, INIT_BASE_STIFFNESS)
    damp_phys = res.x[njnt:] * np.full(njnt, INIT_BASE_DAMPING)
    return res, stiff_phys, damp_phys, history


def save_validation_plot(
    gt_qpos: np.ndarray,
    est_qpos: np.ndarray,
    out_path: Path,
    gt_force_times: Optional[np.ndarray] = None,
    gt_f0: Optional[np.ndarray] = None,
    gt_f1: Optional[np.ndarray] = None,
    sim_times: Optional[np.ndarray] = None,
    sim_f0: Optional[np.ndarray] = None,
    sim_f1: Optional[np.ndarray] = None,
):
    if plt is None:
        return
    njnt = gt_qpos.shape[1]
    # create 3 rows: GT qpos, sim qpos, forces
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    colors = plt.cm.tab20(np.linspace(0, 1, njnt))
    for i in range(njnt):
        axs[0].plot(gt_qpos[:, i], color=colors[i], label="ground truth" if i == 0 else None)
    axs[0].set_title("Ground Truth Trajectories")

    for i in range(njnt):
        axs[1].plot(est_qpos[:, i], color=colors[i], label="identified" if i == 0 else None)
    axs[1].set_title("Simulated Trajectories (Identified)")
    axs[1].legend(loc='center left', bbox_to_anchor=(1.0, 1.0))

    # Forces: plot GT forces (if provided) and applied sim forces (if provided)
    if (gt_force_times is not None and gt_f0 is not None) or (sim_times is not None and sim_f0 is not None):
        if gt_force_times is not None and gt_f0 is not None:
            axs[2].plot(gt_force_times, gt_f0, color="steelblue", label="gt force 0")
        if gt_force_times is not None and gt_f1 is not None:
            axs[2].plot(gt_force_times, gt_f1, color="darkorange", label="gt force 1")
        if sim_times is not None and sim_f0 is not None:
            axs[2].plot(sim_times, sim_f0, color="navy", linestyle="--", label="sim applied f0")
        if sim_times is not None and sim_f1 is not None:
            axs[2].plot(sim_times, sim_f1, color="orangered", linestyle="--", label="sim applied f1")
        axs[2].set_title("Forces: ground-truth (solid) vs applied (dashed)")
        axs[2].set_ylabel("force (N)")
        axs[2].legend(loc='best')

    axs[-1].set_xlabel("time (samples)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-time", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--inner-maxiter", type=int, default=4)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--outlier-thresh", type=float, default=3.5, help="MAD z-score threshold for outlier detection")
    ap.add_argument("--simple-outlier", action="store_true", help="Use simple neighbor-diff outlier detection")
    ap.add_argument("--outlier-abs-thresh", type=float, default=0.1, help="Absolute threshold for simple outlier detection (units: same as signal)")
    ap.add_argument("--record-dt", type=float, default=0.0, help="Recording interval in seconds (e.g., 0.1 for 10 Hz); 0 = no downsampling")
    ap.add_argument(
        "--reverse-real-joints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reverse real joint order to match simulation indexing (default: module global)",
    )
    ap.add_argument("--preprocess-only", action="store_true", help="Run only preprocessing and save plots, skip optimization")
    ap.add_argument("--tol", type=float, default=0.01, help="Tolerance for differential_evolution convergence")
    ap.add_argument("--pop-mult", type=float, default=5.0, help="Multiplier for DE popsize (per-dimension). total_pop = pop_mult * nvars")
    ap.add_argument("--enable-viewer", action=argparse.BooleanOptionalAction, default=False, help="Show passive MuJoCo previews during optimization")
    ap.add_argument("--viewer-interval", type=int, default=100, help="Show the first evaluation and then every Nth evaluation in the passive viewer")
    args = ap.parse_args()

    if args.data:
        data_path = Path(args.data)
    else:
        base = Path(__file__).resolve().parent / "build"
        cand1 = base / "sys_id_auto_GX010070.parquet"
        cand2 = base / "sys_id_auto.parquet"
        if cand1.exists():
            data_path = cand1
        elif cand2.exists():
            data_path = cand2
        else:
            print(f"Data file not found in {base} (tried sys_id_auto_GX010070.parquet and sys_id_auto.parquet)")
            return

    print("Loading and preprocessing data...")
    global _RECORD_DT, _REVERSE_REAL_JOINT_ORDER, _ACTUAL_SIM_TIME
    _RECORD_DT = float(args.record_dt)
    if args.reverse_real_joints is not None:
        _REVERSE_REAL_JOINT_ORDER = bool(args.reverse_real_joints)
    global _VIEWER_ENABLED, _VIEWER_INTERVAL
    _VIEWER_ENABLED = bool(args.enable_viewer)
    _VIEWER_INTERVAL = max(1, int(args.viewer_interval))

    # Cleanup stale lock from older versions of this script
    debug_lock_path = Path(__file__).resolve().parent / "build" / "cost_function_debug_first_run.lock"
    try:
        if debug_lock_path.exists():
            debug_lock_path.unlink()
    except Exception:
        pass

    meta = load_and_preprocess(
        data_path,
        args.sim_time,
        outlier_thresh=args.outlier_thresh,
        simple_outlier=args.simple_outlier,
        outlier_abs_thresh=args.outlier_abs_thresh,
        record_dt=_RECORD_DT,
    )
    njnt = meta["qpos"].shape[1]
    ntendon = 2

    real_samples = len(meta["t"]) if meta and "t" in meta else 0
    real_duration = meta["t"][-1] if meta and "t" in meta and len(meta["t"]) > 0 else 0
    print(f"Preprocessing: real samples after downsampling: {real_samples}, real_duration: {real_duration:.4f}s")
    print(f"Requested sim_time: {args.sim_time:.4f}s, actual_sim_time: {_ACTUAL_SIM_TIME:.4f}s")
    if _RECORD_DT > 0 and _ACTUAL_SIM_TIME > 0:
        dt_sim = 0.004
        record_every = max(1, int(round(_RECORD_DT / dt_sim)))
        total_steps = int(_ACTUAL_SIM_TIME / dt_sim)
        sim_samples = total_steps // record_every + 1
        print(f"Expected simulated samples (simulate record_dt={_RECORD_DT}s, actual_sim_time={_ACTUAL_SIM_TIME:.4f}s): {sim_samples}")
        if sim_samples != real_samples:
            print(f"Note: mismatch detected ({sim_samples} sim vs {real_samples} real) — will be resampled uniformly during cost computation.")

    # Debug parquet once from main process (all quantities in degrees)
    try:
        debug_out_path = Path(__file__).resolve().parent / "build" / "cost_function_debug_first_run.parquet"
        model_dbg, data_dbg = get_local_model_and_data(None)
        set_params(
            model_dbg,
            np.full(njnt, INIT_BASE_STIFFNESS),
            np.full(njnt, INIT_BASE_DAMPING),
            np.full(ntendon, 500.0),
        )
        est_qpos_dbg, est_qvel_dbg, _, _, _ = simulate_and_sample(
            model_dbg,
            data_dbg,
            _ACTUAL_SIM_TIME,
            _GT_QPOS[0],
            record_dt=_RECORD_DT,
        )

        len_gt_dbg = len(_GT_QPOS)
        len_est_dbg = len(est_qpos_dbg)
        if len_gt_dbg > len_est_dbg:
            idx_dbg = np.round(np.linspace(0, len_gt_dbg - 1, len_est_dbg)).astype(int)
            gt_pos_match_dbg = _GT_QPOS[idx_dbg]
            gt_vel_match_dbg = _GT_QVEL[idx_dbg]
            est_pos_match_dbg = est_qpos_dbg
            est_vel_match_dbg = est_qvel_dbg
            dbg_mode = "resample_gt_to_est"
        elif len_est_dbg > len_gt_dbg:
            idx_dbg = np.round(np.linspace(0, len_est_dbg - 1, len_gt_dbg)).astype(int)
            est_pos_match_dbg = est_qpos_dbg[idx_dbg]
            est_vel_match_dbg = est_qvel_dbg[idx_dbg]
            gt_pos_match_dbg = _GT_QPOS
            gt_vel_match_dbg = _GT_QVEL
            dbg_mode = "resample_est_to_gt"
        else:
            gt_pos_match_dbg = _GT_QPOS
            gt_vel_match_dbg = _GT_QVEL
            est_pos_match_dbg = est_qpos_dbg
            est_vel_match_dbg = est_qvel_dbg
            dbg_mode = "equal"

        print(
            "[cost_match_debug] "
            f"mode={dbg_mode}, gt_pos_match shape={gt_pos_match_dbg.shape}, est_pos_match shape={est_pos_match_dbg.shape}"
        )
        gt_span_dbg = np.rad2deg(np.max(gt_pos_match_dbg, axis=0) - np.min(gt_pos_match_dbg, axis=0))
        est_span_dbg = np.rad2deg(np.max(est_pos_match_dbg, axis=0) - np.min(est_pos_match_dbg, axis=0))
        print(
            "[motion_span_deg] "
            f"gt mean={float(np.mean(gt_span_dbg)):.3f}, est mean={float(np.mean(est_span_dbg)):.3f}, "
            f"gt max={float(np.max(gt_span_dbg)):.3f}, est max={float(np.max(est_span_dbg)):.3f}"
        )
        write_cost_function_debug_parquet(
            gt_pos_match_dbg,
            est_pos_match_dbg,
            gt_vel_match_dbg,
            est_vel_match_dbg,
            debug_out_path,
        )
        print(f"[cost_match_debug] Saved full step-by-step cost computation data to: {debug_out_path}")
    except Exception as exc:
        print(f"[cost_match_debug] Failed to write debug parquet: {exc}")

    if args.preprocess_only:
        print("Preprocessing complete — saved plots to build/preprocessing_plots. Exiting due to --preprocess-only.")
        return

    print("Starting optimization (stiffness + damping)...")
    if _VIEWER_ENABLED and args.workers != 1:
        print("[viewer] enabled with workers != 1; previews may be skipped or unstable. For reliable real-time preview, use --workers 1.")

    res, stiff_res, damp_res, history = optimize_stiffness_damping(
        args.sim_time,
        njnt,
        ntendon,
        None,
        workers=args.workers,
        maxiter=args.inner_maxiter,
        tol=args.tol,
        pop_mult=args.pop_mult,
    )

    model, data = get_local_model_and_data(None)
    set_params(model, stiff_res, damp_res, np.full(ntendon, 500.0))
    est_qpos, _, sim_f0, sim_f1, sim_t = simulate_and_sample(
        model,
        data,
        _ACTUAL_SIM_TIME,
        _GT_QPOS[0],
        record_dt=_RECORD_DT,
    )

    out_plot = Path(__file__).resolve().parent / "build" / "sysid_real_modular_validation.png"
    gt_ft = None
    gt_f0 = None
    gt_f1 = None
    try:
        if _SIM_TIMESTEPS is not None and _FORCE_INTERP_0 is not None:
            gt_ft = _SIM_TIMESTEPS
            gt_f0 = _FORCE_INTERP_0(_SIM_TIMESTEPS)
        if _SIM_TIMESTEPS is not None and _FORCE_INTERP_1 is not None:
            gt_f1 = _FORCE_INTERP_1(_SIM_TIMESTEPS)
    except Exception:
        gt_ft = None
        gt_f0 = None
        gt_f1 = None

    save_validation_plot(
        _GT_QPOS,
        est_qpos,
        out_plot,
        gt_force_times=gt_ft,
        gt_f0=gt_f0,
        gt_f1=gt_f1,
        sim_times=sim_t,
        sim_f0=sim_f0,
        sim_f1=sim_f1,
    )
    print(f"Saved validation plot to: {out_plot}")

    out_file = Path(__file__).resolve().parent / "build" / "sysid_real_modular_params.json"
    out = {
        "stiffness": stiff_res.tolist(),
        "damping": damp_res.tolist(),
        "cost": float(res.fun),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved params to: {out_file}")


if __name__ == "__main__":
    main()
