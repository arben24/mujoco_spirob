import argparse
import time
from pathlib import Path
from typing import Any

import mujoco as mj
import mujoco.viewer as mj_viewer
import numpy as np
from scipy.optimize import differential_evolution

import math_spirob.spirob_generator as sg

# ── Modell-Geometrie (feststehend) ──────────────────────────────────
L_TARGET = 0.44
BASE_D = 0.1
TIP_D = 0.03
DELTA_THETA_DEG = 30.0

# ── Ground-Truth-Base-Parameter ────────────────────────────────────
GT_BASE_STIFFNESS = 0.08
GT_BASE_DAMPING = 0.12
GT_BASE_TENDON_STIFFNESS = 500.0
# GT_BASE_ARMATURE = 0.015

# ── Startwerte für die Optimierung (Init Base) ─────────────────────
INIT_BASE_STIFFNESS = 0.8
INIT_BASE_DAMPING = 0.02
INIT_BASE_TENDON_STIFFNESS = 500.0
# INIT_BASE_ARMATURE = 0.005

# ── Suchraum-Grenzen (physikalisch sinnvoll) ──────────────────────
BOUNDS_STIFFNESS = (0.01, 1)
BOUNDS_DAMPING = (0.01, 0.2)
BOUNDS_TENDON_STIFFNESS = (450.0, 550.0)
# BOUNDS_ARMATURE = (1e-3, 0.1)

# ── Initiale Joint-Konfiguration (Startkonfiguration) ──────────────
# Dieser Winkel (in Grad) wird auf alle Gelenke angewendet
INIT_JOINT_ANGLE_DEG = 20.0  # 0 = Nullposition; z.B. 20.0 für 20° auf jedem Joint


# ====================================================================
#  Hilfsfunktionen
# ====================================================================

def make_model() -> mj.MjModel:
    xml = sg.generate_xml_string(
        L_target=L_TARGET,
        base_d=BASE_D,
        tip_d=TIP_D,
        Delta_theta_deg=DELTA_THETA_DEG,
        auto_format=True,
    )
    return mj.MjModel.from_xml_string(xml)

def set_params(model: mj.MjModel,
               stiffness: np.ndarray,
               damping: np.ndarray,
               tendon_stiffness: np.ndarray) -> None:
    """Setzt individuelle physikalische Parameter (Arrays) auf alle Joints / Tendons."""
    for i in range(model.njnt):
        model.jnt_stiffness[i] = stiffness[i]
        dof = model.jnt_dofadr[i]
        model.dof_damping[dof] = damping[i]
        # model.dof_armature[dof] = armature[i]
    for i in range(model.ntendon):
        model.tendon_stiffness[i] = tendon_stiffness[i]

def controller(model: mj.MjModel, data: mj.MjData, t: float) -> None:
    # ramp = min(t / 0.5, 1.0)
    # s1 = np.sin(2 * np.pi * 0.5 * t)
    # s2 = np.sin(2 * np.pi * 1.5 * t)
    # s3 = np.sin(2 * np.pi * 3.0 * t)
    # data.ctrl[0] = -ramp * (5.0 + 3.0 * s1 + 1.5 * s2)
    # data.ctrl[1] = -ramp * (8.0 + 4.0 * s1 - 2.0 * s3)
    data.ctrl[0] = -30.0 * np.sin(2 * np.pi * 0.5 * t) 
    data.ctrl[1] = -20.0 * np.sin(2 * np.pi * 0.5 * t + np.pi / 4)

def simulate(model: mj.MjModel, sim_time: float,
             record_dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    
    # Setze initiale Joint-Positionen (in Radiant)
    init_angle_rad = np.deg2rad(INIT_JOINT_ANGLE_DEG)
    for i in range(model.njnt):
        data.qpos[i] = init_angle_rad
    mj.mj_forward(model, data)  # Update internal state
    
    model.opt.timestep = 0.02
    dt = model.opt.timestep
    record_every = max(1, round(record_dt / dt))
    total_steps = int(sim_time / dt)

    qpos_list, qvel_list = [], []
    for step in range(total_steps + 1):
        if step % record_every == 0:
            qpos_list.append(data.qpos.copy())
            qvel_list.append(data.qvel.copy())
        controller(model, data, data.time)
        mj.mj_step(model, data)
    #print(f"Simulated {len(qpos_list)} recorded steps out of {total_steps} total steps.")
    #print(max(qpos_list[:][0]), min(qpos_list[:][0]))
    return np.array(qpos_list), np.array(qvel_list)

def decode_params(x_scaled: np.ndarray, njnt: int, ntendon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrahiert die 3 Parameter-Arrays aus dem flachen skalaren Vektor."""
    stiffness = x_scaled[0:njnt]
    damping = x_scaled[njnt:2*njnt]
    # armature = x_scaled[2*njnt:3*njnt]
    tendon_stiffness = x_scaled[2*njnt:2*njnt+ntendon]
    return stiffness, damping, tendon_stiffness

# ====================================================================
#  Multiprocessing-Hilfen
# ====================================================================

_LOCAL_MODEL = None

def get_local_model() -> mj.MjModel:
    """Caching des MuJoCo-Modells pro Thread/Prozess, da MjModels nicht serialisierbar (pickable) sind."""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        _LOCAL_MODEL = make_model()
    return _LOCAL_MODEL

def global_objective(x: np.ndarray, scale: np.ndarray, gt_qpos: np.ndarray, gt_qvel: np.ndarray, sim_time: float, njnt: int, ntendon: int) -> float:
    """Globale Objective-Funktion, die auch bei Multiprocessing (workers>1) erfolgreich serialisiert werden kann."""
    model = get_local_model()
    # Timing wird bei MP auf None gesetzt, um Multiprocessing-Race-Conditions/Synchronisations overhead zu vermeiden
    return cost_function(x, scale, gt_qpos, gt_qvel, sim_time, model, njnt, ntendon, timing=None)

# ====================================================================
#  Kostenfunktion
# ====================================================================

def cost_function(params_norm: np.ndarray,
                  scale: np.ndarray,
                  gt_qpos: np.ndarray,
                  gt_qvel: np.ndarray,
                  sim_time: float,
                  model: mj.MjModel,
                  njnt: int,
                  ntendon: int,
                  timing: dict[str, Any] | None = None) -> float:
    t_cost_start = time.perf_counter()
    phys = params_norm * scale
    
    if timing is not None:
        timing["cost_calls"] += 1

    if np.any(phys <= 0):
        if timing is not None:
            timing["cost_invalid_param_calls"] += 1
            timing["cost_total_s"] += time.perf_counter() - t_cost_start
        return 1e6

    stiff, damp, t_stiff = decode_params(phys, njnt, ntendon)

    try:
        t0 = time.perf_counter()
        set_params(model, stiff, damp, t_stiff)
        if timing is not None:
            timing["cost_set_params_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        est_qpos, est_qvel = simulate(model, sim_time)
        if timing is not None:
            timing["cost_simulate_s"] += time.perf_counter() - t0
    except Exception:
        if timing is not None:
            timing["cost_exception_calls"] += 1
            timing["cost_total_s"] += time.perf_counter() - t_cost_start
        return 1e6

    t0 = time.perf_counter()
    n = min(len(gt_qpos), len(est_qpos))
    w = np.linspace(0.5, 1.5, n).reshape(-1, 1)
    mse_pos = np.mean(w * (gt_qpos[:n] - est_qpos[:n]) ** 2)
    mse_vel = np.mean(w * (gt_qvel[:n] - est_qvel[:n]) ** 2)

    if timing is not None:
        timing["cost_postprocess_s"] += time.perf_counter() - t0
        timing["cost_total_s"] += time.perf_counter() - t_cost_start

    return mse_pos + 0.5 * mse_vel


# ====================================================================
#  Main
# ====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-Parameter SpiRob System-ID")
    ap.add_argument("--sim-time", type=float, default=2.0)
    ap.add_argument("--maxiter", type=int, default=50) # Ähnliche Werte wie in simple
    ap.add_argument("--tol", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=10, help="Anzahl parallel laufender Prozesse bei differential_evolution")
    ap.add_argument("--profile-timing", action="store_true")
    args = ap.parse_args()

    sim_time: float = args.sim_time
    maxiter: int = args.maxiter
    tol: float = args.tol
    workers: int = args.workers

    timing: dict[str, Any] = {
        "objective_calls": 0, "objective_total_s": 0.0,
        "cost_calls": 0, "cost_total_s": 0.0,
        "opt_model_build_s": 0.0, "cost_set_params_s": 0.0,
        "cost_simulate_s": 0.0, "cost_postprocess_s": 0.0,
        "cost_invalid_param_calls": 0, "cost_exception_calls": 0,
    }

    print("=" * 55)
    print("Multi-Parameter Ground-Truth simulieren …")
    
    gt_model = make_model()
    njnt = gt_model.njnt
    ntendon = gt_model.ntendon
    
    # -------------------------------------------------------------
    # MANUELLE GROUND TRUTH WERTE HIER (Arrays der entsprechenden Länge)
    # Beispielhaft leicht gestreute Werte um den Basis-Parameter
    # (Damit jedes Gelenk etwas anders reagiert / du dies manuell ändern kannst)
    # -------------------------------------------------------------
    np.random.seed(42)
    gt_stiffness = np.random.normal(GT_BASE_STIFFNESS, 0.005, njnt)
    gt_damping = np.random.normal(GT_BASE_DAMPING, 0.005, njnt)
    # gt_armature = np.random.normal(GT_BASE_ARMATURE, 0.001, njnt)
    gt_tendon_stiffness = np.random.normal(GT_BASE_TENDON_STIFFNESS, 4.0, ntendon)
    
    # Falls du feste Arrays eintragen möchtest, einfach hier überschreiben, z.B.:
    # gt_stiffness = np.array([0.08, 0.09, 0.08, 0.1, ... <njnt-Werte>])

    print(f"  Modell: {njnt} Joints, {ntendon} Tendons")
    print(f"  Suchraum: {2 * njnt + ntendon} Dimensionen (Parameter) total!")
    
    print("\n  --- Ground Truth Werte ---")
    print(f"  GT Stiffness: {np.array2string(gt_stiffness, precision=4, max_line_width=120)}")
    print(f"  GT Damping:   {np.array2string(gt_damping, precision=4, max_line_width=120)}")
    # print(f"  GT Armature:  {np.array2string(gt_armature, precision=5, max_line_width=120)}")
    print(f"  GT T_Stiff:   {np.array2string(gt_tendon_stiffness, precision=2, max_line_width=120)}")
    print("  --------------------------\n")
    
    set_params(gt_model, gt_stiffness, gt_damping, gt_tendon_stiffness)

    gt_qpos, gt_qvel = simulate(gt_model, sim_time)
    print(f"  {len(gt_qpos)} Datenpunkte simuliert")

    # ── 2. Optimierung vorbereiten ──────────────────────────────────
    # Wir kombinieren alle Parameter in ein einzelnes 1D-Array
    scale = np.concatenate([
        np.full(njnt, INIT_BASE_STIFFNESS),
        np.full(njnt, INIT_BASE_DAMPING),
        # np.full(njnt, INIT_BASE_ARMATURE),
        np.full(ntendon, INIT_BASE_TENDON_STIFFNESS)
    ])

    norm_bounds = []
    for _ in range(njnt): norm_bounds.append((BOUNDS_STIFFNESS[0]/INIT_BASE_STIFFNESS, BOUNDS_STIFFNESS[1]/INIT_BASE_STIFFNESS))
    for _ in range(njnt): norm_bounds.append((BOUNDS_DAMPING[0]/INIT_BASE_DAMPING, BOUNDS_DAMPING[1]/INIT_BASE_DAMPING))
    # for _ in range(njnt): norm_bounds.append((BOUNDS_ARMATURE[0]/INIT_BASE_ARMATURE, BOUNDS_ARMATURE[1]/INIT_BASE_ARMATURE))
    for _ in range(ntendon): norm_bounds.append((BOUNDS_TENDON_STIFFNESS[0]/INIT_BASE_TENDON_STIFFNESS, BOUNDS_TENDON_STIFFNESS[1]/INIT_BASE_TENDON_STIFFNESS))

    x0 = np.ones(len(scale))

    t_model_build = time.perf_counter()
    opt_model = make_model()
    timing["opt_model_build_s"] = time.perf_counter() - t_model_build

    # ── 3. Optimierung laufen lassen ────────────────────────────────
    print("\n" + "=" * 55 + "\nOptimierung starten (differential_evolution)\n" + "-" * 55)

    cost_history = []  # Track best cost after each generation
    
    def callback_collect_cost(xk, convergence=None):
        """Callback to collect the best cost after each generation."""
        current_cost = global_objective(xk, scale, gt_qpos, gt_qvel, sim_time, njnt, ntendon)
        cost_history.append(current_cost)

    t0 = time.time()
    
    result = differential_evolution(
        global_objective,
        args=(scale, gt_qpos, gt_qvel, sim_time, njnt, ntendon),
        bounds=norm_bounds,
        x0=x0,
        maxiter=maxiter,
        tol=tol,
        seed=42,
        polish=True,
        init="sobol",
        popsize=2*workers,
        #mutation=(0.5, 1.5),
        recombination=0.9,
        workers=workers,
        disp=True,
        callback=callback_collect_cost,
    )
        
    elapsed = time.time() - t0

    # ── 4. Ergebnis ────────────────────────────────────────────────
    phys = result.x * scale
    stiff_res, damp_res, t_stiff_res = decode_params(phys, njnt, ntendon)
    
    print("\n" + "=" * 55 + "\nERGEBNIS")
    print(f"  Status  : {result.message}\n  Cost    : {result.fun:.10e}\n  Aufrufe : {result.nfev}\n  Dauer   : {elapsed:.1f} s\n")
    
    print("  Durchschnittliche Parameter-Fehler (MEAN):")
    print(f"  {'Parameter':>20s}  {'GT Mean':>10s}  {'Identif.':>10s}  {'Fehler':>8s}")
    print("  " + "-" * 54)
    
    res_groups = [
        ("joint_stiffness", gt_stiffness.mean(), stiff_res.mean()),
        ("joint_damping", gt_damping.mean(), damp_res.mean()),
        # ("armature", gt_armature.mean(), arm_res.mean()),
        ("tendon_stiffness", gt_tendon_stiffness.mean(), t_stiff_res.mean())
    ]
    for lbl, gt, val in res_groups:
        err = abs(val - gt) / gt * 100 if gt else 0
        print(f"  {lbl:>20s}  {gt:10.5f}  {val:10.5f}  {err:7.2f} %")
        
    print("\n  Detaillierte Parameter-Fehler (PRO GELENK / SEHNE):")
    print(f"  {'Index':>5s} | {'Stiffness (GT/Id/Err%)':>25s} | {'Damping (GT/Id/Err%)':>25s}")
    print("  " + "-" * 60)
    for i in range(njnt):
        err_s = abs(stiff_res[i] - gt_stiffness[i]) / gt_stiffness[i] * 100 if gt_stiffness[i] else 0
        err_d = abs(damp_res[i] - gt_damping[i]) / gt_damping[i] * 100 if gt_damping[i] else 0
        # err_a = abs(arm_res[i] - gt_armature[i]) / gt_armature[i] * 100 if gt_armature[i] else 0
        
        s_str = f"{gt_stiffness[i]:.4f}/{stiff_res[i]:.4f}/{err_s:5.1f}%"
        d_str = f"{gt_damping[i]:.4f}/{damp_res[i]:.4f}/{err_d:5.1f}%"
        # a_str = f"{gt_armature[i]:.4f}/{arm_res[i]:.4f}/{err_a:5.1f}%"
        print(f"  {i:5d} | {s_str:>25s} | {d_str:>25s}")

    if ntendon > 0:
        print(f"\n  {'Index':>5s} | {'Tendon Stiff (GT/Id/Err%)':>25s}")
        print("  " + "-" * 35)
        for i in range(ntendon):
            err_ts = abs(t_stiff_res[i] - gt_tendon_stiffness[i]) / gt_tendon_stiffness[i] * 100 if gt_tendon_stiffness[i] else 0
            ts_str = f"{gt_tendon_stiffness[i]:.2f}/{t_stiff_res[i]:.2f}/{err_ts:5.1f}%"
            print(f"  {i:5d} | {ts_str:>25s}")

    print("=" * 55)

    if args.profile_timing:
        obj_calls, cost_calls, elapsed_safe = max(1, timing["objective_calls"]), max(1, timing["cost_calls"]), max(elapsed, 1e-12)
        print("\nZEITPROFILING")
        print(f"  Objective gesamt: {timing['objective_total_s']:.4f} s ({timing['objective_total_s'] / elapsed_safe * 100:.1f} %)")
        print(f"  Cost gesamt:      {timing['cost_total_s']:.4f} s ({timing['cost_total_s'] / elapsed_safe * 100:.1f} %)")
        print(f"  One-time Modellbau: {timing['opt_model_build_s']:.4f} s")
        cost_total_safe = max(timing["cost_total_s"], 1e-12)
        print("  Cost-Aufteilung:")
        print(f"    set_params:   {timing['cost_set_params_s']:.4f} s ({timing['cost_set_params_s'] / cost_total_safe * 100:.1f} %)")
        print(f"    simulate:     {timing['cost_simulate_s']:.4f} s ({timing['cost_simulate_s'] / cost_total_safe * 100:.1f} %)")
        print(f"    postprocess:  {timing['cost_postprocess_s']:.4f} s ({timing['cost_postprocess_s'] / cost_total_safe * 100:.1f} %)")

    # ── Cost convergence plot
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if len(cost_history) > 0:
            ax.semilogy(cost_history, marker='o', linestyle='-', linewidth=2, markersize=4, color='steelblue')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Best Cost (log scale)')
            ax.set_title('Optimization Convergence: Best Cost over Generations')
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.tight_layout()
            cost_plot = Path(__file__).resolve().parent / "build" / "sysid_multi_convergence.png"
            fig.savefig(cost_plot, dpi=150)
            plt.close(fig)
            print(f"  [Info] Konvergenzplot gespeichert unter: {cost_plot}")
        else:
            print(f"  [Warn] Keine Kostenhistorie vorhanden.")
            plt.close(fig)
    except Exception as exc:
        print(f"  [Warn] Konvergenzplot konnte nicht erstellt werden: {exc}")

    # ── Final validation plot: Ground truth (top) vs Identified simulation (bottom)
    try:
        opt_model = make_model()
        set_params(opt_model, stiff_res, damp_res, t_stiff_res)
        est_qpos_val, est_qvel_val = simulate(opt_model, sim_time)

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
        colors = plt.cm.tab20(np.linspace(0, 1, njnt))

        # Ground truth (top)
        for i in range(njnt):
            axs[0].plot(gt_qpos[:, i], label=f"J{i}", color=colors[i])
        axs[0].set_title("Ground Truth Trajectories")
        axs[0].set_ylabel("qpos (rad)")
        axs[0].grid(True, linestyle="--", alpha=0.5)

        # Identified / simulated (bottom)
        for i in range(njnt):
            axs[1].plot(est_qpos_val[:, i], label=f"J{i}", color=colors[i])
        axs[1].set_title("Simulated Trajectories (Identified Parameters)")
        axs[1].set_xlabel("Samples")
        axs[1].set_ylabel("qpos (rad)")
        axs[1].grid(True, linestyle="--", alpha=0.5)

        axs[1].legend(loc='center left', bbox_to_anchor=(1.0, 1.0))
        plt.tight_layout()
        out_plot = Path(__file__).resolve().parent / "build" / "sysid_multi_validation.png"
        fig.savefig(out_plot, dpi=150)
        plt.close(fig)
        print(f"\n  [Info] Validierungsplot gespeichert unter: {out_plot}")
    except Exception as exc:
        print(f"\n  [Warn] Validierungsplot konnte nicht erstellt werden: {exc}")

if __name__ == "__main__":
    main()