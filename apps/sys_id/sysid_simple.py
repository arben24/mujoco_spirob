"""
Minimale Systemidentifikation für SpiRob via MuJoCo.

Identifiziert 4 Parameter (joint_stiffness, joint_damping, tendon_stiffness,
armature) durch Vergleich einer Ground-Truth-Simulation mit einer
Kandidaten-Simulation.  Optimierung über scipy differential_evolution.

Verwendung:
  uv run apps/sysid_simple.py
  uv run apps/sysid_simple.py --sim-time 3.0 --maxiter 200
"""

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

# ── Ground-Truth-Parameter (die "wahren" Werte) ────────────────────
GT_STIFFNESS = 0.08
GT_DAMPING = 0.12
GT_TENDON_STIFFNESS = 60.0
# GT_ARMATURE = 0.015

# ── Startwerte für die Optimierung (bewusst daneben) ───────────────
INIT_STIFFNESS = 0.02
INIT_DAMPING = 0.02
INIT_TENDON_STIFFNESS = 30.0
# INIT_ARMATURE = 0.005

# ── Suchraum-Grenzen (physikalisch sinnvoll) ──────────────────────
BOUNDS_STIFFNESS = (1e-3, 1.0)
BOUNDS_DAMPING = (1e-3, 1.0)
BOUNDS_TENDON_STIFFNESS = (20.0, 120.0)
# BOUNDS_ARMATURE = (1e-3, 0.1)


# ====================================================================
#  Hilfsfunktionen
# ====================================================================

def make_model() -> mj.MjModel:
    """Erzeuge ein frisches MuJoCo-Modell aus den Spirob-Parametern."""
    xml = sg.generate_xml_string(
        L_target=L_TARGET,
        base_d=BASE_D,
        tip_d=TIP_D,
        Delta_theta_deg=DELTA_THETA_DEG,
        auto_format=True,
    )
    return mj.MjModel.from_xml_string(xml)


def set_params(model: mj.MjModel,
               stiffness: float,
               damping: float,
               tendon_stiffness: float) -> None:
    """Setzt uniforme physikalische Parameter auf alle Joints / Tendons."""
    for i in range(model.njnt):
        model.jnt_stiffness[i] = stiffness
        dof = model.jnt_dofadr[i]
        model.dof_damping[dof] = damping
        # model.dof_armature[dof] = armature
    for i in range(model.ntendon):
        model.tendon_stiffness[i] = tendon_stiffness


def controller(model: mj.MjModel, data: mj.MjData, t: float) -> None:
    """Multi-Frequenz-Anregung.  Negative ctrl → Tendon-Zug."""
    ramp = min(t / 0.5, 1.0)
    s1 = np.sin(2 * np.pi * 0.5 * t)
    s2 = np.sin(2 * np.pi * 1.5 * t)
    s3 = np.sin(2 * np.pi * 3.0 * t)
    data.ctrl[0] = -ramp * (5.0 + 3.0 * s1 + 1.5 * s2)
    data.ctrl[1] = -ramp * (8.0 + 4.0 * s1 - 2.0 * s3)


def simulate(model: mj.MjModel, sim_time: float,
             record_dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:  # record_dt: Aufzeichnungsintervall in Sekunden ANPASSEN!
    """Simulation ausführen → (qpos_array, qvel_array)."""
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    model.opt.timestep = 0.02  # 10 ms Zeitschritt  ANPASSEN!
    dt = model.opt.timestep
    record_every = max(1, round(record_dt / dt))
    total_steps = int(sim_time / dt)

    qpos_list: list[np.ndarray] = []
    qvel_list: list[np.ndarray] = []

    for step in range(total_steps + 1):
        if step % record_every == 0:
            qpos_list.append(data.qpos.copy())
            qvel_list.append(data.qvel.copy())
        controller(model, data, data.time)
        mj.mj_step(model, data)

    return np.array(qpos_list), np.array(qvel_list)


def model_to_xml_string(model: mj.MjModel,
                        tmp_path: str = "build/gt_model_after_set_params.xml") -> str:
    """Serialisiert den aktuellen Modellzustand nach XML und liefert den Inhalt."""
    path = Path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mj.mj_saveLastXML(str(path), model)
    return path.read_text(encoding="utf-8")


def parse_param_vector(text: str) -> np.ndarray:
    """Parst einen 3er-Vektor aus 'a,b,c'."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("Erwarte genau 3 Werte im Format a,b,c")
    return np.array([float(p) for p in parts], dtype=float)


def evaluate_fit_point(label: str,
                       x_norm: np.ndarray,
                       scale: np.ndarray,
                       gt_qpos: np.ndarray,
                       gt_qvel: np.ndarray,
                       sim_time: float) -> None:
    """Bewertet einen Punkt und gibt normierte + physikalische Parameter aus."""
    eval_model = make_model()
    cost = cost_function(
        x_norm,
        scale,
        gt_qpos,
        gt_qvel,
        sim_time,
        eval_model,
        timing=None,
    )
    p = x_norm * scale
    print(f"  {label:<18s} cost={cost:.6e}  "
          f"x_norm=[{x_norm[0]:.4f}, {x_norm[1]:.4f}, {x_norm[2]:.4f}]  "
          f"phys=[{p[0]:.5f}, {p[1]:.5f}, {p[2]:.3f}]")


def build_mid_point(args: argparse.Namespace,
                    x0: np.ndarray,
                    x_best: np.ndarray) -> tuple[np.ndarray, str]:
    """Ermittelt den Zwischenpunkt aus CLI-Optionen."""
    if args.mid_point_norm is not None:
        return parse_param_vector(args.mid_point_norm), "Zwischenpunkt(user)"

    alpha = float(np.clip(args.mid_alpha, 0.0, 1.0))
    mid_x = (1.0 - alpha) * x0 + alpha * x_best
    return mid_x, f"Zwischenpunkt(a={alpha:.2f})"


def play_fit_points_in_viewer(points: list[tuple[str, np.ndarray]],
                              scale: np.ndarray,
                              sim_time_per_point: float,
                              model: mj.MjModel | None = None,
                              use_gt_model: bool = False,
                              loop: bool = True) -> None:
    """Spielt Initial/Mitte/Best nacheinander in einem MuJoCo-Viewer ab.
    
    Args:
        points: Liste von (label, x_norm) Tupeln
        scale: scale-Vektor zur Parameternormalisierung
        sim_time_per_point: Anzeigedauer pro Punkt in Sekunden
        model: Optional vorgebautes Modell (nur bei use_gt_model=True)
        use_gt_model: Wenn True, wird model direkt verwendet (GT bereits gesetzt)
        loop: Wenn True, wird die Sequenz wiederholt, bis der Viewer geschlossen wird
    """
    if use_gt_model and model is None:
        raise ValueError("use_gt_model=True erfordert model-Parameter")
    
    if not use_gt_model:
        model = make_model()
    
    data = mj.MjData(model)

    with mj_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for label, x_norm in points:
                if not viewer.is_running():
                    break

                if use_gt_model:
                    # GT-Modell wird bereits mit korrekten Parametern verwendet
                    print(f"[Viewer] {label}: (Ground Truth bereits gesetzt)")
                else:
                    p = x_norm * scale
                    print(f"[Viewer] {label}: phys=[{p[0]:.5f}, {p[1]:.5f}, {p[2]:.3f}]")
                    set_params(model, p[0], p[1], p[2])
                
                mj.mj_resetData(model, data)

                phase_start = time.time()
                while viewer.is_running() and (time.time() - phase_start < sim_time_per_point):
                    step_start = time.time()
                    controller(model, data, data.time)
                    mj.mj_step(model, data)
                    viewer.sync()

                    # Rudimentaere Echtzeitsynchronisation.
                    dt_left = model.opt.timestep - (time.time() - step_start)
                    if dt_left > 0:
                        time.sleep(dt_left)
            
            if not loop:
                break

    print("Viewer-Sequenz abgeschlossen.")


# ====================================================================
#  Kostenfunktion  (das Herzstück)
# ====================================================================

def cost_function(params_norm: np.ndarray,
                  scale: np.ndarray,
                  gt_qpos: np.ndarray,
                  gt_qvel: np.ndarray,
                  sim_time: float,
                  model: mj.MjModel,
                  timing: dict[str, Any] | None = None) -> float:
    """
    Berechne den gewichteten MSE zwischen GT und Kandidat.

    Parameters
    ----------
    params_norm : (3,) normalisierte Parameter  [stiff, damp, t_stiff]
    scale       : (3,) Skalierungsfaktoren (= Startwerte)
    gt_qpos     : GT-Positionstrajektorie
    gt_qvel     : GT-Geschwindigkeitstrajektorie
    sim_time    : Simulationszeit in Sekunden

    Returns
    -------
    float  Kostenwert (kleiner = besser)
    """
    t_cost_start = time.perf_counter()
    phys = params_norm * scale
    stiff, damp, t_stiff = phys

    if timing is not None:
        timing["cost_calls"] += 1

    if stiff <= 0 or damp <= 0 or t_stiff <= 0:
        if timing is not None:
            timing["cost_invalid_param_calls"] += 1
            timing["cost_total_s"] += time.perf_counter() - t_cost_start
        return 1e6

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
    #print(f"    Simulierte {n} Datenpunkte für Kostenberechnung")
    #print(f"len() gt_qpos={len(gt_qpos)}  est_qpos={len(est_qpos)}")
    # Zeitgewichtung: spätere Samples stärker gewichten
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
    ap = argparse.ArgumentParser(description="Minimale SpiRob System-ID")
    ap.add_argument("--sim-time", type=float, default=2.0)
    ap.add_argument("--maxiter", type=int, default=50)
    ap.add_argument("--tol", type=float, default=0.2)
    ap.add_argument(
        "--report-fit-points",
        action="store_true",
        help="Zeigt Initial-Fit, Zwischenpunkt und Best-Fit am Ende an",
    )
    ap.add_argument(
        "--mid-alpha",
        type=float,
        default=0.5,
        help="Zwischenpunkt zwischen Start und Ende: x_mid=(1-a)*x0+a*x_best",
    )
    ap.add_argument(
        "--mid-point-norm",
        type=str,
        default=None,
        help="Eigener Zwischenpunkt im normierten Raum als a,b,c",
    )
    ap.add_argument(
        "--visualize-fit-points",
        action="store_true",
        help="Zeigt Initial/Mitte/Best nacheinander im MuJoCo-Viewer",
    )
    ap.add_argument(
        "--visualize-duration",
        type=float,
        default=None,
        help="Anzeigedauer pro Fit-Punkt im Viewer (Sekunden), default=sim-time",
    )
    ap.add_argument(
        "--profile-timing",
        action="store_true",
        help="Zeigt Laufzeitaufschluesselung fuer objective/cost/simulate an",
    )
    ap.add_argument(
        "--show-gt-xml",
        action="store_true",
        help=(
            "Speichert das GT-Modell nach set_params als XML unter "
            "build/gt_model_after_set_params.xml und gibt den XML-String aus"
        ),
    )
    ap.add_argument(
        "--visualize-gt",
        action="store_true",
        help="Zeigt nur die Ground-Truth-Trajektorie im Viewer und beendet das Programm",
    )
    args = ap.parse_args()

    sim_time: float = args.sim_time
    maxiter: int = args.maxiter
    tol: float = args.tol

    timing: dict[str, Any] = {
        "objective_calls": 0,
        "objective_total_s": 0.0,
        "cost_calls": 0,
        "cost_total_s": 0.0,
        "opt_model_build_s": 0.0,
        "cost_set_params_s": 0.0,
        "cost_simulate_s": 0.0,
        "cost_postprocess_s": 0.0,
        "cost_invalid_param_calls": 0,
        "cost_exception_calls": 0,
    }

    # ── 1. Ground-Truth erzeugen ────────────────────────────────────
    print("=" * 55)
    print("Ground-Truth simulieren …")
    print(f"  stiffness={GT_STIFFNESS}  damping={GT_DAMPING}"
          f"  tendon={GT_TENDON_STIFFNESS}")

    gt_model = make_model()
    njnt = gt_model.njnt
    print(f"  Modell: {njnt} Joints, {gt_model.ntendon} Tendons, "
          f"{gt_model.nu} Aktuatoren")
    set_params(gt_model, GT_STIFFNESS, GT_DAMPING,
               GT_TENDON_STIFFNESS)
    if args.show_gt_xml:
        xml_string = model_to_xml_string(gt_model)
        print("  XML-Datei gespeichert: build/gt_model_after_set_params.xml")
        print("\n--- XML nach set_params (GT) ---")
        print(xml_string)
        print("--- Ende XML ---\n")
    print()

    gt_qpos, gt_qvel = simulate(gt_model, sim_time)
    print(f"  {len(gt_qpos)} Datenpunkte  |  "
          f"qpos-Range={np.ptp(gt_qpos):.6f}  qvel-Range={np.ptp(gt_qvel):.6f}")

    # ── Visualisierung Ground Truth (optional) ─────────────────────
    if args.visualize_gt:
        print()
        print("="*55)
        print("GROUND TRUTH VISUALISIERUNG")
        print(f"  Dauer: {sim_time:.2f} s")
        print("="*55)
        play_fit_points_in_viewer(
            [("Ground Truth", np.ones(3))],  # dummy normalized vector (will be replaced with GT in viewer)
            np.array([GT_STIFFNESS, GT_DAMPING, GT_TENDON_STIFFNESS]),
            sim_time_per_point=sim_time,
            model=gt_model,  # Use GT model directly, not through parameter setting
            use_gt_model=True
        )
        print()
        print("Ground Truth Visualisierung abgeschlossen.")
        return

    # ── 2. Optimierung vorbereiten ──────────────────────────────────
    scale = np.array([INIT_STIFFNESS, INIT_DAMPING,
                      INIT_TENDON_STIFFNESS])

    # Normalisierte Bounds  (physikalisch / scale)
    norm_bounds = [
        (BOUNDS_STIFFNESS[0] / scale[0],        BOUNDS_STIFFNESS[1] / scale[0]),
        (BOUNDS_DAMPING[0] / scale[1],           BOUNDS_DAMPING[1] / scale[1]),
        (BOUNDS_TENDON_STIFFNESS[0] / scale[2],  BOUNDS_TENDON_STIFFNESS[1] / scale[2]),
        # (BOUNDS_ARMATURE[0] / scale[3],           BOUNDS_ARMATURE[1] / scale[3]),
    ]

    x0 = np.ones(3)  # Startpunkt = Startwerte (normalisiert)

    # Ein Modell fuer alle Kandidaten wiederverwenden (workers=1 vorausgesetzt).
    t_model_build = time.perf_counter()
    opt_model = make_model()
    timing["opt_model_build_s"] = time.perf_counter() - t_model_build

    iteration_counter = [0]  # mutable für Closure

    def objective(x: np.ndarray) -> float:
        t_obj_start = time.perf_counter()
        #print(f"  Evaluating candidate: {x }")
        c = cost_function(x, scale, gt_qpos, gt_qvel, sim_time, opt_model, timing=timing)
        iteration_counter[0] += 1
        timing["objective_calls"] += 1
        timing["objective_total_s"] += time.perf_counter() - t_obj_start
        it = iteration_counter[0]
        if it == 1 or it % 50 == 0:
            p = x * scale
            print(f"  [{it:5d}]  cost={c:.6e}  "
                  f"stiff={p[0]:.5f} damp={p[1]:.5f} "
                  f"t_stiff={p[2]:.2f}")
        return c

    print(f"x0 = {x0}")

    # ── 3. Optimierung laufen lassen ────────────────────────────────
    print()
    print("=" * 55)
    print("Optimierung starten  (differential_evolution)")
    print(f"  maxiter={maxiter}  tol={tol}")
    print("-" * 55)

    t0 = time.time()
    result = differential_evolution(
        objective,
        bounds=norm_bounds,
        x0=x0,
        maxiter=maxiter,
        tol=tol,
        seed=42,
        polish=True,
        init="sobol",
        popsize=15,
        mutation=(0.5, 1.5),
        recombination=0.9,
        workers=1,
        disp=False,
    )
    elapsed = time.time() - t0

    # ── 4. Ergebnis ────────────────────────────────────────────────
    phys = result.x * scale

    print()
    print("=" * 55)
    print("ERGEBNIS")
    print(f"  Status  : {result.message}")
    print(f"  Cost    : {result.fun:.10e}")
    print(f"  Aufrufe : {result.nfev}")
    print(f"  Dauer   : {elapsed:.1f} s")
    print()
    labels = ["joint_stiffness", "joint_damping", "tendon_stiffness"]
    # gt_vals = [GT_STIFFNESS, GT_DAMPING, GT_TENDON_STIFFNESS, GT_ARMATURE]
    gt_vals = [GT_STIFFNESS, GT_DAMPING, GT_TENDON_STIFFNESS]
    print(f"  {'Parameter':>20s}  {'GT':>10s}  {'Identif.':>10s}  {'Fehler':>8s}")
    print("  " + "-" * 54)
    for lbl, gt, val in zip(labels, gt_vals, phys):
        err = abs(val - gt) / gt * 100 if gt else 0
        print(f"  {lbl:>20s}  {gt:10.5f}  {val:10.5f}  {err:7.2f} %")
    print("=" * 55)

    if args.report_fit_points:
        print()
        print("FIT-VERGLEICH (3 PUNKTE)")
        print("  Initial-Fit = Startpunkt x0")
        print("  Best-Fit    = Optimierungsergebnis")
        mid_x, mid_label = build_mid_point(args, x0, result.x)

        evaluate_fit_point("Initial-Fit", x0, scale, gt_qpos, gt_qvel, sim_time)
        evaluate_fit_point(mid_label, mid_x, scale, gt_qpos, gt_qvel, sim_time)
        evaluate_fit_point("Best-Fit", result.x, scale, gt_qpos, gt_qvel, sim_time)

    if args.visualize_fit_points:
        mid_x, mid_label = build_mid_point(args, x0, result.x)
        sim_time_per_point = args.visualize_duration if args.visualize_duration is not None else sim_time
        sim_time_per_point = max(0.05, float(sim_time_per_point))
        
        print()
        print("=" * 55)
        print("MUJOCO VISUALISIERUNG 1/2: Ground Truth")
        print("-> Schließe das Fenster (X), um zur nächsten Visualisierung zu wechseln!")
        print("=" * 55)
        play_fit_points_in_viewer(
            [("Ground Truth", np.ones(3))],
            np.array([GT_STIFFNESS, GT_DAMPING, GT_TENDON_STIFFNESS]),
            sim_time_per_point=sim_time_per_point,
            model=gt_model,
            use_gt_model=True,
            loop=True
        )

        print()
        print("=" * 55)
        print("MUJOCO VISUALISIERUNG 2/2: Initial -> Mitte -> Best")
        print("-> Schließe das Fenster (X), um das Programm zu beenden!")
        print("=" * 55)
        play_fit_points_in_viewer(
            [
                ("Initial-Fit", x0),
                (mid_label, mid_x),
                ("Best-Fit", result.x),
            ],
            scale,
            sim_time_per_point,
            loop=True
        )

    if args.profile_timing:
        obj_calls = max(1, timing["objective_calls"])
        cost_calls = max(1, timing["cost_calls"])
        elapsed_safe = max(elapsed, 1e-12)
        print()
        print("ZEITPROFILING")
        print(f"  Objective gesamt: {timing['objective_total_s']:.4f} s  "
            f"({timing['objective_total_s'] / elapsed_safe * 100:.1f} % von Gesamt)")
        print(f"    pro Call: {timing['objective_total_s'] / obj_calls * 1000:.3f} ms")
        print(f"  Cost gesamt:      {timing['cost_total_s']:.4f} s  "
            f"({timing['cost_total_s'] / elapsed_safe * 100:.1f} % von Gesamt)")
        print(f"    pro Call: {timing['cost_total_s'] / cost_calls * 1000:.3f} ms")
        print(f"  One-time Modellbau: {timing['opt_model_build_s']:.4f} s")

        cost_total_safe = max(timing["cost_total_s"], 1e-12)
        print("  Cost-Aufteilung:")
        print(f"    set_params:   {timing['cost_set_params_s']:.4f} s  "
            f"({timing['cost_set_params_s'] / cost_total_safe * 100:.1f} %)" )
        print(f"    simulate:     {timing['cost_simulate_s']:.4f} s  "
            f"({timing['cost_simulate_s'] / cost_total_safe * 100:.1f} %)" )
        print(f"    postprocess:  {timing['cost_postprocess_s']:.4f} s  "
            f"({timing['cost_postprocess_s'] / cost_total_safe * 100:.1f} %)" )
        print(f"  Sonderfaelle: invalid={timing['cost_invalid_param_calls']}  "
            f"exceptions={timing['cost_exception_calls']}")


if __name__ == "__main__":
    main()
