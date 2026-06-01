"""
SpiRob System Identification via MuJoCo Simulation
===================================================

Identifiziert die physikalischen Parameter (Joint-Damping, Joint-Stiffness,
Tendon-Stiffness, Armature) eines Tendon-gesteuerten Spirob-Modells durch
iterative Optimierung gegen eine Ground-Truth-Simulation.

Ablauf:
  1. Ground-Truth-Modell mit bekannten Parametern simulieren → Referenztrajektorie
  2. Identifikationsmodell mit Schätzwerten simulieren → Vergleichstrajektorie
  3. MSE-Fehler minimieren via scipy.optimize (Nelder-Mead oder L-BFGS-B)
  4. Konvergenz-Plots und finale Parameterausgabe

Verwendung:
  uv run apps/spirob_sysid.py                           # Standard (uniform, Nelder-Mead)
  uv run apps/spirob_sysid.py --mode per-joint           # Pro-Joint-Optimierung
  uv run apps/spirob_sysid.py --method L-BFGS-B          # Anderer Solver
  uv run apps/spirob_sysid.py --sim-time 3.0 --maxiter 200
  uv run apps/spirob_sysid.py --visualize                # Zeige MuJoCo-Viewer nach Konvergenz
  uv run apps/spirob_sysid.py --save-params build/sysid_params.json
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from scipy.optimize import minimize, differential_evolution

import math_spirob.spirob_generator as sg

# ─────────────────────────────────────────────────────────────────────
# 1. Datenklassen & Konfiguration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SysIdConfig:
    """Konfiguration für die System-Identifikation."""
    # Spiral-Modell-Parameter
    L_target: float = 0.44 
    base_d: float = 0.1
    tip_d: float = 0.03
    Delta_theta_deg: float = 30.0

    # Simulation
    sim_time: float = 5.0
    record_dt: float = 0.01  # Aufzeichnungsintervall (nicht jeder Timestep)

    # Ground-Truth-Parameter (die "wahren" Werte, die wir identifizieren wollen)
    gt_joint_stiffness: float = 0.08
    gt_joint_damping: float = 0.12
    gt_tendon_stiffness: float = 60.0
    gt_armature: float = 0.015

    # Optionale per-Joint-Werte für Ground-Truth (überschreiben uniforme Werte wenn gesetzt)
    # Länge muss njnt entsprechen; partielle Angabe möglich (None = uniforme Fallback-Werte)
    gt_joint_stiffness_vec: Optional[List[float]] = None
    gt_joint_damping_vec: Optional[List[float]] = None

    # Startwerte für die Optimierung (bewusst falsch gewählt)
    init_joint_stiffness: float = 0.02
    init_joint_damping: float = 0.02
    init_tendon_stiffness: float = 30.0
    init_armature: float = 0.005

    # Optimierung
    mode: str = "uniform"  # "uniform" oder "per-joint"
    method: str = "differential_evolution"  # "differential_evolution", "Nelder-Mead", "Powell", "L-BFGS-B"
    maxiter: int = 150
    tol: float = 1e-8

    # Parametergrenzen (für L-BFGS-B u.a.)
    bounds_stiffness: Tuple[float, float] = (1e-4, 1.0)
    bounds_damping: Tuple[float, float] = (1e-4, 1.0)
    bounds_tendon_stiffness: Tuple[float, float] = (1.0, 200.0)
    bounds_armature: Tuple[float, float] = (1e-4, 0.1)


@dataclass
class OptimizationLog:
    """Speichert den Verlauf der Optimierung."""
    iterations: List[int] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    params_history: List[np.ndarray] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)


@dataclass
class TrajectoryData:
    """Container für aufgezeichnete Simulationstrajektorien."""
    time: np.ndarray
    qpos: np.ndarray  # shape: (n_steps, nq)
    qvel: np.ndarray  # shape: (n_steps, nv)


# ─────────────────────────────────────────────────────────────────────
# 2. Modell-Erzeugung & Parameterzugriff
# ─────────────────────────────────────────────────────────────────────

def generate_base_model(config: SysIdConfig) -> mj.MjModel:
    """Erzeugt das Basis-MuJoCo-Modell aus den Spirob-Parametern."""
    xml_string = sg.generate_xml_string(
        L_target=config.L_target,
        base_d=config.base_d,
        tip_d=config.tip_d,
        Delta_theta_deg=config.Delta_theta_deg,
        model_name="spirob_sysid",
        auto_format=True,
    )
    return mj.MjModel.from_xml_string(xml_string)


def set_uniform_params(
    model: mj.MjModel,
    joint_stiffness: float,
    joint_damping: float,
    tendon_stiffness: float,
    armature: float,
) -> None:
    """Setzt identische Parameter für alle Joints/Tendons."""
    for i in range(model.njnt):
        model.jnt_stiffness[i] = joint_stiffness
        dof_id = model.jnt_dofadr[i]
        model.dof_damping[dof_id] = joint_damping
        model.dof_armature[dof_id] = armature
    for i in range(model.ntendon):
        model.tendon_stiffness[i] = tendon_stiffness


def set_per_joint_params(
    model: mj.MjModel,
    stiffness_vec: np.ndarray,
    damping_vec: np.ndarray,
    tendon_stiffness: float,
    armature: float,
) -> None:
    """Setzt individuelle Stiffness/Damping pro Joint."""
    n = model.njnt
    for i in range(n):
        model.jnt_stiffness[i] = stiffness_vec[i]
        dof_id = model.jnt_dofadr[i]
        model.dof_damping[dof_id] = damping_vec[i]
        model.dof_armature[dof_id] = armature
    for i in range(model.ntendon):
        model.tendon_stiffness[i] = tendon_stiffness


def get_current_params(model: mj.MjModel) -> Dict[str, np.ndarray]:
    """Liest aktuelle Parameter aus dem Modell."""
    stiffness = np.array([model.jnt_stiffness[i] for i in range(model.njnt)])
    damping = np.array([model.dof_damping[model.jnt_dofadr[i]] for i in range(model.njnt)])
    armature = np.array([model.dof_armature[model.jnt_dofadr[i]] for i in range(model.njnt)])
    tendon_stiffness = np.array([model.tendon_stiffness[i] for i in range(model.ntendon)])
    return {
        "joint_stiffness": stiffness,
        "joint_damping": damping,
        "armature": armature,
        "tendon_stiffness": tendon_stiffness,
    }


# ─────────────────────────────────────────────────────────────────────
# 3. Controller für die Anregung des Systems
# ─────────────────────────────────────────────────────────────────────

def sysid_controller(model: mj.MjModel, data: mj.MjData, t: float, step: int) -> None:
    """
    Multi-Frequenz-Controller zur Systemanregung.
    Kombiniert verschiedene Frequenzen für bessere Anregung aller Moden.

    WICHTIG: Tendon-Aktuatoren haben ctrlrange=[-50, 0] (nur Zug, negative Werte).
    Positive Werte werden auf 0 geclippt und erzeugen keine Bewegung.
    """
    ramp = min(t / 0.5, 1.0)  # Linearer Anstieg über 0.5s
    f1 = np.sin(2 * np.pi * 0.5 * t)    # 0.5 Hz
    f2 = np.sin(2 * np.pi * 1.5 * t)    # 1.5 Hz
    f3 = np.sin(2 * np.pi * 3.0 * t)    # 3.0 Hz

    # Tendon 0: Rampe + Multi-Frequenz-Sinus (negativ = Zug)
    data.ctrl[0] = -ramp * (5.0 + 3.0 * f1 + 1.5 * f2)
    # Tendon 1: Gegenläufige Anregung (negativ = Zug)
    data.ctrl[1] = -ramp * (8.0 + 4.0 * f1 - 2.0 * f3)


# ─────────────────────────────────────────────────────────────────────
# 4. Simulation & Datenaufzeichnung
# ─────────────────────────────────────────────────────────────────────

def run_simulation(
    model: mj.MjModel,
    controller: Callable,
    sim_time: float,
    record_dt: float = 0.01,
) -> TrajectoryData:
    """
    Führt eine MuJoCo-Simulation aus und zeichnet Trajektorien auf.

    Parameters
    ----------
    model : MjModel
        Das (bereits parametrisierte) MuJoCo-Modell.
    controller : Callable
        Controller-Funktion (ControllerFunc-Signatur).
    sim_time : float
        Simulationszeit in Sekunden.
    record_dt : float
        Zeitintervall zwischen Aufzeichnungspunkten.

    Returns
    -------
    TrajectoryData
        Aufgezeichnete Zeitreihen (time, qpos, qvel).
    """
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    dt = model.opt.timestep
    record_every = max(1, int(round(record_dt / dt)))
    total_steps = int(sim_time / dt)
    n_records = total_steps // record_every + 1

    time_arr = np.zeros(n_records)
    qpos_arr = np.zeros((n_records, model.nq))
    qvel_arr = np.zeros((n_records, model.nv))

    rec_idx = 0
    for step in range(total_steps + 1):
        if step % record_every == 0 and rec_idx < n_records:
            time_arr[rec_idx] = data.time
            qpos_arr[rec_idx] = data.qpos.copy()
            qvel_arr[rec_idx] = data.qvel.copy()
            rec_idx += 1

        controller(model, data, data.time, step)
        mj.mj_step(model, data)

    # Auf tatsächliche Anzahl trimmen
    time_arr = time_arr[:rec_idx]
    qpos_arr = qpos_arr[:rec_idx]
    qvel_arr = qvel_arr[:rec_idx]

    return TrajectoryData(time=time_arr, qpos=qpos_arr, qvel=qvel_arr)


# ─────────────────────────────────────────────────────────────────────
# 5. Fehlermetrik
# ─────────────────────────────────────────────────────────────────────

def compute_trajectory_error(
    gt: TrajectoryData,
    est: TrajectoryData,
    weight_qpos: float = 1.0,
    weight_qvel: float = 0.5,
) -> float:
    """
    Berechnet den gewichteten MSE zwischen Ground-Truth und geschätzter Trajektorie.

    Verwendet:
    - qpos MSE (Positionsgenauigkeit)
    - qvel MSE (Geschwindigkeitsdynamik - wichtig für Damping-Identifikation)
    - Zeitgewichtung: spätere Zeitpunkte stärker gewichten (mehr Abweichung sichtbar)
    """
    n = min(len(gt.time), len(est.time))

    # Zeitgewichtung: linear ansteigend von 0.5 bis 1.5
    time_weights = np.linspace(0.5, 1.5, n).reshape(-1, 1)

    # Gewichteter MSE für Position und Geschwindigkeit
    mse_qpos = np.mean(time_weights * (gt.qpos[:n] - est.qpos[:n]) ** 2)
    mse_qvel = np.mean(time_weights * (gt.qvel[:n] - est.qvel[:n]) ** 2)

    return weight_qpos * mse_qpos + weight_qvel * mse_qvel


# ─────────────────────────────────────────────────────────────────────
# 6. Optimierungskern
# ─────────────────────────────────────────────────────────────────────

class SystemIdentifier:
    """
    Orchestriert die iterative Parameteridentifikation.

    Ablauf:
    1. Ground-Truth erzeugen und simulieren
    2. Kostenfunktion definieren (simuliert mit aktuellen Parametern → MSE)
    3. scipy.optimize aufrufen
    4. Ergebnisse loggen und visualisieren
    """

    def __init__(self, config: SysIdConfig):
        self.config = config
        self.log = OptimizationLog()
        self._iteration = 0

        # Basismodell erzeugen (wird für GT und Identifikation kopiert)
        self.base_model = generate_base_model(config)
        self.njnt = self.base_model.njnt
        print(f"Modell erzeugt: {self.njnt} Joints, "
              f"{self.base_model.ntendon} Tendons, "
              f"{self.base_model.nu} Aktuatoren")

        # Ground-Truth-Trajektorie erzeugen
        self.gt_traj = self._generate_ground_truth()

    def _generate_ground_truth(self) -> TrajectoryData:
        """Erzeugt die Referenztrajektorie mit den wahren Parametern."""
        print("\n" + "=" * 60)
        print("GROUND-TRUTH SIMULATION")
        if self.config.gt_joint_stiffness_vec is not None:
            print(f"  joint_stiffness = [per-joint, {len(self.config.gt_joint_stiffness_vec)} Werte]")
        else:
            print(f"  joint_stiffness = {self.config.gt_joint_stiffness} (uniform)")
        if self.config.gt_joint_damping_vec is not None:
            print(f"  joint_damping   = [per-joint, {len(self.config.gt_joint_damping_vec)} Werte]")
        else:
            print(f"  joint_damping   = {self.config.gt_joint_damping} (uniform)")
        print(f"  tendon_stiffness= {self.config.gt_tendon_stiffness}")
        print(f"  armature        = {self.config.gt_armature}")
        print(f"  sim_time        = {self.config.sim_time}s")

        gt_model = generate_base_model(self.config)

        use_per_joint = (
            self.config.gt_joint_stiffness_vec is not None
            or self.config.gt_joint_damping_vec is not None
        )

        # Effektive GT-Vektoren immer als vollständige Arrays berechnen
        stiffness_vec = (
            np.array(self.config.gt_joint_stiffness_vec)
            if self.config.gt_joint_stiffness_vec is not None
            else np.full(self.njnt, self.config.gt_joint_stiffness)
        )
        damping_vec = (
            np.array(self.config.gt_joint_damping_vec)
            if self.config.gt_joint_damping_vec is not None
            else np.full(self.njnt, self.config.gt_joint_damping)
        )
        if use_per_joint and (len(stiffness_vec) != self.njnt or len(damping_vec) != self.njnt):
            raise ValueError(
                f"Per-Joint-GT-Vektoren müssen Länge {self.njnt} haben, "
                f"aber stiffness={len(stiffness_vec)}, damping={len(damping_vec)}."
            )

        # GT-Vektoren als Instanzvariablen speichern (für Vergleiche in Kostenfunktionen)
        self._gt_stiffness_vec = stiffness_vec.copy()
        self._gt_damping_vec = damping_vec.copy()

        # Pro-Joint-Tabelle ausgeben
        label_s = "per-joint" if self.config.gt_joint_stiffness_vec is not None else "uniform"
        label_d = "per-joint" if self.config.gt_joint_damping_vec is not None else "uniform"
        print()
        print(f"  {'J':>3} | {'Stiffness':>10} ({label_s}) | {'Damping':>10} ({label_d})")
        print("  " + "-"*3 + "-+-" + "-"*22 + "-+-" + "-"*21)
        for _ji in range(self.njnt):
            print(f"  {_ji:>3} | {stiffness_vec[_ji]:>10.5f}           | {damping_vec[_ji]:>10.5f}")
        print(f"  tendon_stiffness : {self.config.gt_tendon_stiffness}")
        print(f"  armature         : {self.config.gt_armature}")

        if use_per_joint:
            set_per_joint_params(
                gt_model,
                stiffness_vec=stiffness_vec,
                damping_vec=damping_vec,
                tendon_stiffness=self.config.gt_tendon_stiffness,
                armature=self.config.gt_armature,
            )
        else:
            set_uniform_params(
                gt_model,
                joint_stiffness=self.config.gt_joint_stiffness,
                joint_damping=self.config.gt_joint_damping,
                tendon_stiffness=self.config.gt_tendon_stiffness,
                armature=self.config.gt_armature,
            )

        traj = run_simulation(
            gt_model, sysid_controller,
            sim_time=self.config.sim_time,
            record_dt=self.config.record_dt,
        )

        # Prüfe ob GT-Daten sinnvoll sind (nicht alle Null)
        qpos_range = np.ptp(traj.qpos)
        qvel_range = np.ptp(traj.qvel)
        print(f"  Aufgezeichnet: {len(traj.time)} Datenpunkte")
        print(f"  qpos Bereich:  {qpos_range:.6f}")
        print(f"  qvel Bereich:  {qvel_range:.6f}")

        if qpos_range < 1e-10:
            print("  WARNUNG: Ground-Truth zeigt kaum Bewegung! "
                  "Controller-Parameter oder Simulationszeit prüfen.")

        return traj

    def _cost_function_uniform(self, params: np.ndarray) -> float:
        """Kostenfunktion für uniforme Parameter [stiffness, damping, tendon_stiffness, armature].

        Parameter werden in normalisierter Form empfangen (skaliert relativ zu
        den Startwerten) und intern zurückskaliert.
        """
        # Rückskalierung: params * scale = physikalische Werte
        phys = params * self._param_scale
        stiffness, damping, tendon_stiffness, armature = phys

        # Physikalische Plausibilität prüfen
        if stiffness <= 0 or damping <= 0 or tendon_stiffness <= 0 or armature <= 0:
            return 1e6

        try:
            model = generate_base_model(self.config)
            set_uniform_params(model, stiffness, damping, tendon_stiffness, armature)

            est_traj = run_simulation(
                model, sysid_controller,
                sim_time=self.config.sim_time,
                record_dt=self.config.record_dt,
            )

            cost = compute_trajectory_error(self.gt_traj, est_traj)

        except Exception as e:
            # Instabile Simulation → hohe Kosten
            print(f"  [!] Simulation instabil bei Iteration {self._iteration}: {e}")
            cost = 1e6

        # Loggen (physikalische Werte anzeigen)
        self._iteration += 1
        self.log.iterations.append(self._iteration)
        self.log.costs.append(cost)
        self.log.params_history.append(phys.copy())  # Speichere physikalische Werte

        if self._iteration % 10 == 0 or self._iteration == 1:
            print(f"  Iter {self._iteration:4d} | Cost: {cost:.8e} | "
                  f"stiff={stiffness:.5f} damp={damping:.5f} "
                  f"t_stiff={tendon_stiffness:.2f} arm={armature:.5f}")

        return cost

    def _cost_function_per_joint(self, params: np.ndarray) -> float:
        """Kostenfunktion für pro-Joint-Parameter (normalisiert)."""
        phys = params * self._param_scale
        n = self.njnt
        stiffness_vec = phys[:n]
        damping_vec = phys[n:2*n]
        tendon_stiffness = phys[2*n]
        armature = phys[2*n + 1]

        # Plausibilität
        if np.any(stiffness_vec <= 0) or np.any(damping_vec <= 0):
            return 1e6
        if tendon_stiffness <= 0 or armature <= 0:
            return 1e6

        try:
            model = generate_base_model(self.config)
            set_per_joint_params(model, stiffness_vec, damping_vec,
                                 tendon_stiffness, armature)

            est_traj = run_simulation(
                model, sysid_controller,
                sim_time=self.config.sim_time,
                record_dt=self.config.record_dt,
            )

            cost = compute_trajectory_error(self.gt_traj, est_traj)

        except Exception as e:
            print(f"  [!] Simulation instabil bei Iteration {self._iteration}: {e}")
            cost = 1e6

        self._iteration += 1
        self.log.iterations.append(self._iteration)
        self.log.costs.append(cost)
        self.log.params_history.append(phys.copy())  # Speichere physikalische Werte

        if self._iteration % 100 == 0 or self._iteration == 1:
            print(f"\n  Iter {self._iteration:4d} | Cost: {cost:.8e}")
            self._print_joint_status_table(stiffness_vec, damping_vec,
                                           tendon_stiffness, armature)

        return cost

    def _print_joint_status_table(
        self,
        stiffness_vec: np.ndarray,
        damping_vec: np.ndarray,
        tendon_stiffness: float,
        armature: float,
    ) -> None:
        """Gibt pro-Joint-Parameter tabellarisch aus, mit GT-Vergleich wenn verfügbar."""
        has_gt = hasattr(self, "_gt_stiffness_vec")
        if has_gt:
            print(f"  {'J':>3} | {'stiff_GT':>9} | {'stiff_ID':>9} | {'Err%':>6}"
                  f" | {'damp_GT':>9} | {'damp_ID':>9} | {'Err%':>6}")
            print("  " + "-"*3 + "-+-" + "-"*9 + "-+-" + "-"*9 + "-+-" + "-"*6
                  + "-+-" + "-"*9 + "-+-" + "-"*9 + "-+-" + "-"*6)
            for _ji in range(len(stiffness_vec)):
                s_gt = self._gt_stiffness_vec[_ji]
                d_gt = self._gt_damping_vec[_ji]
                s_id = stiffness_vec[_ji]
                d_id = damping_vec[_ji]
                s_err = abs(s_id - s_gt) / s_gt * 100 if s_gt != 0 else 0.0
                d_err = abs(d_id - d_gt) / d_gt * 100 if d_gt != 0 else 0.0
                print(f"  {_ji:>3} | {s_gt:>9.5f} | {s_id:>9.5f} | {s_err:>5.1f}%"
                      f" | {d_gt:>9.5f} | {d_id:>9.5f} | {d_err:>5.1f}%")
        else:
            print(f"  {'J':>3} | {'stiffness':>10} | {'damping':>10}")
            print("  " + "-"*3 + "-+-" + "-"*10 + "-+-" + "-"*10)
            for _ji in range(len(stiffness_vec)):
                print(f"  {_ji:>3} | {stiffness_vec[_ji]:>10.5f} | {damping_vec[_ji]:>10.5f}")
        t_gt = self.config.gt_tendon_stiffness
        a_gt = self.config.gt_armature
        if has_gt:
            t_err = abs(tendon_stiffness - t_gt) / t_gt * 100 if t_gt != 0 else 0.0
            a_err = abs(armature - a_gt) / a_gt * 100 if a_gt != 0 else 0.0
            print(f"  tendon_stiffness: {tendon_stiffness:>9.4f}  (GT: {t_gt:.4f}, Err: {t_err:.1f}%)")
            print(f"  armature:         {armature:>9.5f}  (GT: {a_gt:.5f}, Err: {a_err:.1f}%)")
        else:
            print(f"  tendon_stiffness: {tendon_stiffness:.5f}")
            print(f"  armature:         {armature:.5f}")

    def _diagnose_termination(self, result, n_params: int) -> None:
        """Analysiert Abbruchgrund und gibt konkrete Empfehlungen zur Konfiguration."""
        cfg = self.config
        method = cfg.method
        converged = result.success

        print()
        print("─" * 60)
        print("ABBRUCH-DIAGNOSE")
        print(f"  scipy-Meldung : {result.message}")
        print(f"  Konvergiert   : {'✓ ja' if converged else '✗ nein'}")
        print(f"  Fkt-Aufrufe   : {result.nfev}")

        if method == "differential_evolution":
            popsize_factor = 10 if cfg.mode == "uniform" else 5
            pop = popsize_factor * n_params
            print(f"\n  [differential_evolution]")
            print(f"  Populationsgröße      : {pop}  ({popsize_factor} × {n_params} Params)")
            print(f"  Abbruchkriterium      : std(Kosten_Population) / |mean| < tol={cfg.tol:.0e}")
            print(f"  Alternativ Abbruch    : Generationen ≥ maxiter={cfg.maxiter}")
            print(f"  Max Fkt-Aufrufe       : ~{cfg.maxiter * pop}  "
                  f"({cfg.maxiter} Gen × {pop} Individuen)")
            if not converged:
                print(f"\n  ⚠  ABBRUCH DURCH MAXITER — tol={cfg.tol:.0e} noch nicht erfüllt")
                print(f"  Die Population ist noch nicht eng genug um das Minimum konvergiert.")
                print(f"\n  EMPFEHLUNGEN:")
                print(f"    A) Mehr Generationen  →  --maxiter {cfg.maxiter * 5}")
                print(f"    B) Toleranz lockern   →  --tol 1e-5  "
                      f"(aktuell {cfg.tol:.0e}; früher abbrechen wenn cost gut genug)")
                if n_params > 10:
                    print(f"    C) Dimensionen senken →  --mode uniform  "
                          f"({n_params} → 4 Parameter; deutlich schneller)")
                print(f"    D) Ergebnis nutzen    →  polish=True hat lokale Feinoptimierung "
                      f"durchgeführt, aktueller cost={result.fun:.3e}")
            else:
                print(f"\n  ✓  Konvergenzkriterium erfüllt: std/|mean| < {cfg.tol:.0e}")

        elif method in ("Nelder-Mead", "Powell"):
            print(f"\n  [{method}]")
            print(f"  Abbruchkriterium : |Δx| < xatol={cfg.tol:.0e}  UND  "
                  f"|Δf| < fatol={cfg.tol:.0e}")
            print(f"  Alternativ       : Iters ≥ maxiter={cfg.maxiter}")
            if not converged:
                print(f"\n  ⚠  ABBRUCH DURCH MAXITER")
                print(f"  EMPFEHLUNGEN:")
                print(f"    A) --maxiter {cfg.maxiter * 5}")
                print(f"    B) --tol 1e-5")
                print(f"    C) --method differential_evolution  (globaler Suchraum)")

        elif method == "L-BFGS-B":
            print(f"\n  [L-BFGS-B]  (lokaler Gradient-Optimizer)")
            print(f"  Abbruchkriterium : |Δf|/max(1,|f|) < ftol  UND  "
                  f"|∇f|∞ < gtol={cfg.tol:.0e}")
            if not converged:
                print(f"\n  ⚠  Nicht konvergiert → --maxiter {cfg.maxiter * 5} "
                      f"oder --method differential_evolution")

        # Kosten-Verlauf-Analyse
        if len(self.log.costs) > 20:
            costs_arr = np.array(self.log.costs)
            valid = costs_arr[costs_arr < 1e5]
            if len(valid) > 10:
                recent_n = max(10, len(valid) // 10)
                recent = valid[-recent_n:]
                total_imp = (costs_arr[0] - valid.min()) / (costs_arr[0] + 1e-15) * 100
                last_imp = (recent[0] - recent[-1]) / (recent[0] + 1e-15) * 100
                print(f"\n  KOSTENENTWICKLUNG:")
                print(f"    Start       : {costs_arr[0]:.4e}")
                print(f"    Minimum     : {valid.min():.4e}")
                print(f"    Ende        : {costs_arr[-1]:.4e}")
                print(f"    Gesamt-Verbesserung        : {total_imp:.1f}%")
                trend = ("→ kaum Fortschritt, nahe Konvergenz"
                         if abs(last_imp) < 0.5
                         else "→ noch aktiv, mehr Iters könnten helfen")
                print(f"    Letzte {recent_n:3d} Iters (letzte 10%) : "
                      f"{last_imp:.2f}% Verbesserung  {trend}")
        print("─" * 60)

    def run_optimization(self) -> dict:
        """
        Führt die Optimierung aus.

        Returns
        -------
        dict
            Ergebnis mit identifizierten Parametern und Metadaten.
        """
        cfg = self.config
        self._iteration = 0

        print("\n" + "=" * 60)
        print("OPTIMIERUNG STARTEN")
        print(f"  Modus:    {cfg.mode}")
        print(f"  Methode:  {cfg.method}")
        print(f"  MaxIter:  {cfg.maxiter}")

        t_start = time.time()

        if cfg.mode == "uniform":
            # Skalierungsfaktoren = Startwerte (Optimizer arbeitet in normalisierten Einheiten)
            self._param_scale = np.array([
                cfg.init_joint_stiffness,
                cfg.init_joint_damping,
                cfg.init_tendon_stiffness,
                cfg.init_armature,
            ])
            # Normalisierter Startpunkt = [1, 1, 1, 1]
            x0 = np.ones(4)

            self.log.param_names = [
                "joint_stiffness", "joint_damping",
                "tendon_stiffness", "armature",
            ]

            # Normalisierte Bounds (immer benötigt für DE)
            norm_bounds = [
                (cfg.bounds_stiffness[0] / self._param_scale[0],
                 cfg.bounds_stiffness[1] / self._param_scale[0]),
                (cfg.bounds_damping[0] / self._param_scale[1],
                 cfg.bounds_damping[1] / self._param_scale[1]),
                (cfg.bounds_tendon_stiffness[0] / self._param_scale[2],
                 cfg.bounds_tendon_stiffness[1] / self._param_scale[2]),
                (cfg.bounds_armature[0] / self._param_scale[3],
                 cfg.bounds_armature[1] / self._param_scale[3]),
            ]

            phys_x0 = x0 * self._param_scale
            print(f"  Startwerte (phys):  {phys_x0}")
            print(f"  Zielwerte:          [{cfg.gt_joint_stiffness}, {cfg.gt_joint_damping}, "
                  f"{cfg.gt_tendon_stiffness}, {cfg.gt_armature}]")
            print("-" * 60)

            if cfg.method == "differential_evolution":
                result = differential_evolution(
                    self._cost_function_uniform,
                    bounds=norm_bounds,
                    x0=x0,
                    maxiter=cfg.maxiter,
                    tol=cfg.tol,
                    seed=42,
                    disp=False,
                    polish=True,        # Lokale Nachoptimierung am Ende
                    init="sobol",       # Quasi-Random Initialisierung
                    popsize=10,         # Populationsgröße (10 * n_params)
                    mutation=(0.5, 1.5),
                    recombination=0.9,
                    workers=1,          # MuJoCo ist nicht thread-safe
                )
            else:
                result = minimize(
                    self._cost_function_uniform,
                    x0,
                    method=cfg.method,
                    bounds=norm_bounds if cfg.method in ("L-BFGS-B", "TNC", "SLSQP") else None,
                    options={"maxiter": cfg.maxiter, "xatol": cfg.tol, "fatol": cfg.tol,
                             "disp": False, "adaptive": True},
                )

            phys_result = result.x * self._param_scale
            identified = {
                "joint_stiffness": float(phys_result[0]),
                "joint_damping": float(phys_result[1]),
                "tendon_stiffness": float(phys_result[2]),
                "armature": float(phys_result[3]),
            }

        elif cfg.mode == "per-joint":
            n = self.njnt
            # Skalierung: pro-Joint-Werte + tendon + armature
            self._param_scale = np.concatenate([
                np.full(n, cfg.init_joint_stiffness),
                np.full(n, cfg.init_joint_damping),
                [cfg.init_tendon_stiffness, cfg.init_armature],
            ])
            x0 = np.ones(2 * n + 2)

            self.log.param_names = (
                [f"stiffness_j{i}" for i in range(n)]
                + [f"damping_j{i}" for i in range(n)]
                + ["tendon_stiffness", "armature"]
            )

            # Normalisierte Bounds
            norm_bounds = (
                [(cfg.bounds_stiffness[0] / cfg.init_joint_stiffness,
                  cfg.bounds_stiffness[1] / cfg.init_joint_stiffness)] * n
                + [(cfg.bounds_damping[0] / cfg.init_joint_damping,
                    cfg.bounds_damping[1] / cfg.init_joint_damping)] * n
                + [(cfg.bounds_tendon_stiffness[0] / cfg.init_tendon_stiffness,
                    cfg.bounds_tendon_stiffness[1] / cfg.init_tendon_stiffness),
                   (cfg.bounds_armature[0] / cfg.init_armature,
                    cfg.bounds_armature[1] / cfg.init_armature)]
            )

            print(f"  Parameter-Anzahl: {len(x0)}")
            # Initialtabelle: Startwerte vs GT
            _start_stiff = np.full(n, cfg.init_joint_stiffness)
            _start_damp  = np.full(n, cfg.init_joint_damping)
            print("\n  INITIALSCHÄTZUNG pro Joint:")
            self._print_joint_status_table(
                _start_stiff, _start_damp,
                cfg.init_tendon_stiffness, cfg.init_armature,
            )
            print("-" * 60)

            if cfg.method == "differential_evolution":
                result = differential_evolution(
                    self._cost_function_per_joint,
                    bounds=norm_bounds,
                    x0=x0,
                    maxiter=cfg.maxiter,
                    tol=cfg.tol,
                    seed=42,
                    disp=False,
                    polish=True,
                    init="sobol",
                    popsize=5,  # Kleiner wegen hoher Dimension
                    mutation=(0.5, 1.5),
                    recombination=0.9,
                    workers=1,
                )
            else:
                result = minimize(
                    self._cost_function_per_joint,
                    x0,
                    method=cfg.method,
                    bounds=norm_bounds if cfg.method in ("L-BFGS-B", "TNC", "SLSQP") else None,
                    options={"maxiter": cfg.maxiter, "disp": False, "adaptive": True},
                )

            phys_result = result.x * self._param_scale
            identified = {
                "stiffness_per_joint": phys_result[:n].tolist(),
                "damping_per_joint": phys_result[n:2*n].tolist(),
                "tendon_stiffness": float(phys_result[2*n]),
                "armature": float(phys_result[2*n + 1]),
            }
        else:
            raise ValueError(f"Unbekannter Modus: {cfg.mode}")

        elapsed = time.time() - t_start

        print("\n" + "=" * 60)
        print("OPTIMIERUNG ABGESCHLOSSEN")
        print(f"  Status:     {result.message}")
        print(f"  Iterationen (Funktionsaufrufe): {result.nfev}")
        print(f"  Finaler Cost: {result.fun:.10e}")
        print(f"  Dauer:      {elapsed:.1f}s")
        print()
        print("IDENTIFIZIERTE PARAMETER:")
        for k, v in identified.items():
            if isinstance(v, list):
                print(f"  {k}: mean={np.mean(v):.6f}, std={np.std(v):.6f}")
            else:
                print(f"  {k}: {v:.6f}")

        # Per-Joint-Ergebnistabelle mit GT-Vergleich
        if cfg.mode == "per-joint":
            print()
            print("VERGLEICH PRO JOINT (Identifiziert vs. Ground-Truth):")
            self._print_joint_status_table(
                np.array(identified.get("stiffness_per_joint", [])),
                np.array(identified.get("damping_per_joint", [])),
                identified["tendon_stiffness"],
                identified["armature"],
            )

        if cfg.mode == "uniform":
            print()
            print("VERGLEICH (Identifiziert vs. Ground-Truth):")
            # Effektive GT-Werte: per-joint-Mittelwert wenn vorhanden, sonst uniformer Wert
            gt_stiff = (
                float(np.mean(cfg.gt_joint_stiffness_vec))
                if cfg.gt_joint_stiffness_vec is not None
                else cfg.gt_joint_stiffness
            )
            gt_damp = (
                float(np.mean(cfg.gt_joint_damping_vec))
                if cfg.gt_joint_damping_vec is not None
                else cfg.gt_joint_damping
            )
            gt_vals = {
                "joint_stiffness": gt_stiff,
                "joint_damping": gt_damp,
                "tendon_stiffness": cfg.gt_tendon_stiffness,
                "armature": cfg.gt_armature,
            }
            has_per_joint = (
                cfg.gt_joint_stiffness_vec is not None
                or cfg.gt_joint_damping_vec is not None
            )
            if has_per_joint:
                print("  (GT-Werte sind Mittelwerte der per-Joint-Vektoren)")
            for k in identified:
                gt_v = gt_vals[k]
                id_v = identified[k]
                err_pct = abs(id_v - gt_v) / gt_v * 100 if gt_v != 0 else 0
                print(f"  {k:25s}  GT={gt_v:.6f}  ID={id_v:.6f}  "
                      f"Err={err_pct:.2f}%")

        # Abbruch-Diagnose
        self._diagnose_termination(result, len(result.x))

        return {
            "identified_params": identified,
            "ground_truth": {
                "joint_stiffness": cfg.gt_joint_stiffness,
                "joint_damping": cfg.gt_joint_damping,
                "tendon_stiffness": cfg.gt_tendon_stiffness,
                "armature": cfg.gt_armature,
                **({
                    "joint_stiffness_per_joint": cfg.gt_joint_stiffness_vec,
                } if cfg.gt_joint_stiffness_vec is not None else {}),
                **({
                    "joint_damping_per_joint": cfg.gt_joint_damping_vec,
                } if cfg.gt_joint_damping_vec is not None else {}),
            },
            "final_cost": float(result.fun),
            "n_iterations": result.nfev,
            "elapsed_s": elapsed,
            "scipy_result_message": result.message,
            "config": {
                "mode": cfg.mode,
                "method": cfg.method,
                "sim_time": cfg.sim_time,
                "L_target": cfg.L_target,
                "base_d": cfg.base_d,
            },
        }


# ─────────────────────────────────────────────────────────────────────
# 7. Visualisierung
# ─────────────────────────────────────────────────────────────────────

def plot_convergence(log: OptimizationLog, save_path: Optional[str] = None) -> None:
    """Plottet den Fehlerverlauf über die Iterationen."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Cost über Iterationen
    ax1 = axes[0]
    ax1.semilogy(log.iterations, log.costs, "b-", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("Cost (MSE, log-Skala)")
    ax1.set_title("Optimierungs-Konvergenz")
    ax1.grid(True, alpha=0.3)

    # Gleitender Durchschnitt
    if len(log.costs) > 10:
        window = min(20, len(log.costs) // 5)
        smoothed = np.convolve(log.costs, np.ones(window)/window, mode="valid")
        ax1.semilogy(
            log.iterations[window-1:], smoothed,
            "r-", linewidth=2, label=f"Gleitender Mittelwert ({window})"
        )
        ax1.legend()

    # Parameter über Iterationen (nur für uniform)
    ax2 = axes[1]
    params_arr = np.array(log.params_history)
    if params_arr.ndim == 2 and params_arr.shape[1] <= 6:
        for i, name in enumerate(log.param_names):
            ax2.plot(log.iterations, params_arr[:, i], label=name, linewidth=1.2)
        ax2.set_ylabel("Parameterwert")
        ax2.legend(fontsize=8)
    else:
        # Per-joint: Zeige nur Mittelwerte
        n = (params_arr.shape[1] - 2) // 2
        stiff_mean = np.mean(params_arr[:, :n], axis=1)
        damp_mean = np.mean(params_arr[:, n:2*n], axis=1)
        ax2.plot(log.iterations, stiff_mean, label="mean(stiffness)", linewidth=1.2)
        ax2.plot(log.iterations, damp_mean, label="mean(damping)", linewidth=1.2)
        ax2.plot(log.iterations, params_arr[:, -2], label="tendon_stiffness", linewidth=1.2)
        ax2.plot(log.iterations, params_arr[:, -1], label="armature", linewidth=1.2)
        ax2.set_ylabel("Parameterwert")
        ax2.legend(fontsize=8)

    ax2.set_xlabel("Iteration (Funktionsaufruf)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Konvergenz-Plot gespeichert: {save_path}")

    plt.show()


def plot_trajectory_comparison(
    gt: TrajectoryData,
    est: TrajectoryData,
    joint_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Vergleicht GT- und identifizierte Trajektorien für ausgewählte Joints."""
    if joint_indices is None:
        # Zeige maximal 6 Joints (gleichmäßig verteilt)
        n = gt.qpos.shape[1]
        if n <= 6:
            joint_indices = list(range(n))
        else:
            joint_indices = list(np.linspace(0, n-1, 6, dtype=int))

    n_joints = len(joint_indices)
    fig, axes = plt.subplots(n_joints, 2, figsize=(14, 3 * n_joints), sharex=True)
    if n_joints == 1:
        axes = axes.reshape(1, -1)

    n = min(len(gt.time), len(est.time))

    for row, ji in enumerate(joint_indices):
        # qpos
        ax = axes[row, 0]
        ax.plot(gt.time[:n], gt.qpos[:n, ji], "b-", linewidth=1.5, label="Ground Truth")
        ax.plot(est.time[:n], est.qpos[:n, ji], "r--", linewidth=1.2, label="Identifiziert")
        ax.set_ylabel(f"Joint {ji}\nqpos [rad]")
        if row == 0:
            ax.set_title("Joint-Position (qpos)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # qvel
        ax = axes[row, 1]
        ax.plot(gt.time[:n], gt.qvel[:n, ji], "b-", linewidth=1.5, label="Ground Truth")
        ax.plot(est.time[:n], est.qvel[:n, ji], "r--", linewidth=1.2, label="Identifiziert")
        ax.set_ylabel(f"Joint {ji}\nqvel [rad/s]")
        if row == 0:
            ax.set_title("Joint-Geschwindigkeit (qvel)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Zeit [s]")
    axes[-1, 1].set_xlabel("Zeit [s]")

    plt.suptitle("Trajektorien-Vergleich: Ground Truth vs. Identifiziert", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Trajektorien-Plot gespeichert: {save_path}")

    plt.show()


def visualize_with_viewer(
    config: SysIdConfig,
    identified_params: dict,
) -> None:
    """Zeigt das identifizierte Modell im MuJoCo-Viewer."""
    model = generate_base_model(config)

    if "joint_stiffness" in identified_params:
        # Uniform
        set_uniform_params(
            model,
            identified_params["joint_stiffness"],
            identified_params["joint_damping"],
            identified_params["tendon_stiffness"],
            identified_params["armature"],
        )
    else:
        # Per-joint
        set_per_joint_params(
            model,
            np.array(identified_params["stiffness_per_joint"]),
            np.array(identified_params["damping_per_joint"]),
            identified_params["tendon_stiffness"],
            identified_params["armature"],
        )

    data = mj.MjData(model)
    print("\nStarte MuJoCo-Viewer mit identifizierten Parametern...")
    print("  (Fenster schließen zum Beenden)")

    with mj.viewer.launch_passive(model, data) as v:
        start = time.time()
        while v.is_running() and time.time() - start < 60:
            step_start = time.time()
            sysid_controller(model, data, data.time, 0)
            mj.mj_step(model, data)
            v.sync()
            dt_wall = model.opt.timestep - (time.time() - step_start)
            if dt_wall > 0:
                time.sleep(dt_wall)


# ─────────────────────────────────────────────────────────────────────
# 8. CLI & Main
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SpiRob System-Identifikation via MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=["uniform", "per-joint"], default="uniform",
                   help="Optimierungsmodus (default: uniform)")
    p.add_argument("--method", default="differential_evolution",
                   help="Optimierungsmethode: differential_evolution (global, empfohlen), "
                        "Nelder-Mead, Powell, L-BFGS-B (default: differential_evolution)")
    p.add_argument("--sim-time", type=float, default=2.0,
                   help="Simulationszeit in Sekunden (default: 2.0)")
    p.add_argument("--maxiter", type=int, default=150,
                   help="Maximale Iterationen (default: 150)")
    p.add_argument("--visualize", action="store_true",
                   help="MuJoCo-Viewer nach Konvergenz öffnen")
    p.add_argument("--save-params", type=str, default=None,
                   help="Pfad zum Speichern der identifizierten Parameter (JSON)")
    p.add_argument("--save-plots", type=str, default=None,
                   help="Verzeichnis zum Speichern der Plots")

    # Ground-Truth-Parameter (zum Experimentieren)
    p.add_argument("--gt-stiffness", type=float, default=0.08,
                   help="Uniformer GT-Stiffness-Wert (default: 0.08)")
    p.add_argument("--gt-damping", type=float, default=0.12,
                   help="Uniformer GT-Damping-Wert (default: 0.12)")
    p.add_argument("--gt-tendon-stiffness", type=float, default=60.0)
    p.add_argument("--gt-armature", type=float, default=0.015)
    p.add_argument("--gt-stiffness-per-joint", type=str, default=None,
                   metavar="V0,V1,...",
                   help="Komma-getrennte Stiffness-Werte pro Joint (überschreibt --gt-stiffness). "
                        "Anzahl muss njnt entsprechen, z.B. '0.06,0.07,0.08,...'")
    p.add_argument("--gt-damping-per-joint", type=str, default=None,
                   metavar="V0,V1,...",
                   help="Komma-getrennte Damping-Werte pro Joint (überschreibt --gt-damping). "
                        "Anzahl muss njnt entsprechen, z.B. '0.10,0.12,0.14,...'")
    p.add_argument("--gt-params-file", type=str, default=None,
                   metavar="FILE.json",
                   help="JSON-Datei mit per-Joint-GT-Parametern: "
                        '{"joint_stiffness": [...], "joint_damping": [...]}')

    # Startwerte
    p.add_argument("--init-stiffness", type=float, default=0.02)
    p.add_argument("--init-damping", type=float, default=0.02)
    p.add_argument("--init-tendon-stiffness", type=float, default=30.0)
    p.add_argument("--init-armature", type=float, default=0.005)

    return p.parse_args()


def main():
    args = parse_args()

    # Per-Joint-GT-Parameter parsen (Priorität: --gt-params-file > --gt-*-per-joint > uniform)
    gt_stiffness_vec: Optional[List[float]] = None
    gt_damping_vec: Optional[List[float]] = None

    if args.gt_params_file:
        with open(args.gt_params_file) as f:
            gt_file = json.load(f)
        if "joint_stiffness" in gt_file:
            gt_stiffness_vec = [float(v) for v in gt_file["joint_stiffness"]]
        if "joint_damping" in gt_file:
            gt_damping_vec = [float(v) for v in gt_file["joint_damping"]]
        print(f"GT-Parameter aus Datei geladen: {args.gt_params_file}")

    if args.gt_stiffness_per_joint:
        gt_stiffness_vec = [float(v) for v in args.gt_stiffness_per_joint.split(",")]
    if args.gt_damping_per_joint:
        gt_damping_vec = [float(v) for v in args.gt_damping_per_joint.split(",")]

    config = SysIdConfig(
        sim_time=args.sim_time,
        mode=args.mode,
        method=args.method,
        maxiter=args.maxiter,
        gt_joint_stiffness=args.gt_stiffness,
        gt_joint_damping=args.gt_damping,
        gt_tendon_stiffness=args.gt_tendon_stiffness,
        gt_armature=args.gt_armature,
        gt_joint_stiffness_vec=gt_stiffness_vec,
        gt_joint_damping_vec=gt_damping_vec,
        init_joint_stiffness=args.init_stiffness,
        init_joint_damping=args.init_damping,
        init_tendon_stiffness=args.init_tendon_stiffness,
        init_armature=args.init_armature,
    )

    # System-Identifikation durchführen
    sysid = SystemIdentifier(config)
    result = sysid.run_optimization()

    # Parameter speichern
    if args.save_params:
        save_path = Path(args.save_params)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nParameter gespeichert: {save_path}")

    # Plots erzeugen
    plot_dir = args.save_plots
    convergence_path = None
    trajectory_path = None
    if plot_dir:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        convergence_path = str(Path(plot_dir) / "sysid_convergence.png")
        trajectory_path = str(Path(plot_dir) / "sysid_trajectories.png")

    # Konvergenz-Plot
    plot_convergence(sysid.log, save_path=convergence_path)

    # Finale Simulation mit identifizierten Parametern für Trajektorien-Vergleich
    id_params = result["identified_params"]
    final_model = generate_base_model(config)
    if "joint_stiffness" in id_params:
        set_uniform_params(
            final_model,
            id_params["joint_stiffness"],
            id_params["joint_damping"],
            id_params["tendon_stiffness"],
            id_params["armature"],
        )
    else:
        set_per_joint_params(
            final_model,
            np.array(id_params["stiffness_per_joint"]),
            np.array(id_params["damping_per_joint"]),
            id_params["tendon_stiffness"],
            id_params["armature"],
        )

    final_traj = run_simulation(
        final_model, sysid_controller,
        sim_time=config.sim_time,
        record_dt=config.record_dt,
    )
    plot_trajectory_comparison(sysid.gt_traj, final_traj, save_path=trajectory_path)

    # Optional: Viewer
    if args.visualize:
        visualize_with_viewer(config, id_params)

    return result


if __name__ == "__main__":
    main()
