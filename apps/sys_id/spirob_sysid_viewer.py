"""
SpiRob Controller Echtzeit-Visualisierung
==========================================

Wählbare Controller und frei einstellbare physikalische Parameter (auch pro Joint).

CONTROLLER  (--controller NAME):
  sysid     Multi-Frequenz-Anregung aus spirob_sysid (default)
  static    Konstante Kraft  (--amp0, --amp1)
  ramp      Lineare Rampe    (--amp0, --amp1, --ramp-time)
  sine      Sinuswelle       (--amp0, --amp1, --freq, --phase-shift, --ramp-time)
  chirp     Frequenz-Sweep   (--amp0, --amp1, --freq-start, --freq-end)
  step      Sprungfunktion   (--amp0, --amp1, --step-time)

PHYSIKALISCHE PARAMETER:
  --stiffness FLOAT                   Uniform für alle Joints
  --damping   FLOAT                   Uniform für alle Joints
  --stiffness-per-joint "v0,v1,..."   Pro Joint (Länge = njnt)
  --damping-per-joint   "v0,v1,..."   Pro Joint (Länge = njnt)
  --tendon-stiffness FLOAT
  --armature FLOAT
  --params FILE.json                  Aus spirob_sysid --save-params laden
                                      (CLI-Werte überschreiben JSON)

BEISPIELE:
  uv run apps/spirob_sysid_viewer.py
  uv run apps/spirob_sysid_viewer.py --controller sine --freq 2.0 --amp0 8 --amp1 10
  uv run apps/spirob_sysid_viewer.py --controller chirp --freq-start 0.1 --freq-end 5.0
  uv run apps/spirob_sysid_viewer.py --controller step --step-time 1.0 --amp0 10 --amp1 12
  uv run apps/spirob_sysid_viewer.py --stiffness 0.05 --damping 0.08
  uv run apps/spirob_sysid_viewer.py --stiffness-per-joint "0.04,0.05,0.06,0.07,0.08,0.08,0.07,0.06,0.05,0.04,0.04,0.05,0.06,0.07,0.08,0.08,0.07,0.06,0.05"
  uv run apps/spirob_sysid_viewer.py --params build/sysid_params.json --controller ramp
  uv run apps/spirob_sysid_viewer.py --speed 2.0 --no-plot
"""

import argparse
import json
import os
import sys
import time
import threading
from collections import deque
from pathlib import Path
from typing import Callable, List, Optional

import mujoco as mj
import numpy as np

# ── sys.path so that `apps.*` is importable when called via `uv run apps/...` ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math_spirob.spirob_generator as sg
from apps.sys_id.spirob_sysid import (
    SysIdConfig,
    generate_base_model,
    set_uniform_params,
    set_per_joint_params,
    sysid_controller,
)

ControllerFunc = Callable[[mj.MjModel, mj.MjData, float, int], None]


# ─────────────────────────────────────────────────────────────────────
# 1. Controller-Bibliothek
# ─────────────────────────────────────────────────────────────────────

def make_static_controller(amp0: float, amp1: float) -> ControllerFunc:
    """Konstante Kraft (immer gleich, kein ramp)."""
    def ctrl(model, data, t, step):
        data.ctrl[0] = -abs(amp0)
        data.ctrl[1] = -abs(amp1)
    return ctrl


def make_ramp_controller(amp0: float, amp1: float, ramp_time: float) -> ControllerFunc:
    """Lineare Rampe von 0 auf amp in ramp_time Sekunden, dann konstant."""
    def ctrl(model, data, t, step):
        r = min(t / max(ramp_time, 1e-9), 1.0)
        data.ctrl[0] = -r * abs(amp0)
        data.ctrl[1] = -r * abs(amp1)
    return ctrl


def make_sine_controller(
    amp0: float, amp1: float,
    freq: float,
    phase_shift: float,
    ramp_time: float,
) -> ControllerFunc:
    """
    Sinuswelle mit optionalem Ramp-Anlauf.
    ctrl = -ramp * amp * 0.5*(1 + sin(2π*freq*t + phase))  → immer negativ (Zug)
    """
    def ctrl(model, data, t, step):
        r  = min(t / max(ramp_time, 1e-9), 1.0)
        s0 = 0.5 * (1.0 + np.sin(2 * np.pi * freq * t))
        s1 = 0.5 * (1.0 + np.sin(2 * np.pi * freq * t + phase_shift))
        data.ctrl[0] = -r * abs(amp0) * s0
        data.ctrl[1] = -r * abs(amp1) * s1
    return ctrl


def make_chirp_controller(
    amp0: float, amp1: float,
    f_start: float, f_end: float,
    sweep_time: float,
) -> ControllerFunc:
    """
    Linearer Frequenz-Sweep von f_start bis f_end über sweep_time Sekunden.
    Danach konstante Frequenz bei f_end. Gut für Systemidentifikation.
    """
    def ctrl(model, data, t, step):
        t_s = min(t, sweep_time)
        # Phase: integral der Momentanfrequenz
        phase = 2 * np.pi * (f_start * t_s + (f_end - f_start) * t_s**2 / (2 * sweep_time))
        # Tendon 1 um π/2 phasenverschoben für orthogonale Anregung
        data.ctrl[0] = -abs(amp0) * 0.5 * (1.0 + np.sin(phase))
        data.ctrl[1] = -abs(amp1) * 0.5 * (1.0 + np.sin(phase + np.pi / 2))
    return ctrl


def make_step_controller(amp0: float, amp1: float, t_step: float) -> ControllerFunc:
    """Sprungfunktion: 0 bis t_step, dann volle Kraft."""
    def ctrl(model, data, t, step):
        if t >= t_step:
            data.ctrl[0] = -abs(amp0)
            data.ctrl[1] = -abs(amp1)
        else:
            data.ctrl[0] = 0.0
            data.ctrl[1] = 0.0
    return ctrl


def build_controller(args) -> tuple:
    """Erzeugt den gewählten Controller und einen Beschreibungs-String."""
    name = args.controller

    if name == "sysid":
        desc = "sysid  −ramp·(5+3·f0.5+1.5·f1.5) / −ramp·(8+4·f0.5−2·f3.0)"
        return sysid_controller, desc

    elif name == "static":
        desc = f"static  amp0={args.amp0:.2f}  amp1={args.amp1:.2f}"
        return make_static_controller(args.amp0, args.amp1), desc

    elif name == "ramp":
        desc = f"ramp  amp0={args.amp0:.2f}  amp1={args.amp1:.2f}  t_ramp={args.ramp_time:.2f}s"
        return make_ramp_controller(args.amp0, args.amp1, args.ramp_time), desc

    elif name == "sine":
        desc = (f"sine  {args.freq:.2f}Hz  amp0={args.amp0:.2f}  amp1={args.amp1:.2f}"
                f"  Δφ={np.degrees(args.phase_shift):.0f}°  ramp={args.ramp_time:.2f}s")
        return make_sine_controller(args.amp0, args.amp1, args.freq,
                                    args.phase_shift, args.ramp_time), desc

    elif name == "chirp":
        desc = (f"chirp  {args.freq_start:.2f}→{args.freq_end:.2f}Hz  "
                f"amp0={args.amp0:.2f}  amp1={args.amp1:.2f}")
        return make_chirp_controller(args.amp0, args.amp1,
                                     args.freq_start, args.freq_end,
                                     args.sim_time), desc

    elif name == "step":
        desc = f"step  t={args.step_time:.2f}s  amp0={args.amp0:.2f}  amp1={args.amp1:.2f}"
        return make_step_controller(args.amp0, args.amp1, args.step_time), desc

    else:
        raise ValueError(f"Unbekannter Controller: '{name}'")


# ─────────────────────────────────────────────────────────────────────
# 2. Modell + Parameter
# ─────────────────────────────────────────────────────────────────────

def resolve_params(args, njnt: int) -> Optional[dict]:
    """
    Löst die physikalischen Parameter auf. Priorität:
      CLI --stiffness-per-joint / --stiffness  >  --params JSON  >  Modell-Standard

    Gibt None zurück wenn keine Parameter angegeben wurden.
    """
    stiffness_vec = None
    damping_vec   = None
    tendon_stiff  = None
    armature      = None

    # Basis: JSON
    if args.params:
        p = Path(args.params)
        if p.exists():
            with open(p) as f:
                jdata = json.load(f)
                print(f"  JSON-Parameter geladen aus '{p}':")
            ip = jdata.get("identified_params", {})
            print(json.dumps(ip, indent=4))
            if "stiffness_per_joint" in ip:
                stiffness_vec = np.array(ip["stiffness_per_joint"])
                damping_vec   = np.array(ip["damping_per_joint"])
            elif "joint_stiffness" in ip:
                stiffness_vec = np.full(njnt, float(ip["joint_stiffness"]))
                damping_vec   = np.full(njnt, float(ip["joint_damping"]))
                print("  WARNUNG: JSON-Parameter verwenden altes Format 'joint_stiffness' und 'joint_damping' (uniform für alle Joints).")
            if "tendon_stiffness" in ip:
                tendon_stiff = float(ip["tendon_stiffness"])
            if "armature" in ip:
                armature = float(ip["armature"])
        else:
            print(f"  WARNUNG: --params Datei nicht gefunden: {p}")

    # CLI überschreibt JSON
    if args.stiffness_per_joint:
        vals = [float(v) for v in args.stiffness_per_joint.split(",")]
        if len(vals) != njnt:
            raise ValueError(
                f"--stiffness-per-joint: {len(vals)} Werte, aber {njnt} Joints erwartet.")
        stiffness_vec = np.array(vals)
    elif args.stiffness is not None:
        stiffness_vec = np.full(njnt, args.stiffness)

    if args.damping_per_joint:
        vals = [float(v) for v in args.damping_per_joint.split(",")]
        if len(vals) != njnt:
            raise ValueError(
                f"--damping-per-joint: {len(vals)} Werte, aber {njnt} Joints erwartet.")
        damping_vec = np.array(vals)
    elif args.damping is not None:
        damping_vec = np.full(njnt, args.damping)

    if args.tendon_stiffness is not None:
        tendon_stiff = args.tendon_stiffness
    if args.armature is not None:
        armature = args.armature

    if all(v is None for v in [stiffness_vec, damping_vec, tendon_stiff, armature]):
        return None

    return dict(stiffness_vec=stiffness_vec, damping_vec=damping_vec,
                tendon_stiff=tendon_stiff, armature=armature)


def build_model(args) -> tuple:
    """Baut das MuJoCo-Modell, setzt Parameter und gibt (model, applied_params) zurück."""
    cfg = SysIdConfig(
        L_target=args.L_target,
        base_d=args.base_d,
        tip_d=args.tip_d,
        Delta_theta_deg=args.delta_theta,
    )
    model = generate_base_model(cfg)
    njnt  = model.njnt

    # Modell-Standardwerte auslesen
    def _read_model():
        s = np.array([model.jnt_stiffness[i] for i in range(njnt)])
        d = np.array([model.dof_damping[model.jnt_dofadr[i]] for i in range(njnt)])
        t = float(model.tendon_stiffness[0]) if model.ntendon > 0 else 0.0
        a = float(model.dof_armature[model.jnt_dofadr[0]])
        return s, d, t, a

    params = resolve_params(args, njnt)

    if params is None:
        s, d, t, a = _read_model()
        src = "model-default"
    else:
        s_def, d_def, t_def, a_def = _read_model()
        s = params["stiffness_vec"] if params["stiffness_vec"] is not None else s_def
        d = params["damping_vec"]   if params["damping_vec"]   is not None else d_def
        t = params["tendon_stiff"]  if params["tendon_stiff"]  is not None else t_def
        a = params["armature"]      if params["armature"]      is not None else a_def
        set_per_joint_params(model, s, d, t, a)
        src = "cli/json"

    applied = dict(stiffness_vec=s, damping_vec=d, tendon_stiff=t, armature=a, source=src)
    return model, applied


def print_param_table(applied: dict, njnt: int) -> None:
    src   = applied.get("source", "?")
    s     = applied["stiffness_vec"]
    d     = applied["damping_vec"]
    t_s   = applied["tendon_stiff"]
    arm   = applied["armature"]
    is_us = np.allclose(s, s[0])
    is_ud = np.allclose(d, d[0])

    print(f"\n  PHYSIKALISCHE PARAMETER  (Quelle: {src})")
    print(f"  {'J':>3} | {'Stiffness':>10} | {'Damping':>10}")
    print("  " + "-"*3 + "-+-" + "-"*10 + "-+-" + "-"*10)
    for ji in range(njnt):
        ms = "" if is_us else " *"
        md = "" if is_ud else " *"
        print(f"  {ji:>3} | {s[ji]:>10.5f}{ms:<2} | {d[ji]:>10.5f}{md}")
    print(f"  tendon_stiffness : {t_s:.4f}")
    print(f"  armature         : {arm:.5f}")
    if not is_us or not is_ud:
        print("  (* = per-joint variiert)")


# ─────────────────────────────────────────────────────────────────────
# 3. Live-Plot
# ─────────────────────────────────────────────────────────────────────

class LivePlot:
    WINDOW_S  = 8.0
    UPDATE_HZ = 20

    def __init__(self, njnt: int, controller_desc: str):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gs

        self._plt = plt
        self.njnt = njnt

        self.t_buf     = deque(maxlen=4000)
        self.ctrl0_buf = deque(maxlen=4000)
        self.ctrl1_buf = deque(maxlen=4000)
        n_show = min(6, njnt)
        self._show_joints = list(np.linspace(0, njnt - 1, n_show, dtype=int))
        self.qpos_bufs = [deque(maxlen=4000) for _ in self._show_joints]
        self._lock = threading.Lock()

        self.fig = plt.figure(figsize=(14, 8), num="SpiRob Controller – Live")
        spec = gs.GridSpec(3, 2, figure=self.fig, hspace=0.48, wspace=0.35)

        self.ax_ctrl  = self.fig.add_subplot(spec[0, :])
        self.ax_q     = self.fig.add_subplot(spec[1, :])
        self.ax_phase = self.fig.add_subplot(spec[2, 0])
        self.ax_info  = self.fig.add_subplot(spec[2, 1])

        self.line_c0, = self.ax_ctrl.plot([], [], color="#2196F3", lw=1.6,
                                          label="ctrl[0]  Tendon 0")
        self.line_c1, = self.ax_ctrl.plot([], [], color="#F44336", lw=1.6,
                                          label="ctrl[1]  Tendon 1")
        self.ax_ctrl.set_ylabel("Ctrl-Signal")
        self.ax_ctrl.set_xlabel("Zeit [s]")
        self.ax_ctrl.set_title(f"Steuersignale  –  {controller_desc}", fontsize=9)
        self.ax_ctrl.legend(fontsize=8, loc="upper right")
        self.ax_ctrl.grid(True, alpha=0.3)
        self.ax_ctrl.set_ylim(-16, 1)

        cmap = plt.cm.plasma
        self.lines_q = []
        for k, ji in enumerate(self._show_joints):
            col = cmap(k / max(len(self._show_joints) - 1, 1))
            ln, = self.ax_q.plot([], [], color=col, lw=1.2, label=f"J{ji}")
            self.lines_q.append(ln)
        self.ax_q.set_ylabel("qpos [rad]")
        self.ax_q.set_xlabel("Zeit [s]")
        self.ax_q.set_title("Joint-Positionen (repr. Auswahl)", fontsize=9)
        self.ax_q.legend(fontsize=7, loc="upper right", ncol=3)
        self.ax_q.grid(True, alpha=0.3)

        self.phase_line, = self.ax_phase.plot([], [], color="#9C27B0", lw=0.8, alpha=0.6)
        self.phase_dot,  = self.ax_phase.plot([], [], "o", color="#9C27B0", ms=6, zorder=5)
        self.ax_phase.set_xlabel("ctrl[0]")
        self.ax_phase.set_ylabel("ctrl[1]")
        self.ax_phase.set_title("Phasenraum Ctrl", fontsize=9)
        self.ax_phase.set_xlim(-16, 1)
        self.ax_phase.set_ylim(-16, 1)
        self.ax_phase.grid(True, alpha=0.3)

        self.ax_info.axis("off")
        self.info_text = self.ax_info.text(
            0.05, 0.97, "", transform=self.ax_info.transAxes,
            va="top", ha="left", fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", alpha=0.85),
        )

        plt.ion()
        plt.show(block=False)

    def push(self, t, ctrl0, ctrl1, qpos):
        with self._lock:
            self.t_buf.append(t)
            self.ctrl0_buf.append(ctrl0)
            self.ctrl1_buf.append(ctrl1)
            for i, ji in enumerate(self._show_joints):
                self.qpos_bufs[i].append(float(qpos[ji]) if ji < len(qpos) else 0.0)

    def update(self, sim_t, step, ctrl0, ctrl1, param_info: str = ""):
        with self._lock:
            t_arr  = np.array(self.t_buf)
            c0_arr = np.array(self.ctrl0_buf)
            c1_arr = np.array(self.ctrl1_buf)
            q_arrs = [np.array(b) for b in self.qpos_bufs]
        if len(t_arr) < 2:
            return

        t_min = t_arr[-1] - self.WINDOW_S
        mask  = t_arr >= t_min
        t_w   = t_arr[mask]

        self.line_c0.set_data(t_w, c0_arr[mask])
        self.line_c1.set_data(t_w, c1_arr[mask])
        self.ax_ctrl.set_xlim(t_w[0], max(t_w[-1], t_w[0] + self.WINDOW_S))
        # Y-Achse anpassen
        c_all = np.concatenate([c0_arr[mask], c1_arr[mask]])
        if np.ptp(c_all) > 0.05:
            mg = np.ptp(c_all) * 0.1
            self.ax_ctrl.set_ylim(c_all.min() - mg, c_all.max() + mg)

        all_q_vals = []
        for k, ln in enumerate(self.lines_q):
            q_w = q_arrs[k][mask]
            ln.set_data(t_w, q_w)
            all_q_vals.append(q_w)
        if all_q_vals:
            all_q = np.concatenate(all_q_vals)
            if np.ptp(all_q) > 1e-6:
                mg = np.ptp(all_q) * 0.12
                self.ax_q.set_ylim(all_q.min() - mg, all_q.max() + mg)
        self.ax_q.set_xlim(t_w[0], max(t_w[-1], t_w[0] + self.WINDOW_S))

        ph_n = min(len(t_arr), 600)
        self.phase_line.set_data(c0_arr[-ph_n:], c1_arr[-ph_n:])
        self.phase_dot.set_data([ctrl0], [ctrl1])
        if len(c0_arr) > 5:
            mn0, mx0 = c0_arr.min(), c0_arr.max()
            mn1, mx1 = c1_arr.min(), c1_arr.max()
            pad = max(0.3, (mx0 - mn0) * 0.05)
            self.ax_phase.set_xlim(mn0 - pad, mx0 + pad)
            self.ax_phase.set_ylim(mn1 - pad, mx1 + pad)

        info = (
            f"t_sim  : {sim_t:7.3f} s\n"
            f"steps  : {step:7d}\n"
            f"\n"
            f"ctrl[0]: {ctrl0:+7.3f}\n"
            f"ctrl[1]: {ctrl1:+7.3f}\n"
            f"\n"
            + param_info
        )
        self.info_text.set_text(info)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        try:
            self._plt.close(self.fig)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
# 4. Echtzeit-Simulation
# ─────────────────────────────────────────────────────────────────────

def run_viewer(args) -> None:
    model, applied = build_model(args)
    data  = mj.MjData(model)
    mj.mj_resetData(model, data)

    controller, ctrl_desc = build_controller(args)

    njnt = model.njnt
    dt   = model.opt.timestep

    print(f"\n{'='*64}")
    print(f"SPIROB CONTROLLER-VIEWER")
    print(f"  Controller   : {ctrl_desc}")
    print(f"  Joints       : {njnt}   Timestep: {dt*1000:.2f} ms")
    print(f"  Speed        : {args.speed:.1f}x    Sim-Zeit/Zyklus: {args.sim_time:.0f}s")
    print_param_table(applied, njnt)

    s     = applied["stiffness_vec"]
    d     = applied["damping_vec"]
    is_us = np.allclose(s, s[0])
    is_ud = np.allclose(d, d[0])
    param_info = (
        f"stiff  : {s[0]:.4f}" + ("" if is_us else f"…{s[-1]:.4f}") + "\n"
        f"damp   : {d[0]:.4f}" + ("" if is_ud else f"…{d[-1]:.4f}") + "\n"
        f"t_stiff: {applied['tendon_stiff']:.3f}\n"
        f"arm    : {applied['armature']:.5f}\n"
    )

    print(f"\n  [Fenster schließen oder Ctrl+C zum Beenden]")
    print(f"{'='*64}\n")

    live = None
    if not args.no_plot:
        try:
            live = LivePlot(njnt, ctrl_desc)
        except Exception as e:
            print(f"  [Matplotlib] Live-Plot nicht verfügbar: {e}")

    plot_every = max(1, int((1.0 / LivePlot.UPDATE_HZ) / dt)) if live else 0
    push_every = max(1, plot_every // 4)
    step       = 0
    wall_start = time.perf_counter()

    with mj.viewer.launch_passive(model, data) as v:
        v.cam.distance  = 0.7
        v.cam.elevation = -20
        v.cam.azimuth   = 140

        while v.is_running():
            t0 = time.perf_counter()

            controller(model, data, data.time, step)
            mj.mj_step(model, data)
            step += 1

            if live:
                if step % push_every == 0:
                    live.push(data.time, float(data.ctrl[0]),
                              float(data.ctrl[1]), data.qpos)
                if step % plot_every == 0:
                    live.update(data.time, step,
                                float(data.ctrl[0]), float(data.ctrl[1]),
                                param_info)

            v.sync()

            if data.time >= args.sim_time:
                mj.mj_resetData(model, data)
                step = 0
                print(f"  [Reset]  t={args.sim_time:.1f}s  wall={time.perf_counter()-wall_start:.1f}s")

            sleep_t = dt / args.speed - (time.perf_counter() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    if live:
        live.close()
    print("Viewer geschlossen.")


# ─────────────────────────────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SpiRob Controller Echtzeit-Visualisierung",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Simulation ──
    p.add_argument("--sim-time", type=float, default=20.0,
                   help="Sim-Zeit pro Zyklus [s], danach Reset (default: 20)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Speed-Faktor: 1.0=Echtzeit, 2.0=doppelt (default: 1.0)")
    p.add_argument("--no-plot", action="store_true",
                   help="Kein Matplotlib-Live-Plot")

    # ── Controller ──
    p.add_argument("--controller", default="sysid",
                   choices=["sysid", "static", "ramp", "sine", "chirp", "step"],
                   help="Controller-Typ (default: sysid)")
    p.add_argument("--amp0",        type=float, default=8.0,
                   help="Amplitude Tendon 0 (positiv, wird intern negiert) (default: 8.0)")
    p.add_argument("--amp1",        type=float, default=10.0,
                   help="Amplitude Tendon 1 (default: 10.0)")
    p.add_argument("--ramp-time",   type=float, default=0.5,
                   help="Ramp-Dauer [s] für ramp/sine (default: 0.5)")
    p.add_argument("--freq",        type=float, default=1.0,
                   help="Frequenz [Hz] für sine (default: 1.0)")
    p.add_argument("--phase-shift", type=float, default=1.5708, metavar="RAD",
                   help="Phasenversatz ctrl[1] [rad] für sine (default: π/2 ≈ 1.5708)")
    p.add_argument("--freq-start",  type=float, default=0.2,
                   help="Start-Frequenz [Hz] für chirp (default: 0.2)")
    p.add_argument("--freq-end",    type=float, default=5.0,
                   help="End-Frequenz [Hz] für chirp (default: 5.0)")
    p.add_argument("--step-time",   type=float, default=1.0,
                   help="Sprung-Zeitpunkt [s] für step (default: 1.0)")

    # ── Physikalische Parameter ──
    p.add_argument("--stiffness", type=float, default=None,
                   help="Uniform Joint-Stiffness (alle Joints)")
    p.add_argument("--damping",   type=float, default=None,
                   help="Uniform Joint-Damping (alle Joints)")
    p.add_argument("--stiffness-per-joint", type=str, default=None, metavar="v0,v1,...",
                   help="Komma-getrennte Stiffness pro Joint (überschreibt --stiffness)")
    p.add_argument("--damping-per-joint",   type=str, default=None, metavar="v0,v1,...",
                   help="Komma-getrennte Damping pro Joint (überschreibt --damping)")
    p.add_argument("--tendon-stiffness", type=float, default=None,
                   help="Tendon-Stiffness (beide Tendons)")
    p.add_argument("--armature",  type=float, default=None,
                   help="DOF-Armature (alle Joints)")
    p.add_argument("--params", type=str, default=None, metavar="FILE.json",
                   help="JSON aus spirob_sysid --save-params laden (CLI überschreibt JSON-Werte)")

    # ── Modell-Geometrie ──
    p.add_argument("--L-target",    type=float, default=0.44)
    p.add_argument("--base-d",      type=float, default=0.1)
    p.add_argument("--tip-d",       type=float, default=0.03)
    p.add_argument("--delta-theta", type=float, default=30.0)

    return p.parse_args()


def main():
    args = parse_args()
    run_viewer(args)


if __name__ == "__main__":
    main()
