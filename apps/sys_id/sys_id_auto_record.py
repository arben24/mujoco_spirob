#!/usr/bin/env python3
"""
SpiRob Digital Twin – Auto-Recording mit vordefinierten Kraft-Trajektorien

Dieses Skript steuert die Motoren automatisch gemäß einer vorgegebenen
Abfolge (Phasen) von Kraftprofilen (Konstant, Rampe, Sinus). Gleichzeitig
wird der Aruco-Synchronisationsmarker angezeigt und alle Daten (inklusive
Hardware-Metriken) in ein CSV-Datenframe geschrieben.
"""

import mujoco as mj
import mujoco.viewer as viewer
import time
import struct
import numpy as np
import serial
import cv2
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod

# =============================================================================
# ── 1. DEFINITION DER KRAFT-PROFILE ──────────────────────────────────────────
# =============================================================================

class ForceProfile(ABC):
    @abstractmethod
    def get_force(self, t: float, duration: float) -> float:
        """Gibt die Soll-Kraft zum lokalen Zeitpunkt t (0 <= t <= duration) zurück."""
        pass

class Constant(ForceProfile):
    def __init__(self, force: float):
        self.force = force
    def get_force(self, t: float, duration: float) -> float:
        return self.force

class Ramp(ForceProfile):
    def __init__(self, start_f: float, end_f: float):
        self.start_f = start_f
        self.end_f = end_f
    def get_force(self, t: float, duration: float) -> float:
        progress = np.clip(t / duration, 0.0, 1.0)
        return self.start_f + progress * (self.end_f - self.start_f)

class Sine(ForceProfile):
    def __init__(self, offset: float, amplitude: float, period: float, min_force: float = 10.0):
        self.offset = offset
        self.amplitude = amplitude
        self.period = period
        self.min_force = min_force
    def get_force(self, t: float, duration: float) -> float:
        val = self.offset + self.amplitude * np.sin(2 * np.pi * t / self.period)
        return max(self.min_force, val)

class Phase:
    def __init__(self, duration: float, m0: ForceProfile, m1: ForceProfile):
        self.duration = duration
        self.m0 = m0
        self.m1 = m1

# =============================================================================
# ── 2. EINSTELLUNGEN DES ABLAUFS (HIER ANPASSEN!) ────────────────────────────
# =============================================================================
# Definiere hier, welche Kräfte nacheinander abgefahren werden sollen.
# Motor 0 = m0, Motor 1 = m1.

max_force = 80.0  # Maximale Kraft
min_force = 10.0   # Minimale Kraft 
offset = 50.0
amplitude = 40.0
period = 10.0

# PHASES = [
#     # 1. Vorspannen: Beide Seile für 5 Sekunden sanft auf 2N halten
#     Phase(duration=5.0, m0=Constant(min_force), m1=Constant(min_force)),
    
#     # 2. Rampe M0: Motor 1 bleibt fix (5N), Motor 0 zieht von 5N auf 50N hoch (Dauer 5s)
#     Phase(duration=10.0, m0=Ramp(min_force, max_force), m1=Constant(min_force)),

#     Phase(duration=5.0, m0=Ramp(max_force, min_force), m1=Constant(min_force)),
    
#     # 3. Rampe M1: Motor 0 bleibt fix (50N), Motor 1 zieht nach (5N auf 50N, Dauer 5s)
#     Phase(duration=10.0, m0=Constant(min_force), m1=Ramp(min_force, max_force)),

#     Phase(duration=5.0, m0=Constant(min_force), m1=Ramp(max_force, min_force)),
    
#     # 4. Sinus-Test: M1 fix (50N), M0 macht Sinus (Mitte 50N, +/- 4N, 2s pro Welle) für 10s
#     Phase(duration=10.0, m0=Ramp(min_force, offset), m1=Constant(min_force)),
#     Phase(duration=10.0, m0=Sine(offset=offset, amplitude=amplitude, period=period), m1=Constant(min_force)),
#     Phase(duration=5.0, m0=Ramp(offset, min_force), m1=Constant(min_force)),


#     Phase(duration=10.0, m0=Constant(min_force), m1=Ramp(min_force, offset)),
#     Phase(duration=10.0, m0=Constant(min_force), m1=Sine(offset=offset, amplitude=amplitude, period=period)),
#     Phase(duration=5.0, m0=Constant(min_force), m1=Ramp(offset, min_force)),
#     # 5. Gezieltes Entspannen: Beide Motoren synchron von 10N wieder auf 0N (Dauer 5s)
#     #Phase(duration=5.0, m0=Ramp(10.0, 0.0), m1=Ramp(10.0, 0.0)),
# ]

# PHASES = [

#     Phase(duration=5.0, m0=Constant(10), m1=Constant(10)),
#     Phase(duration=5.0, m0=Constant(80), m1=Constant(10)),
#     Phase(duration=5.0, m0=Constant(30), m1=Constant(10)),
#     Phase(duration=5.0, m0=Constant(100), m1=Constant(10)),

#     Phase(duration=5.0, m0=Constant(10), m1=Constant(10)),
#     Phase(duration=5.0, m0=Constant(10), m1=Constant(80)),
#     Phase(duration=5.0, m0=Constant(10), m1=Constant(30)),
#     Phase(duration=5.0, m0=Constant(10), m1=Constant(100)),

#     Phase(duration=5.0, m0=Constant(10), m1=Constant(30)),
#     Phase(duration=5.0, m0=Constant(80), m1=Constant(30)),
#     Phase(duration=5.0, m0=Constant(30), m1=Constant(30)),
#     Phase(duration=5.0, m0=Constant(100), m1=Constant(30)),

#     Phase(duration=5.0, m0=Constant(30), m1=Constant(10)),
#     Phase(duration=5.0, m0=Constant(30), m1=Constant(80)),
#     Phase(duration=5.0, m0=Constant(30), m1=Constant(30)),
#     Phase(duration=5.0, m0=Constant(30), m1=Constant(100)),
    
# ]

# PHASES = [

#     Phase(duration=1.0, m0=Constant(10), m1=Constant(10)),

#     Phase(duration=10.0, m0=Sine(offset=30, amplitude=80, period=10.0), m1=Constant(10)),
#     Phase(duration=10.0, m0=Constant(10), m1=Sine(offset=30, amplitude=80, period=10.0)),
#     Phase(duration=10.0, m0=Sine(offset=30, amplitude=80, period=10.0), m1=Constant(30)),
#     Phase(duration=5.0, m0=Sine(offset=30, amplitude=30, period=10.0), m1=Sine(offset=50, amplitude=50, period=5.0)),
#     Phase(duration=5.0, m0=Sine(offset=60, amplitude=30, period=5.0), m1=Sine(offset=50, amplitude=50, period=10.0)),
#     Phase(duration=15.0, m0=Sine(offset=60, amplitude=50, period=5.0), m1=Sine(offset=50, amplitude=50, period=10.0)),
#     #Phase(duration=5.0, m0=Constant(10), m1=Sine(offset=50, amplitude=30, period=2.0)),
#     #Phase(duration=5.0, m0=Constant(30), m1=Sine(offset=50, amplitude=30, period=2.0)),

#     Phase(duration=1.0, m0=Constant(10), m1=Constant(10)),

    
# ]


PHASES = [

    Phase(duration=1.0, m0=Constant(10), m1=Constant(10)),

    Phase(duration=10.0, m0=Constant(10), m1=Ramp(10, 120)),
    Phase(duration=5.0, m0=Constant(10), m1=Ramp(120, 10)),

    Phase(duration=10.0, m0=Ramp(10, 120), m1=Constant(10)),
    Phase(duration=5.0, m0=Ramp(120, 10), m1=Constant(10)),

    Phase(duration=10.0, m0=Constant(30), m1=Ramp(10, 120)),
    Phase(duration=5.0, m0=Constant(30), m1=Ramp(120, 10)),

    Phase(duration=10.0, m0=Ramp(10, 120), m1=Constant(30)),
    Phase(duration=5.0, m0=Ramp(120, 10), m1=Constant(30)),

    Phase(duration=1.0, m0=Constant(10), m1=Constant(10)),

    
]


# =============================================================================
# ── INTERNE KONFIGURATION (SERIELL / MUJOCO / ARUCO) ─────────────────────────
# =============================================================================

PORT = "/dev/ttyUSB0"
BAUDRATE = 460800
STRUCT_FMT = "<I ff ff" 
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)

SEND_HZ = 50 
SEND_INTERVAL = 1.0 / SEND_HZ
FORCE_DEADBAND = 0.05 

START_MARKER_ID = 20
MARKER_UPDATE_INTERVAL = 5.0
MARKER_SIZE_PX = 1200

def send_cmd(ser: serial.Serial, cmd: str) -> None:
    ser.write((cmd + "\n").encode("ascii"))

def drain_telemetry(ser: serial.Serial):
    latest = None
    while ser.in_waiting >= 2 + STRUCT_SIZE:
        b0 = ser.read(1)
        if b0 == b"\xaa":
            b1 = ser.read(1)
            if b1 == b"\x55":
                pkt = ser.read(STRUCT_SIZE)
                if len(pkt) == STRUCT_SIZE:
                    latest = struct.unpack(STRUCT_FMT, pkt)
        elif b0 == b"\xbb":
            ser.read(2)
    return latest

# =============================================================================
# ── HAUPTPROGRAMM ────────────────────────────────────────────────────────────
# =============================================================================

def main():
    # ── MuJoCo model setup
    model_path = Path(__file__).resolve().parent / "spiral_chain.xml"
    if not model_path.exists():
        print(f"Warning: {model_path} not found. Trying local 'spiral_chain.xml'")
        model_path = Path("spiral_chain.xml")
        
    spec = mj.MjSpec.from_file(str(model_path))

    cylinder = spec.worldbody.add_body(name="cylinder", pos=[-0.11, 0.00, 0.11])
    cylinder.add_geom(
        name="cyl_geom",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        size=[0.05, 0.15, 0.05],
        euler=[90, 0, 0],
        rgba=[0.2, 0.8, 0.5, 1],
        density=1000,
    )

    model = spec.compile()
    data = mj.MjData(model)
    print("MuJoCo-Modell geladen.")

    # ── Serial connection
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.001)
        time.sleep(0.1)
        ser.reset_input_buffer()
        print(f"Hardware verbunden: {PORT} @ {BAUDRATE}")
        send_cmd(ser, "start all")
    except Exception as e:
        print(f"Konnte Serielle Verbindung nicht öffnen (Hardware an?). Fehler: {e}")
        return

    # ── ArUco Setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    current_marker_id = START_MARKER_ID
    cv2.namedWindow("Sync Marker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sync Marker", MARKER_SIZE_PX, MARKER_SIZE_PX)

    records = []
    
    prev_f = [None, None]
    last_send_t = 0.0
    last_marker_t = time.time()

    # ── Phasen-Logik Setup
    current_phase_idx = 0
    phase_start_time = None
    
    # Berechne die Gesamtdauer aller Phasen für den Benutzer
    total_duration = sum([p.duration for p in PHASES])
    print(f"\nStarte automatische Trajektorien! ({len(PHASES)} Phasen, Total: {total_duration:.1f}s)")

    try:
        with mj.viewer.launch_passive(model, data) as v:
            t0 = time.time()
            phase_start_time = t0
            
            while v.is_running() and current_phase_idx < len(PHASES):
                now = time.time()
                step_start = now
                global_ts = now - t0
                phase_time = now - phase_start_time
                
                # Check for Phase transition
                current_phase = PHASES[current_phase_idx]
                if phase_time >= current_phase.duration:
                    current_phase_idx += 1
                    phase_start_time = now
                    if current_phase_idx >= len(PHASES):
                        print("\nAlle Phasen erfolgreich abgeschlossen.")
                        break
                    current_phase = PHASES[current_phase_idx]
                    phase_time = 0.0

                # 1) Handle ArUco Marker Window
                if now - last_marker_t >= MARKER_UPDATE_INTERVAL:
                    current_marker_id += 1
                    last_marker_t = now

                marker_img = cv2.aruco.generateImageMarker(aruco_dict, current_marker_id, MARKER_SIZE_PX)
                cv2.imshow("Sync Marker", marker_img)
                cv2.waitKey(1)

                # 2) Calculate Actuator Forces from Profile
                f0 = current_phase.m0.get_force(phase_time, current_phase.duration)
                f1 = current_phase.m1.get_force(phase_time, current_phase.duration)
                
                # Update viewer UI (optional, so you see what the auto-system specifies)
                data.ctrl[0] = -f0
                data.ctrl[1] = -f1

                # 3) Forward to hardware (throttled)
                if now - last_send_t >= SEND_INTERVAL:
                    if prev_f[0] is None or abs(f0 - prev_f[0]) > FORCE_DEADBAND:
                        send_cmd(ser, f"f 0 {f0:.2f}")
                        prev_f[0] = f0
                    if prev_f[1] is None or abs(f1 - prev_f[1]) > FORCE_DEADBAND:
                        send_cmd(ser, f"f 1 {f1:.2f}")
                        prev_f[1] = f1
                    last_send_t = now

                # 4) Read Telemetry & Record
                hw = drain_telemetry(ser)
                if hw:
                    hw_ts, hf0, hf1, hr0, hr1 = hw
                    print(f"\r[Phase {current_phase_idx+1}/{len(PHASES)}] Zeit: {global_ts:5.1f}s | "
                          f"Soll: {f0:5.1f}N {f1:5.1f}N | "
                          f"Ist: {hf0:5.1f}N {hf1:5.1f}N (Sync:{current_marker_id})   ", end="", flush=True)
                    
                    records.append({
                        "global_timestamp_s": global_ts,
                        "phase_idx": current_phase_idx + 1,
                        "aruco_id": current_marker_id,
                        "cmd_force_0_N": f0,
                        "cmd_force_1_N": f1,
                        "meas_force_0_N": hf0,
                        "meas_force_1_N": hf1,
                        "meas_length_0_mm": hr0,
                        "meas_length_1_mm": hr1,
                        "hw_timestamp_us": hw_ts
                    })

                # 5) Step simulation
                mj.mj_step(model, data)
                v.sync()

                dt = model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)

    except KeyboardInterrupt:
        print("\nAufzeichnung vorzeitig abgebrochen durch Benutzer.")

    finally:
        # Cleanup
        cv2.destroyAllWindows()
        send_cmd(ser, "f 0 0.0")
        send_cmd(ser, "f 1 0.0")
        send_cmd(ser, "stop")
        print("\nMotoren gestoppt.")
        ser.close()
        print("Serielle Verbindung geschlossen.")

        # Save data
        if records:
            df = pd.DataFrame(records)
            out_dir = Path(__file__).resolve().parent / "build"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "recorded_sys_id_auto_data.csv"
            df.to_csv(out_file, index=False)
            print(f"Daten ({len(df)} Einträge) erfolgreich gespeichert nach:\n{out_file}")
        else:
            print("\nKeine Daten aufgezeichnet.")

if __name__ == "__main__":
    main()
