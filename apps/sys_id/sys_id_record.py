#!/usr/bin/env python3
"""
SpiRob Digital Twin – MuJoCo ↔ Hardware Bridge with Data Recording

Opens the MuJoCo viewer and an OpenCV window displaying an ArUco marker.
The ArUco marker starts at ID 20 and increments every 5 seconds for video synchronization.
Simultaneously, commanded forces, measured forces, and lengths are recorded into a Pandas DataFrame.
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

# ── Serial config ───────────────────────────────────────────────────────────
PORT = "/dev/ttyUSB0"
BAUDRATE = 460800
HEADER = b"\xaa\x55"
STEP_END = b"\xbb\x66"
STRUCT_FMT = "<I ff ff"  # uint32 ts_us, float force[2], float rope_mm[2]
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)  # 20

SEND_HZ = 50  # max command rate to ESP32
SEND_INTERVAL = 1.0 / SEND_HZ
FORCE_DEADBAND = 0.05  # N – ignore changes smaller than this

# ── ArUco config ────────────────────────────────────────────────────────────
START_MARKER_ID = 20
MARKER_UPDATE_INTERVAL = 5.0  # seconds
MARKER_SIZE_PX = 1200

# ── Helpers ──────────────────────────────────────────────────────────────────
def send_cmd(ser: serial.Serial, cmd: str) -> None:
    """Send an ASCII command (auto-appends newline)."""
    ser.write((cmd + "\n").encode("ascii"))

def drain_telemetry(ser: serial.Serial):
    """Read all pending binary packets; return latest status tuple or None."""
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
            ser.read(2)  # step-end: marker byte + motor index
    return latest

# ── Main ────────────────────────────────────────────────────────────────────
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

    data.ctrl[0] = -5.0
    data.ctrl[1] = -5.0

    # ── Serial connection
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.001)
    time.sleep(0.1)
    ser.reset_input_buffer()
    print(f"Verbunden: {PORT} @ {BAUDRATE}")

    send_cmd(ser, "start all")
    print("Kraftregelung gestartet (start all)")

    # ── ArUco Setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    current_marker_id = START_MARKER_ID
    
    cv2.namedWindow("Sync Marker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sync Marker", MARKER_SIZE_PX, MARKER_SIZE_PX)

    # ── Data Recording Setup
    records = []

    # ── Main loop
    prev_f = [None, None]
    last_send_t = 0.0
    last_marker_t = time.time()

    try:
        with mj.viewer.launch_passive(model, data) as v:
            t0 = time.time()
            
            while v.is_running() and time.time() - t0 < 360:
                now = time.time()
                step_start = now
                global_ts = now - t0

                # 1) Handle ArUco Marker Window
                if now - last_marker_t >= MARKER_UPDATE_INTERVAL:
                    current_marker_id += 1
                    last_marker_t = now

                marker_img = cv2.aruco.generateImageMarker(aruco_dict, current_marker_id, MARKER_SIZE_PX)
                cv2.imshow("Sync Marker", marker_img)
                cv2.waitKey(1)

                # 2) Read actuator ctrl values from GUI (Newtons)
                f0 = -1 * float(data.ctrl[0])
                f1 = -1 * float(data.ctrl[1])

                # 3) Forward to hardware (throttled, with dead-band)
                if now - last_send_t >= SEND_INTERVAL:
                    if prev_f[0] is None or abs(f0 - prev_f[0]) > FORCE_DEADBAND:
                        send_cmd(ser, f"f 0 {f0:.2f}")
                        prev_f[0] = f0
                    if prev_f[1] is None or abs(f1 - prev_f[1]) > FORCE_DEADBAND:
                        send_cmd(ser, f"f 1 {f1:.2f}")
                        prev_f[1] = f1
                    last_send_t = now

                # 4) Drain hardware telemetry and Record Data
                hw = drain_telemetry(ser)
                if hw:
                    hw_ts, hf0, hf1, hr0, hr1 = hw
                    print(f"\rHW: {hf0:6.2f}N {hf1:6.2f}N | {hr0:7.1f}mm {hr1:7.1f}mm | ArUco: {current_marker_id} ", end="", flush=True)
                    
                    records.append({
                        "global_timestamp_s": global_ts,
                        "aruco_id": current_marker_id,
                        "cmd_force_0_N": f0,
                        "cmd_force_1_N": f1,
                        "meas_force_0_N": hf0,
                        "meas_force_1_N": hf1,
                        "meas_length_0_mm": hr0,
                        "meas_length_1_mm": hr1,
                        "hw_timestamp_us": hw_ts
                    })

                # 5) Step simulation & sync viewer
                mj.mj_step(model, data)
                v.sync()

                dt = model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)

    except KeyboardInterrupt:
        print("\nAufzeichnung abgebrochen durch Benutzer.")

    finally:
        # Cleanup
        cv2.destroyAllWindows()
        send_cmd(ser, "stop")
        print("\nMotoren gestoppt.")
        ser.close()
        print("Serielle Verbindung geschlossen.")

        # Save data
        if records:
            df = pd.DataFrame(records)
            out_dir = Path(__file__).resolve().parent / "build"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "recorded_sys_id_data.csv"
            df.to_csv(out_file, index=False)
            print(f"Daten ({len(df)} Einträge) erfolgreich gespeichert nach:\n{out_file}")
        else:
            print("\nKeine Daten aufgezeichnet.")

if __name__ == "__main__":
    main()
