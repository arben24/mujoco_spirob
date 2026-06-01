#!/usr/bin/env python3
"""
SpiRob – Synced Data Visualization

Reads the merged data (sys_id_combined.parquet) and the original video,
and draws the joint angles, measured forces, timestamps, and ArUco markers 
back into the frames. Generates a new synchronized video file.
"""

import cv2
import polars as pl
from pathlib import Path
import time
import sys
import numpy as np

def main():
    base_dir = Path(__file__).resolve().parent
    video_path = base_dir / "Video" / "GX010070.MP4"
    parquet_path = base_dir / "build" / "sys_id_auto_GX010070.parquet"
    output_path = base_dir / "build" / "sys_id_synced_visualized_GX010070.mp4"

    if not video_path.exists():
        print(f"Fehler: Video nicht gefunden: {video_path}")
        sys.exit(1)
    if not parquet_path.exists():
        print(f"Fehler: Parquet-Datei nicht gefunden: {parquet_path}")
        sys.exit(1)

    print(f"Lade vorkalkulierte Daten aus:\n{parquet_path}")
    df = pl.read_parquet(parquet_path)
    
    # Für schnellen Zugriff auf Basis der Frame-Nummer
    frames_data = {row["frame"]: row for row in df.to_dicts()}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Fehler: Konnte Video nicht öffnen.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0: fps = 30.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Starte Visualisierung... gerendertes Video wird gespeichert unter:\n{output_path}")
    print(f"Insgesamt {total_frames} Frames zu verarbeiten.")
    
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        data = frames_data.get(frame_idx, {})
        
        # ── 1. Gelenkwinkel & Skeleton einzeichnen (aus Parquet-Daten)
        for joint_id in range(1, 14):
            angle_val = data.get(f"joint_{joint_id}_deg")
            px = data.get(f"x_{joint_id-1}")
            py = data.get(f"y_{joint_id-1}")
            cx = data.get(f"x_{joint_id}")
            cy = data.get(f"y_{joint_id}")
            
            if all(v is not None and not np.isnan(v) for v in [angle_val, px, py, cx, cy]):
                px, py, cx, cy = int(px), int(py), int(cx), int(cy)
                
                # Linie zwischen zwei Markern
                cv2.line(frame, (px, py), (cx, cy), (255, 0, 0), 2)
                
                # Mittelpunkt als Gelenk visualisieren
                mx, my = int((px + cx) / 2), int((py + cy) / 2)
                cv2.circle(frame, (mx, my), 5, (0, 0, 255), -1)
                
                # Winkel-Text
                text = f"J{joint_id}: {angle_val:.1f}deg"
                cv2.putText(frame, text, (mx + 10, my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ── 3. Sensor-Werte und Timestamps (als Overlay Text einzeichnen)
        vid_ts = data.get("video_timestamp_s", frame_idx / fps)
        global_ts = data.get("global_timestamp_s")
        f0 = data.get("meas_force_0_N")
        f1 = data.get("meas_force_1_N")
        
        # Textbox generieren
        info_lines = [
            f"Frame: {frame_idx} / {total_frames}",
            f"Video Time:  {vid_ts:.2f} s",
            f"Global Time: {global_ts:.2f} s" if global_ts is not None and not np.isnan(global_ts) else "Global Time: N/A"
        ]
        
        if f0 is not None and not np.isnan(f0):
            info_lines.append(f"Force 0: {f0:.2f} N")
        if f1 is not None and not np.isnan(f1):
            info_lines.append(f"Force 1: {f1:.2f} N")

        y_offset = 40
        for i, line in enumerate(info_lines):
            # Schwarzen Hintergrund für bessere Lesbarkeit
            cv2.putText(frame, line, (20, y_offset + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
            # Grüner Text drüber
            cv2.putText(frame, line, (20, y_offset + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        video_writer.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_idx / elapsed
            print(f"Gerendert: {frame_idx}/{total_frames} ({fps_proc:.1f} FPS)")

    video_writer.release()
    cap.release()
    print("\nVisualisierung erfolgreich beendet!")
    print(f"Video gespeichert als: {output_path}")

if __name__ == "__main__":
    main()
