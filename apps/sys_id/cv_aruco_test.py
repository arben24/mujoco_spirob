import cv2
import polars as pl
from pathlib import Path
import time
import sys
import numpy as np

class ArucoTracker:
    def __init__(self, video_path: str | Path, output_path: str | Path):
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.chain_ids = set(range(14))
        
    def process_video(self):
        if not self.video_path.exists(): raise FileNotFoundError()
            
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened(): raise ValueError()
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0: fps = 30.0
        
        print(f"Starte Tracking: {self.video_path.name} | FPS: {fps:.1f} | Frames gesamt: {total_frames}")
        
        records = []
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret: break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = self.detector.detectMarkers(gray)
            timestamp = frame_idx / fps
            
            if ids is not None:
                seen_markers = set()
                for i, marker_id in enumerate(ids.flatten()):
                    mid = int(marker_id)
                    if mid in self.chain_ids and mid not in seen_markers:
                        seen_markers.add(mid)
                        c = corners[i][0]
                        dx = float(c[1, 0] - c[0, 0] + c[2, 0] - c[3, 0])
                        dy = float(c[1, 1] - c[0, 1] + c[2, 1] - c[3, 1])
                        angle = np.degrees(np.arctan2(dy, dx))
                        
                        cx = float(c[:, 0].mean())
                        cy = float(c[:, 1].mean())
                        
                        records.append({
                            "frame": frame_idx,
                            "timestamp": timestamp,
                            "marker_id": int(marker_id),
                            "x": cx,
                            "y": cy,
                            "angle_deg": angle
                        })
            
            frame_idx += 1
            if frame_idx % 250 == 0:
                elapsed = time.time() - start_time
                print(f"Verarbeitet: {frame_idx}/{total_frames} ({frame_idx/elapsed:.1f} FPS)")
                
        cap.release()
        
        if records:
            df_long = pl.DataFrame(records, schema={
                "frame": pl.UInt32,
                "timestamp": pl.Float64,
                "marker_id": pl.Int32,
                "x": pl.Float64,
                "y": pl.Float64,
                "angle_deg": pl.Float64
            })
            
            df_wide = df_long.pivot(on="marker_id", index=["frame", "timestamp"], values=["x", "y", "angle_deg"])
            
            df_frames = pl.DataFrame({"frame": range(frame_idx)}, schema={"frame": pl.UInt32})
            df_wide = df_frames.join(df_wide, on="frame", how="left")
            df_wide = df_wide.with_columns(pl.col("timestamp").fill_null(pl.col("frame") / fps))

            marker_cols = [c for c in df_wide.columns if c not in ["frame", "timestamp"]]
            if marker_cols:
                df_wide = df_wide.with_columns([
                    pl.col(c).interpolate().forward_fill().backward_fill() for c in marker_cols
                ])

            joint_exprs = []
            for i in range(1, 14):
                prev_ang = f"angle_deg_{i-1}"
                curr_ang = f"angle_deg_{i}"
                if prev_ang in df_wide.columns and curr_ang in df_wide.columns:
                    joint_ang = (pl.col(curr_ang) - pl.col(prev_ang) + 180) % 360 - 180
                    joint_exprs.append(joint_ang.alias(f"joint_{i}_deg"))
            
            if joint_exprs:
                df_wide = df_wide.with_columns(joint_exprs)

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            df_wide.write_parquet(self.output_path)
            
            print("\nTracking erfolgreich abgeschlossen!")
            print(f"Daten gespeichert unter: {self.output_path}")
            print(df_wide.head())
        else:
            print("\nWarnung: Keine Marker im Video gefunden.")

    def visualize_tracking(self, output_video_path: str | Path):
        output_video_path = Path(output_video_path)
        if not self.video_path.exists() or not self.output_path.exists():
            return
            
        print(f"\nStarte Visualisierung (inkl. interpolierter Marker)... -> {output_video_path}")
        df = pl.read_parquet(self.output_path)
        frames_data = {row["frame"]: row for row in df.to_dicts()}
        
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0: fps = 30.0
        
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret: break
                
            data = frames_data.get(frame_idx, {})
            for joint_id in range(1, 14):
                angle_val = data.get(f"joint_{joint_id}_deg")
                px = data.get(f"x_{joint_id-1}")
                py = data.get(f"y_{joint_id-1}")
                cx = data.get(f"x_{joint_id}")
                cy = data.get(f"y_{joint_id}")
                
                if all(v is not None and not np.isnan(v) for v in [angle_val, px, py, cx, cy]):
                    px, py, cx, cy = int(px), int(py), int(cx), int(cy)
                    cv2.line(frame, (px, py), (cx, cy), (255, 0, 0), 2)
                    mx, my = int((px + cx) / 2), int((py + cy) / 2)
                    cv2.circle(frame, (mx, my), 5, (0, 0, 255), -1)
                    text = f"J{joint_id}: {angle_val:.1f}deg"
                    cv2.putText(frame, text, (mx + 10, my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            video_writer.write(frame)
            frame_idx += 1
            if frame_idx % 250 == 0:
                elapsed = time.time() - start_time
                print(f"Visualisierung gerendert: {frame_idx}/{total_frames} ({frame_idx/elapsed:.1f} FPS)")
                
        video_writer.release(); cap.release()
        print("Visualisierung erfolgreich erstellt!")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    video_file = base_dir / "Video" / "GX010059.MP4" 
    output_file = base_dir / "build" / "aruco_tracking.parquet"
    output_video_file = base_dir / "build" / "aruco_visualized.mp4"
    tracker = ArucoTracker(video_file, output_file)
    try:
        tracker.process_video()
        tracker.visualize_tracking(output_video_file) # Auskommentieren oder aktivieren je nach Bedarf
    except Exception as e:
        print(f"Fehler: {e}")
        sys.exit(1)
