import polars as pl
from pathlib import Path
import sys

class DataTrimmer:
    def __init__(self, input_path: str | Path, output_path: str | Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
    def trim_by_time(self, start_time_s: float = 0.0, end_time_s: float = None, reset_zero: bool = True):
        """
        Schneidet den Datensatz basierend auf der globalen Zeit (global_timestamp_s) zu.
        Setzt optional die Zähler (Zeit, Frames) wieder auf 0.
        """
        if not self.input_path.exists():
            raise FileNotFoundError(f"Eingabedatei nicht gefunden: {self.input_path}")
            
        print(f"Lese Daten von: {self.input_path}")
        df = pl.read_parquet(self.input_path)
        
        original_len = len(df)
        
        # Basis-Minima ermitteln, falls wir relativ zum Originalanfang bleiben wollen
        original_min_global = df["global_timestamp_s"].min()
        original_min_video = df["video_timestamp_s"].min()
        
        # 1. Filtern der Daten
        filter_expr = pl.lit(True)
        if start_time_s > 0:
            actual_start = original_min_global + start_time_s
            filter_expr = filter_expr & (pl.col("global_timestamp_s") >= actual_start)
            
        if end_time_s is not None:
            actual_end = original_min_global + end_time_s
            filter_expr = filter_expr & (pl.col("global_timestamp_s") <= actual_end)
            
        df = df.filter(filter_expr)
            
        new_len = len(df)
        print(f"Datensatz getrimmt: {original_len} -> {new_len} Zeilen")
        
        if new_len == 0:
            print("WARNUNG: Der resultierende Datensatz ist leer! Bitte Trim-Werte überprüfen.")
            return None
            
        # 2. Reset der Zeit und Frame-Zähler auf 0
        if reset_zero:
            print("Setze globale Zeit, Video-Zeit und Frame-IDs zurück auf 0...")
            df = df.with_columns([
                (pl.col("global_timestamp_s") - pl.col("global_timestamp_s").min()).alias("global_timestamp_s"),
                (pl.col("video_timestamp_s") - pl.col("video_timestamp_s").min()).alias("video_timestamp_s"),
                (pl.col("frame") - pl.col("frame").min()).alias("frame")
            ])
        else:
            print("Zeitachsen und Frame-IDs bleiben absolut (kein Reset auf 0).")
        
        # 3. Speichern
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(self.output_path)
        print(f"Gespeichert unter: {self.output_path}")

        return df

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_file = base_dir / "build" / "sys_id_combined.parquet"
    output_file = base_dir / "build" / "sys_id_trimmed_10_35.parquet"
    
    trimmer = DataTrimmer(input_file, output_file)
    
    # Konfiguration für das Trimmen:
    # START_ZEIT_SEKUNDEN und END_ZEIT_SEKUNDEN sind relativ zum START des ursprünglichen Datensatzes gemeint 
    # (also 0 = Anfang). END_ZEIT_SEKUNDEN = None trimmt nicht hinten ab.
    START_ZEIT_SEKUNDEN = 10.0 
    END_ZEIT_SEKUNDEN = 35.0
    
    try:
        df_result = trimmer.trim_by_time(start_time_s=START_ZEIT_SEKUNDEN, end_time_s=END_ZEIT_SEKUNDEN, reset_zero=True)
        if df_result is not None:
            print("\nVorschau der neuen (angepassten) Daten:")
            print(df_result.select(["frame", "global_timestamp_s", "video_timestamp_s"]).head())
    except Exception as e:
        print(f"Fehler: {e}")
        sys.exit(1)
