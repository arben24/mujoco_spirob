import mujoco as mj
import mujoco.viewer as viewer
import time
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import math_spirob.spirob_generator as sg
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum, auto

# ---------------------------------------------------------
# 1. KONFIGURATION & KONSTANTEN
# ---------------------------------------------------------
USE_VIEWER = False          
REALTIME = False            
VIEWER_PASSIVE = True      
SIM_TIME = 2.0             

# Welche Daten sollen geplottet werden?
PLOT_CONFIG = {
    "ACC": True,
    "GYRO": True,
    "TENDON_FRC": True,
    "JOINT_POS": True,
    "FORCE_LOCAL": True, # Kontaktkräfte
    "GEOM_POS": True    # Geometrie-Positionen (meist uninteressant zu plotten)
}

# ---------------------------------------------------------
# 2. DATENSTRUKTUREN (CLEAN CODE SETUP)
# ---------------------------------------------------------

class DataGroup(Enum):
    """Definiert Kategorien für Zeitreihen-Daten."""
    ACC = "acc"
    GYRO = "gyro"
    TENDON_FRC = "tendon_frc"
    TENDON_POS = "tendon_pos"
    TENDON_VEL = "tendon_vel"
    JOINT_POS = "joint_pos"
    JOINT_VEL = "joint_vel"
    # Eigene berechnete Gruppen
    GEOM_POS = "geom_pos"
    FORCE_LOCAL = "force_local"
    MOMENT_LOCAL = "moment_local"

# Mapping von MuJoCo Sensor-Typen zu unseren Gruppen
SENSOR_MAPPING = {
    mj.mjtSensor.mjSENS_ACCELEROMETER:    DataGroup.ACC,
    mj.mjtSensor.mjSENS_GYRO:             DataGroup.GYRO,
    mj.mjtSensor.mjSENS_TENDONACTFRC:     DataGroup.TENDON_FRC,
    mj.mjtSensor.mjSENS_TENDONPOS:        DataGroup.TENDON_POS,
    mj.mjtSensor.mjSENS_TENDONVEL:        DataGroup.TENDON_VEL,
    mj.mjtSensor.mjSENS_JOINTPOS:         DataGroup.JOINT_POS,
    mj.mjtSensor.mjSENS_JOINTVEL:         DataGroup.JOINT_VEL,
}

@dataclass
class TimeData:
    """
    Repräsentiert eine einzelne Zeitreihe (z.B. ein Sensor oder ein Geom-Wert).
    Hält Metadaten und das Speicher-Array zusammen.
    """
    name: str                       # Name (z.B. 'sensor_A' oder 'g_0')
    group: DataGroup                # Zu welcher Gruppe gehört das (z.B. ACC)
    array: np.ndarray               # Das Speicher-Array (Nx1 oder Nx3)
    mujoco_id: int = -1             # ID in MuJoCo (Sensor-ID oder Geom-ID)
    dim: int = 1                    # Dimension (1 oder 3)

# ---------------------------------------------------------
# 3. HILFSFUNKTIONEN
# ---------------------------------------------------------

def get_geom_contact_forces(
    model: mj.MjModel, 
    data: mj.MjData, 
    target_geom_ids: Dict[int, str]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Berechnet summierte Kontaktkräfte und Momente pro Geom.
    Args:
        target_geom_ids: Dictionary {geom_id: geom_name} der zu überwachenden Geoms.
    """
    results = {}
    
    # Puffer für MuJoCo Funktion
    c_force_vector = np.zeros(6, dtype=np.float64)

    for i in range(data.ncon):
        contact = data.contact[i]
        
        # Prüfen, ob eines der beteiligten Geoms für uns interessant ist
        geom_id = -1
        if contact.geom1 in target_geom_ids:
            geom_id = contact.geom1
        elif contact.geom2 in target_geom_ids:
            geom_id = contact.geom2
            
        if geom_id != -1:
            # Kraft im Kontakt-Frame berechnen
            mj.mj_contactForce(model, data, i, c_force_vector)
            
            # [Fx, Fy, Fz, Tx, Ty, Tz]
            force = c_force_vector[0:3]
            moment = c_force_vector[3:6]
            
            name = target_geom_ids[geom_id]
            
            if name not in results:
                results[name] = {
                    'force': np.zeros(3, dtype=np.float64), 
                    'moment': np.zeros(3, dtype=np.float64)
                }
            
            results[name]['force'] += force
            results[name]['moment'] += moment

    return results

# ---------------------------------------------------------
# 4. MODELL ERSTELLUNG
# ---------------------------------------------------------

xml_string = sg.generate_xml_string(
    L_target=0.30, base_d=0.06, tip_d=0.01, Delta_theta_deg=30,
    model_name="spiral_chain_plot", auto_format=True
)
print("XML generiert.")

spec = mj.MjSpec.from_string(xml_string)
model = spec.compile()
data = mj.MjData(model)

# ---------------------------------------------------------
# 5. INITIALISIERUNG & SPEICHER-ALLOKATION
# ---------------------------------------------------------

num_steps = int(SIM_TIME / model.opt.timestep) + 1
print(f"Allokiere Speicher für {num_steps} Schritte...")

# Zentrale Liste für ALLE Zeitreihen (Sensoren + Geoms)
all_metadata: List[TimeData] = []
# Hilfs-Map für schnellen Zugriff auf Geom-Namen via ID (für Kontaktberechnung)
geom_id_to_name_map: Dict[int, str] = {}

# A) Sensoren initialisieren
for i in range(model.nsensor):
    stype = model.sensor_type[i]
    if stype in SENSOR_MAPPING:
        group = SENSOR_MAPPING[stype]
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)
        dim = model.sensor_dim[i]
        
        all_metadata.append(TimeData(
            name=name, group=group, mujoco_id=i, dim=dim,
            array=np.zeros((num_steps, dim), dtype=np.float64)
        ))

# B) Geoms initialisieren (Position, Kraft, Moment)
i = 0
while True:
    name = f"g_{i}"
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
    if gid == -1: break
    
    geom_id_to_name_map[gid] = name
    
    # 1. Position
    all_metadata.append(TimeData(
        name=name, group=DataGroup.GEOM_POS, mujoco_id=gid, dim=3,
        array=np.zeros((num_steps, 3), dtype=np.float64)
    ))
    # 2. Lokale Kraft
    all_metadata.append(TimeData(
        name=name, group=DataGroup.FORCE_LOCAL, mujoco_id=gid, dim=3,
        array=np.zeros((num_steps, 3), dtype=np.float64)
    ))
    # 3. Lokales Moment
    all_metadata.append(TimeData(
        name=name, group=DataGroup.MOMENT_LOCAL, mujoco_id=gid, dim=3,
        array=np.zeros((num_steps, 3), dtype=np.float64)
    ))
    i += 1

print(f"Metadaten initialisiert: {len(all_metadata)} Zeitreihen tracked.")
time_array = np.zeros(num_steps, dtype=np.float64)

# ---------------------------------------------------------
# 6. SIMULATION
# ---------------------------------------------------------

def run_simulation_steps(steps_count, start_index):
    """Führt N Schritte aus und füllt die Arrays."""
    idx = start_index
    for _ in range(steps_count):
        idx += 1
        
        # Physics Step
        data.ctrl[0] = 0.2
        mj.mj_step(model, data)
        
        # --- DATEN SPEICHERN ---
        time_array[idx] = data.time
        
        # 1. Kontakte berechnen (nur einmal pro Step)
        current_contacts = get_geom_contact_forces(model, data, geom_id_to_name_map)
        
        # 2. Durch alle Metadaten iterieren und Arrays füllen
        for meta in all_metadata:
            
            # -- Standard Sensoren --
            if meta.group in SENSOR_MAPPING.values():
                meta.array[idx] = data.sensor(meta.mujoco_id).data
            
            # -- Geom Position --
            elif meta.group == DataGroup.GEOM_POS:
                meta.array[idx] = data.geom_xpos[meta.mujoco_id]
                
            # -- Kontaktkräfte --
            elif meta.group == DataGroup.FORCE_LOCAL:
                if meta.name in current_contacts:
                    meta.array[idx] = current_contacts[meta.name]['force']
                # Sonst bleibt es 0.0 (durch Initialisierung)
                
            elif meta.group == DataGroup.MOMENT_LOCAL:
                if meta.name in current_contacts:
                    meta.array[idx] = current_contacts[meta.name]['moment']

    return idx

# --- HAUPTABLAUF ---

if not USE_VIEWER:
    print(f"Simulation (Headless) startet für {SIM_TIME}s...")
    start_t = time.time()
    
    # t=0 speichern (Initialzustand)
    time_array[0] = data.time
    # Hinweis: Bei t=0 sind Kräfte 0 und Sensoren ggf. auch, wir lassen es bei 0 stehen
    
    final_step_idx = run_simulation_steps(num_steps - 1, 0)
    
    print(f"Fertig in {time.time() - start_t:.4f}s.")

else:
    print("Simulation mit Viewer...")
    launch_fn = viewer.launch_passive if VIEWER_PASSIVE else viewer.launch
    with launch_fn(model, data) as v:
        start_t = time.time()
        curr_idx = 0
        while v.is_running() and time.time() - start_t < SIM_TIME:
            step_start = time.time()
            
            # Führe einen Schritt aus (hier vereinfacht direkt im Loop)
            data.ctrl[0] = 0.2
            mj.mj_step(model, data)
            v.sync()
            
            # Daten speichern wäre hier analog zum headless mode nötig, 
            # wird oft im Viewer-Modus weggelassen um Performance zu sparen.
            
            if REALTIME:
                dt = model.opt.timestep - (time.time() - step_start)
                if dt > 0: time.sleep(dt)
        final_step_idx = 0 # Dummy für Viewer Mode

# ---------------------------------------------------------
# 7. DATEN-EXPORT (POLARS) & VISUALISIERUNG
# ---------------------------------------------------------

if not USE_VIEWER:
    
    # Auf tatsächliche Länge kürzen
    valid_len = final_step_idx + 1
    t_data = time_array[:valid_len]
    
    print("Erstelle DataFrame...")
    
    # Polars Columns erstellen
    cols = [pl.Series("time_s", t_data)]
    
    for meta in all_metadata:
        # Array kürzen
        arr = meta.array[:valid_len]
        
        # Spaltennamen generieren: Gruppe_Name[_Achse]
        base_name = f"{meta.group.value}_{meta.name}"
        
        if meta.dim == 1:
            cols.append(pl.Series(base_name, arr.squeeze()))
        elif meta.dim == 3:
            cols.append(pl.Series(f"{base_name}_X", arr[:, 0]))
            cols.append(pl.Series(f"{base_name}_Y", arr[:, 1]))
            cols.append(pl.Series(f"{base_name}_Z", arr[:, 2]))
            
    df = pl.DataFrame(cols)
    
    print(f"DataFrame erstellt: {df.shape[0]} Zeilen, {df.shape[1]} Spalten.")
    # print(df.head())

    # --- PLOTTING ---
    
    def plot_group(group_enum: DataGroup, title: str):
        """Plottet alle Daten einer bestimmten Gruppe."""
        # Filtere relevante Metadaten
        group_metas = [m for m in all_metadata if m.group == group_enum]
        if not group_metas: return

        # Check Dimension des ersten Elements
        is_3d = (group_metas[0].dim == 3)
        
        if is_3d:
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f"{title} (3D)", fontsize=14)
            axes_labels = ["X", "Y", "Z"]
            
            for i, ax_lbl in enumerate(axes_labels):
                for meta in group_metas:
                    # Nur plotten wenn Daten nicht komplett null sind (optional)
                    if np.max(np.abs(meta.array[:valid_len, i])) > 1e-6:
                        axs[i].plot(t_data, meta.array[:valid_len, i], label=meta.name)
                axs[i].set_ylabel(ax_lbl)
                axs[i].grid(True, alpha=0.3)
                # Legende nur oben, wenn nicht zu viele
                if i == 0 and len(group_metas) < 15: axs[i].legend(fontsize='x-small', ncol=2)
            axs[-1].set_xlabel("Zeit (s)")
            
        else:
            plt.figure(figsize=(10, 5))
            plt.title(f"{title} (1D)")
            for meta in group_metas:
                 if np.max(np.abs(meta.array[:valid_len])) > 1e-6:
                    plt.plot(t_data, meta.array[:valid_len].squeeze(), label=meta.name)
            plt.xlabel("Zeit (s)")
            plt.ylabel("Wert")
            plt.grid(True, alpha=0.3)
            if len(group_metas) < 15: plt.legend(fontsize='x-small')

    # Plots generieren basierend auf Config
    if PLOT_CONFIG["ACC"]: plot_group(DataGroup.ACC, "Beschleunigung")
    if PLOT_CONFIG["GYRO"]: plot_group(DataGroup.GYRO, "Gyroskop")
    if PLOT_CONFIG["TENDON_FRC"]: plot_group(DataGroup.TENDON_FRC, "Seilkräfte")
    if PLOT_CONFIG["GEOM_POS"]: plot_group(DataGroup.GEOM_POS, "Geom Positionen")
    if PLOT_CONFIG["JOINT_POS"]: plot_group(DataGroup.JOINT_POS, "Gelenkwinkel")
    if PLOT_CONFIG["FORCE_LOCAL"]: plot_group(DataGroup.FORCE_LOCAL, "Kontaktkräfte (Lokal)")
    
    plt.show()