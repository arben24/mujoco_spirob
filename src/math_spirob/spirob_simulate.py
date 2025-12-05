import mujoco as mj
import numpy as np
import polars as pl
from typing import Dict, Any, List, Tuple, Callable
import math
import itertools
import json

# Definieren des Controller-Interface (Callback-Signatur)
# Ein Controller muss mj.MjModel, mj.MjData, die aktuelle Zeit (float) 
# und den Schrittindex (int) als Argumente akzeptieren.
ControllerFunc = Callable[[mj.MjModel, mj.MjData, float, int], None]

# --- 1. Helper-Funktionen (Datenverarbeitung) ---

def get_sliced_dict(data_dict: Dict[str, np.ndarray], final_length: int) -> Dict[str, np.ndarray]:
    """Schneidet alle Arrays im Dictionary auf die tatsächliche Länge (Synchronisation)."""
    return {name: arr[:final_length] for name, arr in data_dict.items()}

def create_single_polars_dataframe(
    sensor_groups: List[Tuple[str, Dict[str, np.ndarray]]], 
    time_data: np.ndarray, 
    final_length: int
) -> pl.DataFrame:
    """
    Erstellt ein einziges, breites Polars DataFrame aus allen Sensor-Gruppen.
    Verwendet die Spaltennamenskonvention: 'sensorname' für 1D und 'sensorname_X/Y/Z' für 3D.
    """
    
    all_columns: List[pl.Series] = [pl.Series("time_s", time_data)]
    
    for group_prefix, data_dict_global in sensor_groups:
        
        if not data_dict_global:
            print(f"Warnung: Gruppe '{group_prefix}' ist leer und wird ignoriert.")
            continue

        sliced_data_dict = get_sliced_dict(data_dict_global, final_length)
        
        for name, values_array in sliced_data_dict.items():
            
            # Bestimmung der Dimension (D) für diesen spezifischen Sensor
            current_D = values_array.shape[1] if values_array.ndim == 2 else 1
            
            if current_D == 3:
                # 3D: Benennung: sensorname_X/Y/Z
                columns_prefix = name
                all_columns.append(pl.Series(f"{columns_prefix}_X", values_array[:, 0]))
                all_columns.append(pl.Series(f"{columns_prefix}_Y", values_array[:, 1]))
                all_columns.append(pl.Series(f"{columns_prefix}_Z", values_array[:, 2]))
            
            elif current_D == 1:
                # 1D: Benennung: sensorname
                column_name = name
                all_columns.append(pl.Series(column_name, values_array.squeeze()))
            
            else:
                # Optionale Warnung für unbekannte Dimensionen
                print(f"Warnung: Sensor '{name}' in Gruppe '{group_prefix}' hat Dimension {current_D} und wird ignoriert.")
                
    return pl.DataFrame(all_columns)

# --- 2. Kernsimulationsfunktion ---

def initialize_data_structures(model: mj.MjModel, sim_time: float) -> Tuple[Dict, Dict, Dict, List, np.ndarray, int]:
    """Initialisiert alle Arrays und Metadaten vor der Simulation."""
    
    num_steps = int(sim_time / model.opt.timestep) + 1

    # Dictionaries für die Sensor-Zeitreihen
    acc_over_time, gyro_over_time, tendon_frc_over_time, tendon_pos_over_time, \
    tendon_vel_over_time, joint_pos_over_time, joint_vel_over_time = {}, {}, {}, {}, {}, {}, {}
    positions_over_time = {} # Für Geoms
    
    SENSOR_CONFIG = {
        mj.mjtSensor.mjSENS_ACCELEROMETER:    ('acc',    acc_over_time),
        mj.mjtSensor.mjSENS_GYRO:             ('gyro',   gyro_over_time),
        mj.mjtSensor.mjSENS_TENDONACTFRC:     ('tendon_frc', tendon_frc_over_time),
        mj.mjtSensor.mjSENS_TENDONPOS:        ('tendon_pos', tendon_pos_over_time),
        mj.mjtSensor.mjSENS_TENDONVEL:        ('tendon_vel', tendon_vel_over_time),
        mj.mjtSensor.mjSENS_JOINTPOS:         ('joint_pos',  joint_pos_over_time),
        mj.mjtSensor.mjSENS_JOINTVEL:         ('joint_vel',  joint_vel_over_time),
    }

    sensor_metadata = []
    
    # 1. Sensoren initialisieren
    for i in range(model.nsensor):
        sensor_type = model.sensor_type[i]
        
        if sensor_type in SENSOR_CONFIG:
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)
            _, data_dict = SENSOR_CONFIG[sensor_type]
            dim = model.sensor_dim[i] 
            
            time_series_array = np.zeros((num_steps, dim), dtype=np.float64)
            data_dict[name] = time_series_array 
            
            sensor_metadata.append({
                'name': name,
                'array': time_series_array, 
                'index': i                  
            })

    # 2. Geoms initialisieren
    geom_metadata = []
    i = 0
    while True:
        name = f"g_{i}"
        geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
        if geom_id == -1:
            break
        
        pos_array = np.zeros((num_steps, 3), dtype=np.float64)
        positions_over_time[name] = pos_array
        
        geom_metadata.append({
            'name': name,
            'array': pos_array,
            'id': geom_id
        })
        i += 1
        
    time_array = np.zeros(num_steps, dtype=np.float64)

    sensor_dicts = {
        "acc": acc_over_time, "gyro": gyro_over_time, "tendon_frc": tendon_frc_over_time, 
        "tendon_pos": tendon_pos_over_time, "tendon_vel": tendon_vel_over_time, 
        "joint_pos": joint_pos_over_time, "joint_vel": joint_vel_over_time, 
        "geom_pos": positions_over_time
    }

    return sensor_dicts, sensor_metadata, geom_metadata, time_array, num_steps

def run_simulation_and_get_dataframe(
    model: mj.MjModel, 
    data: mj.MjData, 
    sim_time: float, 
    controller: ControllerFunc,
    include_geom_pos: bool = False
) -> pl.DataFrame:
    """
    Führt die Simulation aus, sammelt Daten und konvertiert sie in ein Polars DataFrame.
    """
    
    # Initialisiere alle Speicherstrukturen
    sensor_dicts, sensor_metadata, geom_metadata, time_array, num_steps = \
        initialize_data_structures(model, sim_time)
        
    # --- 1. Simulation ---
    
    step_index = 0
    
    # Zustand t=0 speichern
    time_array[step_index] = data.time
    for meta in sensor_metadata:
        meta['array'][step_index] = data.sensor(meta['index']).data
    if include_geom_pos:
        for meta in geom_metadata:
            meta['array'][step_index] = data.geom_xpos[meta['id']]
        
    steps_to_run = num_steps - 1

    for _ in range(steps_to_run):
            
        step_index += 1
        
        # --- CALL THE EXTERNAL CONTROLLER ---
        controller(model, data, data.time, step_index) 
        
        # --- Simulationsschritt ---
        mj.mj_step(model, data)
        
        # --- Datenspeicherung ---
        time_array[step_index] = data.time
        for meta in sensor_metadata:
            meta['array'][step_index] = data.sensor(meta['index']).data
        if include_geom_pos:
            for meta in geom_metadata:
                meta['array'][step_index] = data.geom_xpos[meta['id']]

    final_length = step_index + 1
    time_series_data = time_array[:final_length] 
    
    # --- 2. Polars Konvertierung ---
    
    SENSOR_GROUPS_CONFIG: List[Tuple[str, Dict[str, np.ndarray]]] = [
        ("acc", sensor_dicts["acc"]),
        ("gyro", sensor_dicts["gyro"]),
        ("tendon_frc", sensor_dicts["tendon_frc"]),
        ("tendon_pos", sensor_dicts["tendon_pos"]),
        ("tendon_vel", sensor_dicts["tendon_vel"]),
        ("joint_pos", sensor_dicts["joint_pos"]),
        ("joint_vel", sensor_dicts["joint_vel"]),
    ]
    
    if include_geom_pos:
         SENSOR_GROUPS_CONFIG.append(("geom_pos", sensor_dicts["geom_pos"]))

    final_wide_df = create_single_polars_dataframe(
        SENSOR_GROUPS_CONFIG, 
        time_series_data, 
        final_length
    )
    
    return final_wide_df

# --- 3. Controller-Templates (Beispiele) ---

def static_controller(model: mj.MjModel, data: mj.MjData, current_time: float, step_index: int):
    """Setzt eine konstante Seilkraft (0.2) auf den ersten Aktuator."""
    # Beispiel: Nur den ersten Aktuator setzen

    data.ctrl[0] = 0.2

def sine_controller(model: mj.MjModel, data: mj.MjData, current_time: float, step_index: int):
    """Setzt eine sinusförmige Kraft (Amplitude 0.5, Frequenz 0.5 Hz) auf den ersten Aktuator."""

    amplitude = 0.5
    frequency = 2.0 * math.pi * 0.5 
    data.ctrl[0] = amplitude * np.sin(frequency * current_time)


def generate_grid_configs(variable_params: Dict[str, Any], fixed_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generiert eine Liste von Konfigurations-Dictionaries aus einem Grid."""
    
    # Trenne die Schlüssel und Werte, um sie itertools.product zu übergeben
    keys = list(variable_params.keys())
    values = [
        # Wenn der Wert ein Dictionary ist (wie bei 'controller'), verwende dessen Werte
        list(v.values()) if isinstance(v, dict) else v
        for v in variable_params.values()
    ]
    
    # Extrahieren der Controllernamen separat (für die ID-Generierung)
    controller_names = list(variable_params.get("controller", {}).keys())

    SIM_CONFIGS = []
    run_counter = 1
    
    # itertools.product erzeugt alle Kombinationen der Werte
    for combination in itertools.product(*values):
        
        # Erstelle ein Dict aus der aktuellen Kombination
        config = dict(zip(keys, combination))
        
        # Hinzufügen der festen Parameter
        config.update(fixed_params)
        
        # Wenn 'controller' im Grid ist, finde den passenden Namen für die ID
        ctrl_name = "Custom"
        if "controller" in keys:
            # Finde den Namen des Controllers anhand seines Funktions-/Objektwerts
            # (Dies ist etwas komplex, da man den Wert zurück auf den Schlüssel mappen muss)
            # Wir machen es einfacher, indem wir annehmen, dass 'controller' die letzte Variable ist:
            if isinstance(variable_params["controller"], dict):
                ctrl_value = config["controller"]
                
                # Finde den Namen, der zum Wert gehört
                ctrl_name = next((name for name, func in variable_params["controller"].items() if func == ctrl_value), "Unknown")
            
        # Generiere die eindeutige ID
        id_parts = [
            ctrl_name,
            f"L{config['L_target']:.2f}",
            f"T{config['sim_time']:.1f}",
            f"d{config['base_d']:.3f}",
            # ... füge weitere wichtige Parameter hinzu
        ]
        config["id"] = f"Run_{run_counter:03d}_{'_'.join(id_parts)}"
        
        SIM_CONFIGS.append(config)
        run_counter += 1
        
    return SIM_CONFIGS


def print_configs_formatted(config_list: List[Dict[str, Any]], preview_limit: int = 5,print_all: bool = False):
    """
    Gibt eine formatierte Vorschau der Konfigurationsliste auf der Konsole aus.
    """
    
    print("\n" + "="*50)
    print(f"📄 VORSCHAU DER KONFIGURATIONEN ({len(config_list)} Läufe) 📄")
    print("="*50)

    if print_all:
        preview_limit = len(config_list)
    
    for i, config in enumerate(config_list):
        if i >= preview_limit:
            print(f"  ... und {len(config_list) - i} weitere Konfigurationen.")
            break
            
        # Extrahieren des Controllernamens
        ctrl = config['controller']
        ctrl_name = ctrl.__name__ if hasattr(ctrl, '__name__') else str(ctrl)

        # Gib die wichtigsten Parameter aus
        print(f"[{i+1}/{len(config_list)}] ID: {config['id']}")
        print(f"  L_target: {config['L_target']:.2f}, Time: {config['sim_time']:.1f}")
        print(f"  base_d: {config['base_d']:.3f}, tip_d: {config['tip_d']:.3f}, Delta_theta_deg: {config['Delta_theta_deg']}")
        print(f"  Controller: {ctrl_name}")
        
    print("\n" + "="*50)

def save_configs_to_json(
    config_list: List[Dict[str, Any]], 
    filename: str = "simulation_configs.json", 
    indent: int = 4
):
    """
    Exportiert die Konfigurationsliste in eine JSON-Datei.
    Nicht-serialisierbare Objekte (Controller-Funktionen/Instanzen) werden in Strings konvertiert.
    
    Args:
        config_list: Die Liste der Konfigurations-Dictionaries (SIM_CONFIGS).
        filename: Der Name der Exportdatei.
        indent: Die Anzahl der Leerzeichen für die JSON-Einrückung.
    """
    
    exportable_list = []
    for config in config_list:
        # Erstelle eine Kopie, um das Original-Config-Dict nicht zu verändern
        export_config = config.copy()
        
        # Konvertiere den Controller-Funktions-/Objektwert in einen String
        ctrl = export_config['controller']
        ctrl_str = ctrl.__name__ if hasattr(ctrl, '__name__') else str(ctrl)
        
        # Füge den String-Info-Wert hinzu
        export_config['controller_info_str'] = ctrl_str
        
        # Entferne den nicht-serialisierbaren Teil
        del export_config['controller']
        
        exportable_list.append(export_config)

    try:
        with open(filename, 'w') as f:
            json.dump(exportable_list, f, indent=indent)
        print(f"\n✅ ERFOLG: Konfigurationen erfolgreich exportiert nach: {filename}")
    except Exception as e:
        print(f"\n❌ FEHLER beim Exportieren nach JSON: {e}")
