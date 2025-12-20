import mujoco as mj
import mujoco.viewer as viewer
import numpy as np
import polars as pl
from typing import Dict, Any, List, Tuple, Callable
import math
import itertools
import json
import time

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
    enable_viewer: bool,
    boost_viewer: float,
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

    if enable_viewer:
        with mj.viewer.launch_passive(model, data) as viewer:
            # Wir nutzen data.time für die Abbruchbedingung, 
            # damit wir genau sim_time Sekunden physikalischer Zeit simulieren
            while viewer.is_running() and data.time < sim_time:
                step_start = time.time()

                step_index += 1
                
                # --- CALL THE EXTERNAL CONTROLLER ---
                controller(model, data, data.time, step_index) 
                
                # --- Simulationsschritt ---
                mj.mj_step(model, data)
                
                # --- Datenspeicherung ---
                # Sicherheitscheck, damit wir nicht über das Array-Ende schreiben
                if step_index < len(time_array):
                    time_array[step_index] = data.time
                    for meta in sensor_metadata:
                        meta['array'][step_index] = data.sensor(meta['index']).data
                    if include_geom_pos:
                        for meta in geom_metadata:
                            meta['array'][step_index] = data.geom_xpos[meta['id']]

                # GUI aktualisieren
                viewer.sync()

                # --- Zeitsteuerung mit Boost ---
                # Wir teilen den physikalischen Zeitschritt durch den Boost-Faktor
                target_step_duration = model.opt.timestep / boost_viewer
                elapsed_time = time.time() - step_start
                
                time_until_next_step = target_step_duration - elapsed_time
                
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    else:
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

    #data.ctrl[0] = 0.2
    data.ctrl[1] = 0.3

def ramped_controller(model: mj.MjModel, data: mj.MjData, current_time: float, step_index: int):
    """Setzt eine linear ansteigende Kraft (0.0 bis 1.0 über 10 Sekunden) auf den ersten Aktuator."""
    max_time = 10.0
    max_force = 1.0
    force = (current_time / max_time) * max_force
    force = min(force, max_force)  # Begrenze auf max_force
    data.ctrl[1] = force

def sine_controller(model: mj.MjModel, data: mj.MjData, current_time: float, step_index: int):
    """Setzt eine sinusförmige Kraft (Amplitude 0.5, Frequenz 0.5 Hz) auf den ersten Aktuator."""

    amplitude = 0.5
    frequency = 2.0 * math.pi * 0.5 
    data.ctrl[0] = amplitude * np.sin(frequency * current_time)

# Diese Funktionen werden SPÄTER im Loop aufgerufen
def setup_cylinder(worldbody, pos, size, euler, **kwargs):
    body = worldbody.add_body(name="cylinder_obj", pos=pos)
    print(f"  Erstelle Zylinder mit Größe {size} an Position {pos} mit Euler {euler}")
    body.add_geom(
        name="cyl_geom",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        size=size,  # Erwartet [radius, half_length, unused]
        euler=euler,
        rgba=[0.2, 0.8, 0.5, 1],
        density=1000
    )

def setup_box(worldbody, pos, size, euler, **kwargs):
    body = worldbody.add_body(name="box_obj", pos=pos)
    print(f"  Erstelle Box mit Größe {size} an Position {pos} mit Euler {euler}")
    body.add_geom(
        name="box_geom",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=size,  # Erwartet [x_half, y_half, z_half]
        euler=euler,
        rgba=[0.8, 0.2, 0.2, 1],
        density=1000
    )

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

def generate_hybrid_grid_configs(common_params, geom_scenarios, fixed_params):
    configs = []
    run_counter = 1

    # 1. Schritt: Erzeuge Grid für die gemeinsamen Parameter (L_target, controller, etc.)
    common_keys = list(common_params.keys())
    # Sonderbehandlung für Controller (wir wollen die Values, nicht Keys, aber Namen für ID)
    common_values = []
    for k, v in common_params.items():
        if k == "controller" and isinstance(v, dict):
            common_values.append(list(v.items())) # Speichert (Name, Funktion) Tupel
        else:
            common_values.append(v)

    # Iteriere über die Basis-Parameter
    for common_prod in itertools.product(*common_values):
        
        # Basis-Config Dictionary bauen
        base_config = {}
        ctrl_name_id = ""
        
        for i, key in enumerate(common_keys):
            val = common_prod[i]
            if key == "controller":
                # Tuple entpacken: (Name, Funktion)
                ctrl_name_id = val[0]
                base_config[key] = val[1]
            else:
                base_config[key] = val

        # 2. Schritt: Für JEDE Basis-Config, iteriere durch die Geometrie-Szenarien
        for scenario in geom_scenarios:
            geom_func = scenario["setup_func"]
            geom_name = scenario["obj_name"]
            
            # Hole die spezifischen Parameter für dieses Szenario (size, pos, euler)
            scen_params = scenario["params"]
            scen_keys = list(scen_params.keys())
            scen_values = list(scen_params.values())
            
            # Mini-Grid für dieses Szenario
            for geom_prod in itertools.product(*scen_values):
                
                # Kopiere Basis-Config, damit wir sie nicht überschreiben
                final_config = base_config.copy()
                final_config.update(fixed_params)
                
                # Füge Geometrie-Daten hinzu
                final_config["geom_func"] = geom_func
                
                # Geometrie-Parameter einzeln ins Config-Dict packen UND in ein 'geom_kwargs' Dict
                geom_kwargs = {}
                geom_id_parts = [geom_name]
                
                for i, key in enumerate(scen_keys):
                    val = geom_prod[i]
                    geom_kwargs[key] = val # Für den Funktionsaufruf später
                    
                    # ID Teil generieren (z.B. Size -> S0.1)
                    if key == "size":
                        s_str = "-".join([f"{x:.2f}" for x in val])
                        geom_id_parts.append(f"Sz{s_str}")
                    elif key == "pos":
                        # Optional, wenn Pos wichtig für ID ist
                        pass 

                final_config["geom_kwargs"] = geom_kwargs
                
                # ID erstellen
                id_parts = [
                    ctrl_name_id,
                    "_".join(geom_id_parts),
                    f"L{base_config['L_target']:.2f}",
                    f"T{base_config['sim_time']:.1f}"
                ]
                final_config["id"] = f"Run_{run_counter:03d}_{'_'.join(id_parts)}"
                
                configs.append(final_config)
                run_counter += 1
                
    return configs



def format_value_for_print(value: Any) -> str:
    """Konvertiert Listen/Arrays in einen kompakten, lesbaren String."""
    if isinstance(value, (list, tuple, np.ndarray)):
        # Runde Floats und konvertiere zu String: [0.10, 0.20]
        return "[" + ", ".join([f"{x:.2f}" for x in value]) + "]"
    
    if isinstance(value, float):
        return f"{value:.3f}"
        
    # Wenn es der String-Name der Funktion ist (z.B. 'setup_cylinder')
    if isinstance(value, str):
        return value.replace('setup_', '')
        
    return str(value)


def print_configs_formatted(config_list: List[Dict[str, Any]], preview_limit: int = 5, print_all: bool = False):
    """
    Gibt eine formatierte Vorschau der Konfigurationsliste auf der Konsole aus.
    Enthält nun Details zur variablen Geometrie.
    """
    
    print("\n" + "="*80)
    print(f"📄 VORSCHAU DER KONFIGURATIONEN ({len(config_list)} Läufe) 📄")
    print("="*80)

    if print_all:
        preview_limit = len(config_list)
    
    for i, config in enumerate(config_list):
        if i >= preview_limit:
            print(f"  ... und {len(config_list) - i} weitere Konfigurationen.")
            break
            
        # --- 1. Controller Name ---
        ctrl = config.get('controller')
        ctrl_name = ctrl.__name__ if hasattr(ctrl, '__name__') else str(ctrl)

        # --- 2. Geometrie-Informationen ---
        
        # Geometrie-Setup-Funktion
        geom_func = config.get('geom_func')
        geom_type_name = geom_func.__name__.replace('setup_', '') if hasattr(geom_func, '__name__') else "Unbekannt"
        
        # Geometrie-Parameter (pos, size, euler)
        geom_kwargs = config.get('geom_kwargs', {})
        
        geom_pos_str = format_value_for_print(geom_kwargs.get('pos', 'N/A'))
        geom_size_str = format_value_for_print(geom_kwargs.get('size', 'N/A'))
        geom_euler_str = format_value_for_print(geom_kwargs.get('euler', 'N/A'))
        
        # --- Ausgabe ---
        print(f"[{i+1}/{len(config_list)}] ID: {config['id']}")
        
        # Allgemeine Parameter
        print(f"  > Modell: L_target={config.get('L_target', 'N/A'):.2f}, base_d={config.get('base_d', 'N/A'):.3f}")
        print(f"  > Kontext: Time={config.get('sim_time', 'N/A'):.1f}, Controller={ctrl_name}")
        
        # Geometrie-Details
        print(f"  > OBJEKT ({geom_type_name.upper()}):")
        print(f"      Pos: {geom_pos_str}, Größe: {geom_size_str}, Euler: {geom_euler_str}")
        
    print("\n" + "="*80)

def save_configs_to_json(
    config_list: List[Dict[str, Any]], 
    filename: str = "simulation_configs.json", 
    indent: int = 4
):
    """
    Exportiert die Konfigurationsliste in eine JSON-Datei.
    Nicht-serialisierbare Objekte (Funktionen, NumPy-Arrays) werden in Strings 
    oder standardmäßige Python-Typen konvertiert.
    
    Args:
        config_list: Die Liste der Konfigurations-Dictionaries (SIM_CONFIGS).
        filename: Der Name der Exportdatei.
        indent: Die Anzahl der Leerzeichen für die JSON-Einrückung.
    """
    
    exportable_list = []
    
    for config in config_list:
        # Erstelle eine Kopie des Dictionarys, um das Original nicht zu verändern
        export_config = config.copy()
        
        # --- 1. Controller behandeln (Funktion/Objekt) ---
        if 'controller' in export_config:
            ctrl = export_config['controller']
            # Speichere den Namen der Funktion/Klasse als String
            ctrl_str = ctrl.__name__ if hasattr(ctrl, '__name__') else str(ctrl)
            export_config['controller_info_str'] = ctrl_str
            # Entferne das nicht-serialisierbare Objekt
            del export_config['controller']

        # --- 2. Geometrie-Funktion behandeln ---
        if 'geom_func' in export_config:
            geom_func = export_config['geom_func']
            # Speichere den Namen der Funktion als String
            geom_func_str = geom_func.__name__ if hasattr(geom_func, '__name__') else "Unbekannte Funktion"
            export_config['geom_func_info_str'] = geom_func_str
            # Entferne das nicht-serialisierbare Objekt
            del export_config['geom_func']

        # --- 3. Geometrie-Argumente (geom_kwargs) und andere Listen behandeln ---
        # Dies ist der kritische Schritt, um NumPy-Arrays in Listen zu konvertieren.
        
        # Hilfsfunktion zur rekursiven Konvertierung von NumPy-Typen
        def convert_to_serializable(item):
            if isinstance(item, (list, tuple, np.ndarray)):
                # Gehe rekursiv Listen/Arrays durch
                return [convert_to_serializable(x) for x in item]
            elif isinstance(item, dict):
                # Gehe rekursiv Dictionarys durch
                return {k: convert_to_serializable(v) for k, v in item.items()}
            elif isinstance(item, (np.float32, np.float64, np.generic)):
                # Konvertiere NumPy-Floats zu nativem Python-Float
                return float(item)
            elif isinstance(item, (np.int32, np.int64)):
                # Konvertiere NumPy-Integers zu nativem Python-Integer
                return int(item)
            else:
                return item

        # Wende die Konvertierung auf die geom_kwargs an (falls vorhanden)
        if 'geom_kwargs' in export_config:
            export_config['geom_kwargs'] = convert_to_serializable(export_config['geom_kwargs'])
            
        # Wende die Konvertierung auch auf andere Top-Level-Werte an, die Listen/Arrays sein könnten
        # (z.B. L_target, die aus np.array erstellt wurden)
        for key, value in export_config.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                export_config[key] = convert_to_serializable(value)
        
        
        # --- 4. Hinzufügen zur Exportliste ---
        exportable_list.append(export_config)

    # --- JSON-Export ---
    try:
        with open(filename, 'w') as f:
            json.dump(exportable_list, f, indent=indent)
        print(f"\n✅ ERFOLG: Konfigurationen erfolgreich exportiert nach: {filename}")
    except Exception as e:
        print(f"\n❌ FEHLER beim Exportieren nach JSON ({filename}): {e}")
        print("Stellen Sie sicher, dass keine nicht-serialisierbaren Typen (z.B. komplexe Objekte) übrig geblieben sind.")
