import mujoco as mj
import math_spirob.spirob_generator as sg
import math_spirob.spirob_simulate as spir_sim # Ihr Bibliotheksmodul
import polars as pl
from typing import List, Dict, Any, Union
import itertools
import numpy as np

# --- 1. Konfiguration der Simulationsläufe ---

ENABLE_REALTIME_VIEWER = True   #False,True
BOOST_VIEWER = 0.3  # Geschwindigkeit des Viewers (1.0 = Echtzeit, >1.0 = schneller)

VARIABLE_PARAMS = {
    "L_target": [0.30, 0.35, 0.40],
    "base_d":   [0.05],
    "sim_time": [2.0],
    "controller": {
        "Static": spir_sim.static_controller,
        #"Sine": spir_sim.sine_controller,
        # "PID_K50": spir_sim.PIDController(target_pos=0.5, Kp=50.0, Ki=5.0) 
    },
}

GEOM_SCENARIOS = [
    # Szenario 1: Zylinder (hat spezifische Größen für Zylinder)
    {
        "obj_name": "Cyl",           # Name für ID
        "setup_func": spir_sim.setup_cylinder, # Die Funktion von oben
        "params": {
            "pos":  [[0.1, 0.0, 0.1]], 
            "size": [[0.02, 0.1, 0.0], [0.05, 0.1, 0.0]], # radius, half-length, unused
            "euler": [[90, 1, 1]]
        }
    },
    # Szenario 2: Box (hat ganz andere Größen-Dimensionen)
    {
        "obj_name": "Box",
        "setup_func": spir_sim.setup_box,
        "params": {
            "pos":  [[0.1, 0.0, 0.1]], # Box steht woanders
            "size": [[0.05, 0.1, 0.05], [0.04, 0.1, 0.04]], # Würfel vs Riegel
            "euler": [[0, 0, 0]]       # Box drehen wir nicht
        }
    }
]

# --- B. Feste Parameter (Der "Fixed Context") ---
# Diese Werte werden zu JEDER Konfiguration hinzugefügt.
FIXED_PARAMS = {
    "tip_d": 0.01,
    "Delta_theta_deg": 30,
    "include_geom_pos": False,
}

# Liste zur Speicherung der Ergebnisse: Jedes Element ist ein Dict {config_data: ..., dataframe: ...}
# Diese Liste wird nun das finale Ergebnis sein.
final_results_list: List[Dict[str, Union[Dict[str, Any], pl.DataFrame]]] = []

# --- 2. Iteration und Ausführung ---

# --- Aufruf der Funktion ---
SIM_CONFIGS = spir_sim.generate_hybrid_grid_configs(VARIABLE_PARAMS, GEOM_SCENARIOS, FIXED_PARAMS)
print(f"Es wurden {len(SIM_CONFIGS)} Simulationsläufe für die Grid Search generiert.")
spir_sim.print_configs_formatted(SIM_CONFIGS,preview_limit=5,print_all=False)
spir_sim.save_configs_to_json(SIM_CONFIGS, filename="simulation_configs.json")

for config in SIM_CONFIGS:
    
    run_id = config["id"]
    print(f"\n==========================================")
    print(f"Starte Simulation: {run_id}")
    print(f"==========================================")

    # A. Modellspezifische Parameter
    xml_string = sg.generate_xml_string(
        L_target=config["L_target"],
        base_d=config["base_d"],
        tip_d=config["tip_d"],
        Delta_theta_deg=config["Delta_theta_deg"],
        model_name=f"model_{run_id}",
        auto_format=False
    )

    # B. Modell laden und Daten initialisieren
    spec = mj.MjSpec.from_string(xml_string)

    setup_func = config["geom_func"]
    geom_args = config["geom_kwargs"]
    #print(f"  Füge Geometrie hinzu:  mit Parametern {geom_args}")
    setup_func(worldbody=spec.worldbody, **geom_args)

    model = spec.compile()
    data = mj.MjData(model)
    
    # C. Simulation ausführen
    current_df = None
    try:
        current_df = spir_sim.run_simulation_and_get_dataframe(
            model=model, 
            data=data, 
            sim_time=config["sim_time"],
            controller=config["controller"], 
            include_geom_pos=config["include_geom_pos"],
            enable_viewer=ENABLE_REALTIME_VIEWER,
            boost_viewer=BOOST_VIEWER
        )
    except Exception as e:
        print(f"Fehler in Lauf {run_id}: {e}")
        continue 

    # D. Metadaten als Spalten hinzufügen
    current_df = current_df.with_columns([
        pl.lit(run_id).alias("run_id"),
        pl.lit(config["L_target"]).alias("L_target_val"),
        pl.lit(config["base_d"]).alias("base_d_val"),
        pl.lit(config["tip_d"]).alias("tip_d_val"),
        pl.lit(config["Delta_theta_deg"]).alias("Delta_theta_deg_val"),
        pl.lit(config["sim_time"]).alias("sim_time_s"),
        pl.lit(str(config["controller"])).alias("controller_info"),
    ])
    
    print(f"Erfolgreich beendet. DataFrame Shape: {current_df.shape}")
    #print(f"  Spaltennamen: {current_df.columns}") # Zeigt die unterschiedlichen Spaltenzahlen
    
    # E. Ergebnis in die Dictionary-Struktur speichern
    final_results_list.append({
        "config_data": config, 
        "dataframe": current_df 
    })

# ---------------------------------------------------------
# --- 3. Finales Ergebnis (Nur Ausgabe der Liste) ---
# ---------------------------------------------------------

if final_results_list:
    print("\n--- ERGEBNISSE GESPEICHERT ---")
    
    # for i, result in enumerate(final_results_list):
    #     df = result["dataframe"]
    #     cfg = result["config_data"]
        
    #     print(f"\nLauf {i+1} ({cfg['id']}):")
    #     print(f"  Konfiguration: L_target={cfg['L_target']}, Time={cfg['sim_time']}")
    #     print(f"  DataFrame Shape: {df.shape}")
    #     print(f"  Kopfzeile:")
    #     print(df.head(10))
        
    # Beispiel für den Zugriff auf den ersten Lauf:
    run_1_df = final_results_list[0]["dataframe"]
    
else:
    print("Keine Ergebnisse erfolgreich erzeugt.")