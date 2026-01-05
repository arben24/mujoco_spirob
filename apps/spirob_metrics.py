import mujoco as mj
import math_spirob.spirob_generator as sg
import math_spirob.spirob_simulate as spir_sim # Ihr Bibliotheksmodul
import math_spirob.data_schema as ds
import math_spirob.exporter as exp
import polars as pl
from typing import List, Dict, Any, Union
import itertools
import numpy as np
import os

# --- 1. Konfiguration der Simulationsläufe ---

ENABLE_REALTIME_VIEWER = False   #False,True
BOOST_VIEWER = 0.3  # Geschwindigkeit des Viewers (1.0 = Echtzeit, >1.0 = schneller)

VARIABLE_PARAMS = {
    # "L_target": [0.30, 0.35, 0.40],
    # "base_d":   [0.05, 0.07, 0.09],
    # "sim_time": [2.0,3.0],
    "L_target": [0.30, 0.35, 0.40],
    "base_d":   [0.05,0.08],
    "sim_time": [2.0],
    "controller": {
        #"Static": spir_sim.static_controller,
        "Ramped": spir_sim.ramped_controller,
        #"Sine": spir_sim.sine_controller,
        # "PID_K50": spir_sim.PIDController(target_pos=0.5, Kp=50.0, Ki=5.0) 
    },
}

GEOM_SCENARIOS = [
    # Szenario 1: Zylinder (hat spezifische Größen für Zylinder)
    # {
    #     "obj_name": "Cyl",           # Name für ID
    #     "setup_func": spir_sim.setup_cylinder, # Die Funktion von oben
    #     "params": {
    #         "pos":  [[0.1, 0.0, 0.1],[0.12, 0.0, 0.1],[0.1, 0.0, 0.12],[0.1, 0.0, 0.08]], 
    #         "size": [[0.02, 0.1, 0.0], [0.05, 0.1, 0.0], [0.08, 0.1, 0.0], [0.03, 0.1, 0.0], [0.06, 0.1, 0.0]], # radius, half-length, unused
    #         "euler": [[90, 0, 0]]       # Zylinder drehen wir um 90° um X
    #     }
    # },
        {
        "obj_name": "Cyl",           # Name für ID
        "setup_func": spir_sim.setup_cylinder, # Die Funktion von oben
        "params": {
            "pos":  [[0.1, 0.0, 0.1],], 
            "size": [[0.02, 0.1, 0.0], [0.06, 0.1, 0.0]], # radius, half-length, unused
            "euler": [[90, 0, 0]]       # Zylinder drehen wir um 90° um X
        }
    },
    # Szenario 2: Box (hat ganz andere Größen-Dimensionen)
    # {
    #     "obj_name": "Box",
    #     "setup_func": spir_sim.setup_box,
    #     "params": {
    #         "pos":  [[0.1, 0.0, 0.1]], # Box steht woanders
    #         "size": [[0.05, 0.1, 0.05], [0.04, 0.1, 0.04]], # Würfel vs Riegel
    #         "euler": [[0, 0, 0]]       # Box drehen wir nicht
    #     }
    # }
]

# --- B. Feste Parameter (Der "Fixed Context") ---
# Diese Werte werden zu JEDER Konfiguration hinzugefügt.
FIXED_PARAMS = {
    "tip_d": 0.01,
    "Delta_theta_deg": 30,
    "include_geom_pos": False,
}

# Liste zur Speicherung der Ergebnisse: Jedes Element ist ein Dict {record: ExperimentRecord}
# Diese Liste wird nun das finale Ergebnis sein.
final_results_list: List[Dict[str, ds.ExperimentRecord]] = []

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
    #try:
    current_df = spir_sim.run_simulation_and_get_dataframe(
            model=model, 
            data=data, 
            sim_time=config["sim_time"],
            controller=config["controller"], 
            include_geom_pos=config["include_geom_pos"],
            enable_viewer=ENABLE_REALTIME_VIEWER,
            boost_viewer=BOOST_VIEWER
        )
    # except Exception as e:
    #     print(f"Fehler in Lauf {run_id}: {e}")
    #     continue 

    # D. ExperimentRecord erstellen und speichern
    exp_config = ds.ExperimentConfig(
        L_target=config["L_target"],
        base_d=config["base_d"],
        tip_d=config["tip_d"],
        Delta_theta_deg=config["Delta_theta_deg"],
        sim_time=config["sim_time"],
        controller_info=str(config["controller"]),
        geom_type=config["geom_func"].__name__.replace('setup_', ''),
        geom_params=config["geom_kwargs"],
        include_geom_pos=config["include_geom_pos"]
    )
    sensors = exp.generate_sensor_meta(current_df)
    record = ds.ExperimentRecord(run_id=run_id, config=exp_config, sensors=sensors)
    exp.save_experiment(current_df, record)
    
    print(f"Erfolgreich beendet. DataFrame Shape: {current_df.shape}")
    
    # E. Ergebnis in die Dictionary-Struktur speichern
    final_results_list.append({
        "record": record
    })

# ---------------------------------------------------------
# --- 3. Finales Ergebnis (Nur Ausgabe der Liste) ---
# ---------------------------------------------------------

if final_results_list:
    print("\n--- ERGEBNISSE GESPEICHERT ---")
    
    for i, result in enumerate(final_results_list):
        record = result["record"]
        
        print(f"\nLauf {i+1} ({record.run_id}):")
        print(f"  Konfiguration: L_target={record.config.L_target}, Time={record.config.sim_time}")
        print(f"  Sensoren: {len(record.sensors)}")
        
    print("Alle Experimente wurden als Parquet und JSON gespeichert.")
    
else:
    print("Keine Ergebnisse erfolgreich erzeugt.")