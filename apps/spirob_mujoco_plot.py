import mujoco as mj
import mujoco.viewer as viewer
import time
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import math_spirob.spirob_generator as sg
from typing import Dict, Any, List,Tuple

# ---------------------------------------------------------
# CONFIG FLAGS
# ---------------------------------------------------------
USE_VIEWER = False          # False = keine Visualisierung, True = mit Visualisierung
REALTIME = False            # False = Simulation schnellstmöglich, True = Echtzeit-Simulation
VIEWER_PASSIVE = True      # True = launch_passive, False = launch
SIM_TIME = 2.0             # Sekunden Simulation
# ---------------------------------------------------------
PLOT_ACC_DATA = True        # Ob Beschleunigungsdaten geplottet werden sollen 
PLOT_GYRO_DATA = True       # Ob Gyroskopdaten geplottet werden sollen
PLOT_TENDONFRC_DATA = True  # Ob Seilkraftdaten geplottet werden sollen
PLOT_TENDONPOS_DATA = True  # Ob Seilpositionsdaten geplottet werden sollen
PLOT_TENDOONVEL_DATA = True  # Ob Seilgeschwindigkeitsdaten geplottet werden sollen
PLOT_JOINTPOS_DATA = True    # Ob Gelenkpositionsdaten geplottet werden sollen
PLOT_JOINTVEL_DATA = True    # Ob Gelenkgeschwindigkeitsdaten geplottet werden sollen
# ---------------------------------------------------------

xml_string = sg.generate_xml_string(
    L_target=0.30,
    base_d=0.06,
    tip_d=0.01,
    Delta_theta_deg=30,
    model_name="spiral_chain_plot",
    auto_format=True
)

print("XML string generated.")

# Modell laden
#spec = mj.MjSpec.from_file("spiral_chain.xml")
spec = mj.MjSpec.from_string(xml_string)

# Function that recursively prints all body names
def print_bodies(parent, level=0):
  body = parent.first_body()
  while body:
    print(''.join(['-' for i in range(level)]) + body.name)
    print_bodies(body, level + 1)
    body = parent.next_body(body)

print("The spec has the following actuators:")
for actuator in spec.actuators:
  print(actuator.name)

print("\nThe spec has the following bodies:")
print_bodies(spec.worldbody)

model = spec.compile()

# Simulationsdaten erstellen
data = mj.MjData(model)

# ---------------------------------------------------------


positions_over_time = {}
acc_over_time = {}
gyro_over_time = {}
tendon_frc_over_time = {}
tendon_pos_over_time = {}
tendon_vel_over_time = {}
joint_pos_over_time = {}
joint_vel_over_time = {}


SENSOR_CONFIG = {
    mj.mjtSensor.mjSENS_ACCELEROMETER:    ('acc',    'acc_over_time'),
    mj.mjtSensor.mjSENS_GYRO:             ('gyro',   'gyro_over_time'),
    mj.mjtSensor.mjSENS_TENDONACTFRC:     ('tendon_frc', 'tendon_frc_over_time'),
    mj.mjtSensor.mjSENS_TENDONPOS:        ('tendon_pos', 'tendon_pos_over_time'),
    mj.mjtSensor.mjSENS_TENDONVEL:        ('tendon_vel', 'tendon_vel_over_time'),
    mj.mjtSensor.mjSENS_JOINTPOS:         ('joint_pos',  'joint_pos_over_time'),
    mj.mjtSensor.mjSENS_JOINTVEL:         ('joint_vel',  'joint_vel_over_time'),
}

sensor_metadata = []

num_steps = int(SIM_TIME / model.opt.timestep) + 1

num_sensors = model.nsensor
print(f"Anzahl der Sensoren im Modell: {num_sensors}")


for i in range(num_sensors):
    sensor_type = model.sensor_type[i]
    
    if sensor_type in SENSOR_CONFIG:
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)
        key_name, dict_name = SENSOR_CONFIG[sensor_type]
        
        # HIER IST DIE KORREKTUR: Abrufen der Dimension (1D oder 3D)
        dim = model.sensor_dim[i] 
        
        # Erstelle ein N x D Array
        # D ist 3 für Acc/Gyro und 1 für die anderen
        time_series_array = np.zeros((num_steps, dim), dtype=np.float64)
        
        # Speichere den Array im globalen Dictionary
        globals()[dict_name][name] = time_series_array 
        
        # Speichere die Metadaten für den Simulations-Loop
        sensor_metadata.append({
            'name': name,
            'array': time_series_array, 
            'index': i                  
        })

print(f"Speicher für {len(sensor_metadata)} Sensor-Zeitreihen vorab zugewiesen.")
# print("Sensor-Metadaten:")
# for meta in sensor_metadata:
#     print(f"  Sensor Name: {meta['name']}, Array Shape: {meta['array'].shape}, Index: {meta['index']}")


# --- GEOM-POSITIONEN ---
geom_metadata = []
i = 0
while True:
    name = f"g_{i}"
    geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
    if geom_id == -1:
        break
    
    # Position hat immer 3 Dimensionen
    pos_array = np.zeros((num_steps, 3), dtype=np.float64)
    positions_over_time[name] = pos_array
    
    geom_metadata.append({
        'name': name,
        'array': pos_array,
        'id': geom_id
    })
    i += 1

print(f"Initialisierung abgeschlossen. Speicher für {num_steps} Schritte zugewiesen.")
print(f"Anzahl der zu verfolgenen Geoms: {len(geom_metadata)}")
# print("Geom-Metadaten:")
# for meta in geom_metadata:
#     print(f"  Geom Name: {meta['name']}, Array Shape: {meta['array'].shape}, ID: {meta['id']}")


# Das Array für die Zeit initialisieren
time_array = np.zeros(num_steps, dtype=np.float64)

# ---------------------------------------------------------
# SIMULATION OHNE VIEWER
# ---------------------------------------------------------
if not USE_VIEWER:

    print(f"Simulation ohne Viewer läuft für {SIM_TIME} Sekunden...")
    start_wall = time.time()
    
    # 0. Starte mit Index 0 (t=0)
    step_index = 0
    
    # Zustand t=0 speichern
    time_array[step_index] = data.time
    for meta in sensor_metadata:
        meta['array'][step_index] = data.sensor(meta['index']).data
    for meta in geom_metadata:
        meta['array'][step_index] = data.geom_xpos[meta['id']]
        
    
    # Schleife für die restlichen Schritte
    steps_to_run = num_steps - 1

    for _ in range(steps_to_run):
            
        # Inkrementiere den Index VOR dem Speichern des neuen Zustands
        step_index += 1
        
        # --- Simulationsschritt ---
        data.ctrl[0] = 0.2
        mj.mj_step(model, data)
        
        # --- EFFIZIENTE DATENSPEICHERUNG ---
        
        # 1. Zeit speichern
        time_array[step_index] = data.time
        
        # 2. Speichere Geom-Positionen
        for meta in geom_metadata:
            meta['array'][step_index] = data.geom_xpos[meta['id']]
            
        # 3. Speichere Sensorwerte
        for meta in sensor_metadata:
            meta['array'][step_index] = data.sensor(meta['index']).data
            
        
        # Der REALTIME-Block sollte in Batch-Simulationen deaktiviert sein.
        # if REALTIME:
        #     time.sleep(model.opt.timestep)

    
    duration = time.time() - start_wall
    final_data_length = step_index + 1
    print(f"Simulation beendet. Daten für {step_index + 1} Schritte gespeichert.")
    print(f"Dauer: {duration:.4f} Sekunden.")


# ---------------------------------------------------------
# SIMULATION MIT VIEWER
# ---------------------------------------------------------
else:
    print("Simulation mit Viewer läuft...")
    start_wall = time.time()

    # Passiver Viewer oder normaler Viewer
    launch_fn = viewer.launch_passive if VIEWER_PASSIVE else viewer.launch

    with launch_fn(model, data) as v:

        start = time.time()
        while v.is_running() and time.time() - start < SIM_TIME:

            step_start = time.time()

            data.ctrl[0] = 0.2
            positions_over_time.append(data.geom_xpos.copy())

            mj.mj_step(model, data)

            v.sync()   # Viewer aktualisieren

            if REALTIME:
                # Echtzeit-Synchronisation
                dt = model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)
        duration = time.time() - start_wall
        print(f"Simulation mit Viewer beendet. Dauer: {duration:.2f} Sekunden")
        print("Viewer geschlossen.")


# ---------------------------------------------------------
# PLOTTEN
# ---------------------------------------------------------

def plot_sensors_grouped_np(sensor_np_dict: Dict[str, np.ndarray], title: str, time_data: np.ndarray):
    """
    Plottet Sensordaten aus den effizienten NumPy-Arrays (N x D).

    sensor_np_dict: z.B. {"acc_0": array([[x, y, z], ...]), ...}
    time_data: Das 1D NumPy Array mit den Zeitstempeln.
    """

    if not sensor_np_dict:
        return

    # Prüfe Dimension: 1D oder 3D Sensor?
    any_key = next(iter(sensor_np_dict))
    sample = sensor_np_dict[any_key]

    # Die Dimension (D) ist die Anzahl der Spalten (1 oder 3)
    D = sample.shape[1] 

    if D == 3:
        # ======== 3D Sensor (Vektor-Sensor) ============
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"{title} (Vektor-Sensor)", fontsize=16)

        labels = ["X-Achse", "Y-Achse", "Z-Achse"]

        for i, label in enumerate(labels):
            ax = axs[i]
            for name, values_array in sensor_np_dict.items():
                # values_array[:, i] holt die i-te Spalte (X, Y oder Z)
                ax.plot(time_data, values_array[:, i], label=name)
            
            ax.set_ylabel(label)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='lower right')

        axs[-1].set_xlabel("Zeit (s)")
        #plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Platz für Suptitle

    elif D == 1:
        # ======== 1D Sensor (Skalar-Sensor) ============
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f"{title} (Skalar-Sensor)", fontsize=16)

        for name, values_array in sensor_np_dict.items():
            # values_array.squeeze() entfernt die unnötige 1er-Dimension (N x 1 -> N)
            ax.plot(time_data, values_array.squeeze(), label=name)

        ax.set_ylabel("Wert")
        ax.set_xlabel("Zeit (s)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    else:
        print(f"WARNUNG: Sensor '{any_key}' hat Dimension {D}. Wird ignoriert.")


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
    """
    
    # 1. Initialisiere mit der Zeitspalte
    all_columns: List[pl.Series] = [pl.Series("time_s", time_data)]
    
    # 2. Iteriere durch alle Sensor-Gruppen
    for group_prefix, data_dict_global in sensor_groups:
        
        if not data_dict_global:
            print(f"Warnung: Gruppe '{group_prefix}' ist leer und wird ignoriert.")
            continue

        # A. Synchronisieren (schneiden) der Daten
        sliced_data_dict = get_sliced_dict(data_dict_global, final_length)
        
        # Holen der Dimension für diese Gruppe
        any_key = next(iter(sliced_data_dict))
        values_array_sample = sliced_data_dict[any_key]
        D = values_array_sample.shape[1]

        # B. Spalten aus dem geschnittenen Dictionary erstellen
        for name, values_array in sliced_data_dict.items():
            
            if D == 3:
                # Vektor-Sensor (3D): Benennung: group_sensorname_Achse (z.B. acc_torso_X)
                columns_prefix = f"{group_prefix}_{name}"
                all_columns.append(pl.Series(f"{columns_prefix}_X", values_array[:, 0]))
                all_columns.append(pl.Series(f"{columns_prefix}_Y", values_array[:, 1]))
                all_columns.append(pl.Series(f"{columns_prefix}_Z", values_array[:, 2]))
            
            elif D == 1:
                # Skalar-Sensor (1D): Benennung: group_sensorname (z.B. tendon_frc_seil1)
                column_name = f"{group_prefix}_{name}"
                all_columns.append(pl.Series(column_name, values_array.squeeze()))
                
    # 3. DataFrame aus allen gesammelten Polars Series erstellen
    return pl.DataFrame(all_columns)

if not USE_VIEWER: # Dies stellt sicher, dass es nach der Batch-Simulation läuft
    
    # WICHTIG: Verwenden Sie hier das globale time_array aus dem effizienten Code
    time_series_data = time_array[:step_index+1] # Nutzen Sie nur die tatsächlich geschriebenen Samples
    
    
    # --- Polars Beispiel ---
    # Optional: Erstellung eines Polars DataFrames für z.B. Accelerometer
    # if PLOT_ACC_DATA and acc_over_time:
    #     acc_df = numpy_dict_to_polars_df(acc_over_time, time_series_data)
    #     print("\n--- Polars DataFrame Beispiel (ACC) ---")
    #     print(acc_df.head(3))
    #     # Sie könnten hier Filtern oder Aggregieren, bevor Sie plotten.

    
    # Definiere alle Sensor-Dictionaries, die du konvertieren möchtest
    # Definiert die Reihenfolge und die zu verarbeitenden Daten
    SENSOR_GROUPS_CONFIG: List[Tuple[str, Dict[str, np.ndarray]]] = [
        ("acc", acc_over_time),
        ("gyro", gyro_over_time),
        ("tendon_frc", tendon_frc_over_time),
        ("tendon_pos", tendon_pos_over_time),
        ("tendon_vel", tendon_vel_over_time),
        ("joint_pos", joint_pos_over_time),
        ("joint_vel", joint_vel_over_time),
        # ("geom_pos", positions_over_time), # Kann hinzugefügt werden
    ]

    #Ermitteln der finalen Länge (Synchronisation)
    final_length = step_index + 1 
    time_series_data = time_array[:final_length] 

    print(f"Konvertiere alle {final_length} Samples in einen einzigen Polars DataFrame...")

    #Erstellung des finalen DataFrames
    final_wide_df = create_single_polars_dataframe(
        SENSOR_GROUPS_CONFIG, 
        time_series_data, 
        final_length
    )
    
    print("\n--- ERGEBNIS: Breiter Polars DataFrame ---")
    print(f"Gesamte Zeilen: {final_wide_df.shape[0]}")
    print(f"Gesamte Spalten: {final_wide_df.shape[1]}")
    print("\nKopfzeile (Head):")
    print(final_wide_df.head(10))

    
    if PLOT_ACC_DATA and acc_over_time:
        plot_sensors_grouped_np(acc_over_time, "Beschleunigungssensoren", time_series_data)
    
    if PLOT_GYRO_DATA and gyro_over_time:
        plot_sensors_grouped_np(gyro_over_time, "Gyroskop-Daten", time_series_data)
        
    if PLOT_TENDONFRC_DATA and tendon_frc_over_time:
        plot_sensors_grouped_np(tendon_frc_over_time, "Seilkraft-Daten", time_series_data)
        
    if PLOT_TENDONPOS_DATA and tendon_pos_over_time:
        plot_sensors_grouped_np(tendon_pos_over_time, "Seilpositions-Daten", time_series_data)
        
    if PLOT_TENDOONVEL_DATA and tendon_vel_over_time:
        plot_sensors_grouped_np(tendon_vel_over_time, "Seilgeschwindigkeits-Daten", time_series_data)
        
    if PLOT_JOINTPOS_DATA and joint_pos_over_time:
        plot_sensors_grouped_np(joint_pos_over_time, "Gelenkpositions-Daten", time_series_data)
        
    if PLOT_JOINTVEL_DATA and joint_vel_over_time:
        plot_sensors_grouped_np(joint_vel_over_time, "Gelenkgeschwindigkeits-Daten", time_series_data)

    
    plt.show() # Zeigt alle erstellten Matplotlib-Fenster an
