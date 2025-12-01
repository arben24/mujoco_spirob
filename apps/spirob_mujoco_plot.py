import mujoco as mj
import mujoco.viewer as viewer
import time
import matplotlib.pyplot as plt
import numpy as np
import math_spirob.spirob_generator as sg
from typing import Dict

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

positions_over_time = {}
acc_over_time = {}
gyro_over_time = {}
tendon_frc_over_time = {}
tendon_pos_over_time = {}
tendon_vel_over_time = {}
joint_pos_over_time = {}
joint_vel_over_time = {}

num_sensors = model.nsensor

print(f"Anzahl der Sensoren im Modell: {num_sensors}")

# Zähler für Accelerometer
num_accelerometers = 0

for i in range(num_sensors):
    if model.sensor_type[i] == mj.mjtSensor.mjSENS_ACCELEROMETER:
        acc_over_time[mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)] = []
        num_accelerometers += 1

print(f"Anzahl der Accelerometer im Modell: {num_accelerometers}")
print(f"Accelerometer Namen: {list(acc_over_time)}")
print(list(acc_over_time.keys())[0])

# Zähler für Gyroskope
num_gyro = 0

for i in range(num_sensors):
    if model.sensor_type[i] == mj.mjtSensor.mjSENS_GYRO:
        gyro_over_time[mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)] = []
        num_gyro += 1

print(f"Anzahl der Gyroskope im Modell: {num_gyro}")
print(f"Gyroskop Namen: {list(gyro_over_time)}")

num_tendon_frc_over_time = 0

for i in range(num_sensors):
    if model.sensor_type[i] == mj.mjtSensor.mjSENS_TENDONACTFRC:
        tendon_frc_over_time[mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)] = []
        num_tendon_frc_over_time += 1

print(f"Anzahl der Seilkraft-Sensoren im Modell: {num_tendon_frc_over_time}")
print(f"Seilkraft-Sensor Namen: {list(tendon_frc_over_time)}")

num_tendon_pos_over_time = 0

for i in range(num_sensors):
    if model.sensor_type[i] == mj.mjtSensor.mjSENS_TENDONPOS:
        tendon_pos_over_time[mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)] = []
        num_tendon_pos_over_time += 1

print(f"Anzahl der Seilpositions-Sensoren im Modell: {num_tendon_pos_over_time}")
print(f"Seilpositions-Sensor Namen: {list(tendon_pos_over_time)}")
num_tendon_vel_over_time = 0
for i in range(num_sensors):
    if model.sensor_type[i] == mj.mjtSensor.mjSENS_TENDONVEL:
        tendon_vel_over_time[mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)] = []
        num_tendon_vel_over_time += 1
print(f"Anzahl der Seilgeschwindigkeits-Sensoren im Modell: {num_tendon_vel_over_time}")
print(f"Seilgeschwindigkeits-Sensor Namen: {list(tendon_vel_over_time)}")
num_joint_pos_over_time = 0
for i in range(num_sensors):
    if model.sensor_type[i] == mj.mjtSensor.mjSENS_JOINTPOS:
        joint_pos_over_time[mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)] = []
        num_joint_pos_over_time += 1
print(f"Anzahl der Gelenkpositions-Sensoren im Modell: {num_joint_pos_over_time}")
print(f"Gelenkpositions-Sensor Namen: {list(joint_pos_over_time)}")
num_joint_vel_over_time = 0
for i in range(num_sensors):
    if model.sensor_type[i] == mj.mjtSensor.mjSENS_JOINTVEL:
        joint_vel_over_time[mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)] = []
        num_joint_vel_over_time += 1
print(f"Anzahl der Gelenkgeschwindigkeits-Sensoren im Modell: {num_joint_vel_over_time}")
print(f"Gelenkgeschwindigkeits-Sensor Namen: {list(joint_vel_over_time)}")
i = 0
target_geom_names = []
while True:
    name = f"g_{i}"
    geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
    if geom_id == -1:
        break
    target_geom_names.append(name)
    positions_over_time[name] = []
    i += 1
print(f"Zielgeoms zum Tracken: {target_geom_names}")

geom_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name) for name in target_geom_names]
print(f"Geom IDs der Zielgeoms: {geom_ids}")

# ---------------------------------------------------------
# SIMULATION OHNE VIEWER
# ---------------------------------------------------------
if not USE_VIEWER:

    print("Simulation ohne Viewer läuft...")
    start_wall = time.time()

    start = time.time()
    steps = int(SIM_TIME / model.opt.timestep)
    for _ in range(steps):

        data.ctrl[0] = 0.2

        for i in range(len(geom_ids)):
                positions_over_time[list(positions_over_time.keys())[i]].append(data.geom_xpos[geom_ids[i]].copy())
        if PLOT_ACC_DATA:
            for i in range(len(list(acc_over_time.keys()))):
                acc_over_time[list(acc_over_time.keys())[i]].append(data.sensor(list(acc_over_time.keys())[i]).data.copy())
        if PLOT_GYRO_DATA:
            for i in range(len(list(gyro_over_time.keys()))):
                gyro_over_time[list(gyro_over_time.keys())[i]].append(data.sensor(list(gyro_over_time.keys())[i]).data.copy())
        if PLOT_TENDONFRC_DATA:
            for i in range(len(list(tendon_frc_over_time.keys()))):
                tendon_frc_over_time[list(tendon_frc_over_time.keys())[i]].append(data.sensor(list(tendon_frc_over_time.keys())[i]).data.copy())
        if PLOT_TENDONPOS_DATA:
            for i in range(len(list(tendon_pos_over_time.keys()))):
                tendon_pos_over_time[list(tendon_pos_over_time.keys())[i]].append(data.sensor(list(tendon_pos_over_time.keys())[i]).data.copy())
        if PLOT_TENDOONVEL_DATA:    
            for i in range(len(list(tendon_vel_over_time.keys()))):
                tendon_vel_over_time[list(tendon_vel_over_time.keys())[i]].append(data.sensor(list(tendon_vel_over_time.keys())[i]).data.copy())
        if PLOT_JOINTPOS_DATA:
            for i in range(len(list(joint_pos_over_time.keys()))):
                joint_pos_over_time[list(joint_pos_over_time.keys())[i]].append(data.sensor(list(joint_pos_over_time.keys())[i]).data.copy())
        if PLOT_JOINTVEL_DATA:
            for i in range(len(list(joint_vel_over_time.keys()))):
                joint_vel_over_time[list(joint_vel_over_time.keys())[i]].append(data.sensor(list(joint_vel_over_time.keys())[i]).data.copy())

        mj.mj_step(model, data)

        if REALTIME:
            time.sleep(model.opt.timestep)
    
    duration = time.time() - start_wall
    print(f"Simulation ohne Viewer beendet. Dauer: {duration:.2f} Sekunden")


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

def dict_to_arrays(sensor_dict):
    return {name: np.array(values) for name, values in sensor_dict.items()}

def plot_sensors_grouped(sensor_np_dict, sensor_prefix, title):
    """
    sensor_np_dict: z.B. {"acc_0": array([[x, y, z], ...]), "acc_1": array([...]), ...}
                    oder {"tendon_frc_0": array([value,...]), ...}

    Automatische Erkennung:
    - 3D-Sensoren → Subplots X/Y/Z
    - 1D-Sensoren → 1 Subplot mit allen Kurven
    """

    # Prüfe Dimension: 1D oder 3D Sensor?
    any_key = next(iter(sensor_np_dict))
    sample = sensor_np_dict[any_key]

    is_vector_sensor = (sample.ndim == 2 and sample.shape[1] == 3)
    is_scalar_sensor = (sample.ndim == 1) or (sample.ndim == 2 and sample.shape[1] == 1)

    if is_vector_sensor:
        # ======== 3D Sensor (acc, gyro, joint_pos, ...) ============
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(title)

        labels = ["X-Achse", "Y-Achse", "Z-Achse"]

        for i, label in enumerate(labels):
            ax = axs[i]
            for name, values in sensor_np_dict.items():
                ax.plot(values[:, i], label=name)
            ax.set_ylabel(label)
            ax.legend()

        axs[-1].set_xlabel("Samples")
        plt.tight_layout()

    elif is_scalar_sensor:
        # ======== 1D Sensor (tendon_frc, joint_vel, ...) ============
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(title)

        for name, values in sensor_np_dict.items():
            ax.plot(values.squeeze(), label=name)

        ax.set_ylabel("Wert")
        ax.set_xlabel("Samples")
        ax.legend()
        plt.tight_layout()

    else:
        raise ValueError("Sensor hat unbekannte Dimension – weder 1D noch 3D.")




positions_np = dict_to_arrays(positions_over_time)
#print(f"Keys in positions_np: {positions_np.items()}")

acc_np = dict_to_arrays(acc_over_time)
gyro_np = dict_to_arrays(gyro_over_time)
tendon_frc_np = dict_to_arrays(tendon_frc_over_time)
tendon_pos_np = dict_to_arrays(tendon_pos_over_time)
tendon_vel_np = dict_to_arrays(tendon_vel_over_time)
joint_pos_np = dict_to_arrays(joint_pos_over_time)
joint_vel_np = dict_to_arrays(joint_vel_over_time)

#print(acc_np)

num_elems = len(positions_np)
any_key = next(iter(positions_np))
num_samples = positions_np[any_key].shape[0]
print(f"Anzahl der Geoms: {num_elems}")
print(f"Anzahl der Samples: {num_samples}")


if PLOT_ACC_DATA and acc_np:
    plot_sensors_grouped(acc_np, "acc", "Beschleunigungssensoren (X/Y/Z gruppiert)")
if PLOT_GYRO_DATA and gyro_np:
    plot_sensors_grouped(gyro_np, "gyro", "Gyroskop-Daten (X/Y/Z gruppiert)")
if PLOT_TENDONFRC_DATA and tendon_frc_np:
    plot_sensors_grouped(tendon_frc_np, "tendon_frc", "Seilkraft-Daten (X/Y/Z gruppiert)")
if PLOT_TENDONPOS_DATA and tendon_pos_np:
    plot_sensors_grouped(tendon_pos_np, "tendon_pos", "Seilpositions-Daten (X/Y/Z gruppiert)")
if PLOT_TENDOONVEL_DATA and tendon_vel_np:
    plot_sensors_grouped(tendon_vel_np, "tendon_vel", "Seilgeschwindigkeits-Daten (X/Y/Z gruppiert)")
if PLOT_JOINTPOS_DATA and joint_pos_np:
    plot_sensors_grouped(joint_pos_np, "joint_pos", "Gelenkpositions-Daten (X/Y/Z gruppiert)")
if PLOT_JOINTVEL_DATA and joint_vel_np:
    plot_sensors_grouped(joint_vel_np, "joint_vel", "Gelenkgeschwindigkeits-Daten (X/Y/Z gruppiert)")

plt.show()
