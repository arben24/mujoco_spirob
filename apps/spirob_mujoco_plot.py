import mujoco as mj
import mujoco.viewer as viewer
import time
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# CONFIG FLAGS
# ---------------------------------------------------------
USE_VIEWER = False          # False = keine Visualisierung, True = mit Visualisierung
REALTIME = False            # False = Simulation schnellstmöglich, True = Echtzeit-Simulation
VIEWER_PASSIVE = True      # True = launch_passive, False = launch
SIM_TIME = 2.0             # Sekunden Simulation
# ---------------------------------------------------------


# Modell laden
spec = mj.MjSpec.from_file("spiral_chain.xml")


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

positions_over_time = []


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

        positions_over_time.append(data.geom_xpos.copy())

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

positions_over_time = np.array(positions_over_time)
num_elems = positions_over_time.shape[1]

plt.figure(figsize=(8, 6))

for elem in range(num_elems):
    x = positions_over_time[:, elem, 0]
    z = positions_over_time[:, elem, 2]
    plt.plot(x, z, label=f'Geom #{elem}')

plt.xlabel('X')
plt.ylabel('Z')
plt.title('Bewegung der Geoms in X-Z Ebene')
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
