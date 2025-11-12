import mujoco
import mujoco.viewer as viewer
import time

# Modell aus XML-Datei laden
model = mujoco.MjModel.from_xml_path("spiral_chain.xml")

# Simulationsdaten erstellen
data = mujoco.MjData(model)
print("Modell und Simulationsdaten erfolgreich geladen.")
print(data.geom_xpos)


with mujoco.viewer.launch_passive(model, data) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()


    data.ctrl[0] = 0.3  # Set a constant control input for demonstration
    print(data.actuator('tendon_act_0'))
    #print(model.sensor('tendon0_pos'))   #.data
    print(data.sensor('tendon0_vel'))
    mujoco.mj_step(model, data)



    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

# while data.time < 10:  # Simulation für 10 Sekunden
#     viewer = viewer.launch_passive(model, data)
#     while viewer.is_running():
        
#         #data.ctrl[0] = 0.1
#         viewer.sync()
#     # Beispiel: Position eines Geoms ausgeben
#     #print(data.geom_xpos)
#     print(f"Zeit: {data.time:.2f}, Position: {data.body('seg_16').xpos.copy()}")
#     #print(data.sensordata)
#     time.sleep(0.1)