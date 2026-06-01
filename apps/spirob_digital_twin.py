import math

import mujoco as mj
import mujoco.viewer as viewer
import time
import math_spirob.spirob_generator as sg
import numpy as np
# Modell aus XML-Datei laden
#model = mujoco.MjModel.from_xml_path("spiral_chain.xml")

xml_string = sg.generate_xml_string(
    L_target=0.44, base_d=0.1, tip_d=0.03, Delta_theta_deg=30,
    model_name="spiral_chain_plot", auto_format=True
)
spec = mj.MjSpec.from_string(xml_string)

cylinder = spec.worldbody.add_body(
    name="cylinder",
    pos=[0.1, 0.0, 0.1]
)

# cylinder.add_geom(
#     name="cyl_geom",
#     type=mj.mjtGeom.mjGEOM_BOX,  #mj.mjtGeom.mjGEOM_CYLINDER,
#     size=[0.02, 0.1, 0.01], # radius, half-length, unused
#     euler=[np.pi/2, 0, 0],
#     rgba=[0.2, 0.8, 0.5, 1],
#     density=1000
# )

cylinder.add_geom(
    name="cyl_geom",
    type=mj.mjtGeom.mjGEOM_CYLINDER,  #mj.mjtGeom.mjGEOM_CYLINDER,
    size=[0.02, 0.1, 0.01], # radius, half-length, unused
    euler=[np.pi/2, 0, 0],
    rgba=[0.2, 0.8, 0.5, 1],
    density=1000
)

spirob = spec.body('seg_0')


model = spec.compile()


# Simulationsdaten erstellen
data = mj.MjData(model)
print("Modell und Simulationsdaten erfolgreich geladen.")
#print(data.geom_xpos)


with mj.viewer.launch_passive(model, data) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 360:
    step_start = time.time()


    #data.ctrl[0] = 0.3  # Set a constant control input for demonstration
    #print(data.actuator('tendon_act_0'))
    #print(model.sensor('tendon0_pos'))   #.data
    #print(data.sensor('tendon0_vel'))
    #print(data.sensor('tendon1_frc'))
    #print(data.sensor('gyro_0'))
    #print(data.geom_xpos)
    angles = [data.qpos[i].copy() for i in range(model.nq)]
    angles = [math.degrees(angle) for angle in angles]
    print(f"Angles: {angles}")
    mj.mj_step(model, data)



    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
