import numpy as np
from pathlib import Path
import math_spirob.spirob_generator as sg  # für High-Level API

# do you want automatic xml formating of the output file? (reqires Mujoco, removes all comments but formats nicely)
auto_formating = False
L_target   = 0.30               # gewünschte Mittelachsenlänge [m]
tip_d      = 0.01              # Spitzendurchmesser [m]
base_d     = 0.06              # Basisdurchmesser [m]
Delta_theta = np.deg2rad(30)    # Diskretisierungsschritt (30°)


# check for library usage
st = sg.generate_xml_string(L_target, base_d, tip_d, Delta_theta,"spiral_chain",auto_format=auto_formating)
out_path = Path("spiral_chain.xml")

# Aufruf der Funktion
saved_file = sg.generate_and_save_xml(
    filepath=out_path,
    L_target=L_target,
    base_d=base_d,
    tip_d=tip_d,
    Delta_theta_deg=np.rad2deg(Delta_theta),
    model_name="spiral_chain_example",
    auto_format = auto_formating
)

#print(st)