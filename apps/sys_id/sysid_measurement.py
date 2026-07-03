import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# Beispiel-Messreihe (Masse in Gramm, Winkel in Grad)
# mass_g = np.array([0, 27, 99, 181, 247, 316, 370])
# angle_deg = np.array([30-30, 30-26, 30-22, 30-16, 30-7, 30-3, 30-2])# r = 0.06

#mass_g = np.array([0, 130, 161, 201, 210, 420])    k = 0.5108
#angle_deg = np.array([0, 10, 15, 20, 25, 30])    #r = 0.077

#mass_g = np.array([0, 61, 86, 155, 197, 200,218])    #k = 0.226
#angle_deg = np.array([0, 5,10, 20, 25, 27,30])    #r = 0.057

#mass_g = np.array([0, 85, 143, 178, 225, 240,270,380])    #k = 0.293
#angle_deg = np.array([0, 5,10, 15, 20, 25, 27,30])    #r = 0.05

mass_g = np.array([0, 90, 143, 180, 227, 280, 291, 320])    #k = 0.293
angle_deg = np.array([0, 5,10, 15, 20, 25, 27, 30])    #r = 0.047

# Konstanten
g = 9.81 # Erdbeschleunigung in m/s^2
r = 0.047 # Hebelarm in Metern (6 cm)

# Umrechnungen
mass_kg = mass_g / 1000.0          # Gramm -> Kilogramm
force_n = mass_kg * g              # Kilogramm -> Newton
torque_nm = force_n * r            # Drehmoment: Kraft * Hebelarm
angle_rad = np.deg2rad(angle_deg)  # Grad -> Radiant

# Linearer Fit (Drehmoment über Winkel)
# M = k * phi  => k ist die Steigung der Geraden (Torsionssteifigkeit)
slope, intercept, r_value, p_value, std_err = linregress(angle_rad, torque_nm)

torsional_stiffness = slope

print(f"=== Ergebnisse der Auswertung ===")
print(f"Massen (g): {mass_g}")
print(f"Winkel (deg): {angle_deg}")
print(f"Berechnetes Drehmoment (Nm): {torque_nm}")
print(f"Berechnete Torsionssteifigkeit (k): {torsional_stiffness:.5f} Nm/rad")
print(f"Bestimmtheitsmaß (R^2): {r_value**2:.5f}")

# Visualisierung
plt.figure(figsize=(10, 6))
plt.scatter(angle_rad, torque_nm, color='blue', label='Messwerte', zorder=5)

# Fit-Linie plotten
fit_x = np.linspace(0, max(angle_rad), 100)
fit_y = intercept + slope * fit_x
plt.plot(fit_x, fit_y, color='red', linestyle='--', label=f'Linearer Fit\nk = {torsional_stiffness:.5f} Nm/rad')

plt.xlabel('Auslenkungswinkel (rad)')
plt.ylabel('Drehmoment (Nm)')
plt.title('Bestimmung der Torsionssteifigkeit')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# Speichern des Bildes im build-Ordner
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(script_dir))
build_dir = os.path.join(workspace_root, 'build')

os.makedirs(build_dir, exist_ok=True)
save_path = os.path.join(build_dir, 'torsionssteifigkeit_plot.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot wurde erfolgreich gespeichert unter: {save_path}")
