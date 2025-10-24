import numpy as np
import math

# Eingabe
seg_len = np.array([
 0.00544258, 0.00601223, 0.0066415 , 0.00733664, 0.00810453,
 0.00895279, 0.00988983, 0.010925  , 0.0120684 , 0.0133316 ,
 0.0147269 , 0.0162683 , 0.017971  , 0.019852  , 0.0219298 ,
 0.024225  , 0.0267606 , 0.0295615
])
half_width = np.array([
 0.005, 0.00552333, 0.00610142, 0.00674003, 0.00744548,
 0.00822476, 0.0090856 , 0.0100365 , 0.011087 , 0.0122474 ,
 0.0135293 , 0.0149454 , 0.0165096 , 0.0182376 , 0.0201465 ,
 0.0222551 , 0.0245844 , 0.0271576
])

phi_taper = np.deg2rad(10.98)   # dein "gewünschter" Taper (zur z-Achse)
b_sol = 0.186  # <--- hier deinen b-Wert einsetzen (aus your bisect)

# 1) projizierte axiale Höhe (richtige Stapelung)
z_proj = np.cumsum(seg_len * np.cos(phi_taper))

# 2) K aus der Spiral-Definition: half_width = K * rho
K = 0.5 * (np.exp(2*np.pi*b_sol) - 1.0)

# 3) Radius aus half_width rekonstruieren (rho = half_width / K)
rho = half_width / K

# --- Fit A: z gegen RADIUS rho  -> sollte phi_taper ergeben
m_rho, b_rho = np.polyfit(rho, z_proj, 1)
phi_from_rho = math.degrees(math.atan(1.0 / m_rho))  # Winkel zur z-Achse

# --- Fit B: z gegen half_width  -> ergibt kleineren Winkel arctan(K*tan phi)
m_hw, b_hw = np.polyfit(half_width, z_proj, 1)
phi_from_hw = math.degrees(math.atan(1.0 / m_hw))

# Erwartungswert für "half_width-Zielwinkel":
phi_hw_target = math.degrees(math.atan(K * math.tan(phi_taper)))

print("=== Konstanten ===")
print(f"b = {b_sol:.6f}")
print(f"K = 0.5*(e^(2πb)-1) = {K:.6f}\n")

print("=== Fit A: z vs rho ===")
print(f"z = {m_rho:.6f} * rho + {b_rho:.6f}")
print(f"Taper aus rho-fit: {phi_from_rho:.3f}°   (Soll: {math.degrees(phi_taper):.3f}°)\n")

print("=== Fit B: z vs half_width ===")
print(f"z = {m_hw:.6f} * half_width + {b_hw:.6f}")
print(f"Taper aus hw-fit:  {phi_from_hw:.3f}°   (Soll-hw: {phi_hw_target:.3f}° = arctan(K*tan(phi)))")
