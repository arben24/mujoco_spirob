import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

# =========================
# Zielgrößen / Spezifikation
# =========================
L_target = 0.25        # desired central-axis length [m]
tip_d    = 0.005       # tip diameter [m]
base_d   = 0.080       # base diameter [m]

# =========================
# Eingabe: Gewichte für die Winkelverteilung
# Länge = Anzahl Segmente N; Werte > 0
# Beispiele:
# weights = np.ones(8)                         # gleichmäßige Verteilung
# weights = np.array([1,1,1,1, 1.2,1.4,1.6,2]) # mehr Winkel Richtung Spitze
# =========================
weights = np.array([1, 1, 1, 1, 1.2, 1.4, 1.6, 2.0], dtype=float)

# =========================
# Helper-Funktionen
# =========================
def rho(theta, a, b):
    return a * np.exp(b * theta)

def rho_c(theta, a, b):
    return 0.5 * a * (np.exp(2*np.pi*b) + 1.0) * np.exp(b * theta)

def length_central(a, b, theta0):
    return (np.sqrt(b**2 + 1.0) / b) * (rho_c(theta0, a, b) - rho_c(0.0, a, b))

def delta_width(theta, a, b):
    return a * (np.exp(b*(theta + 2*np.pi)) - np.exp(b*theta))

def theta0_from_ratio(b, base_d, tip_d):
    return np.log(base_d / tip_d) / b

def a_from_tip(b, tip_d):
    return tip_d / (np.exp(2*np.pi*b) - 1.0)

def L_from_b(b):
    th0 = theta0_from_ratio(b, base_d, tip_d)
    a_b = a_from_tip(b, tip_d)
    return length_central(a_b, b, th0)

def f_of_b(b):
    return L_from_b(b) - L_target

def taper_angle_phi(b):  # Eq.(3) aus dem Paper
    num = b * (np.exp(2*np.pi*b) - 1.0)
    den = np.sqrt(b**2 + 1.0) * (np.exp(2*np.pi*b) + 1.0)
    return 2.0 * np.arctan(num / den)

def segment_vector_angles(rc, theta):
    """Geometrischer Winkel zwischen aufeinanderfolgenden Sekanten auf der Spirale (nur Info)."""
    x = rc * np.cos(theta)
    y = rc * np.sin(theta)
    v = np.stack([np.diff(x), np.diff(y)], axis=1)  # (N,2)
    if len(v) < 2:
        return np.array([]), np.array([])
    ang = []
    for i in range(len(v)-1):
        a = v[i]; b = v[i+1]
        cos_phi = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        ang.append(np.arccos(cos_phi))
    ang = np.array(ang)
    return ang, np.rad2deg(ang)

# =========================
# 1) b lösen (aus L_target & Durchmesserverhältnis), dann a, theta0
# =========================
b_sol  = bisect(f_of_b, 1e-4, 1.0, xtol=1e-12, rtol=1e-12, maxiter=100)
theta0 = theta0_from_ratio(b_sol, base_d, tip_d)
a      = a_from_tip(b_sol, tip_d)
L_check = length_central(a, b_sol, theta0)
phi = taper_angle_phi(b_sol)

print("Gelöste Spiral-Parameter:")
print(f"  b       = {b_sol:.6g}")
print(f"  theta0  = {theta0:.6g} rad = {np.rad2deg(theta0):.2f} deg")
print(f"  a       = {a:.6g}")
print(f"  L_check = {L_check:.6g} m  (Ziel {L_target} m)  |Δ|={abs(L_check-L_target):.3e}")
print(f"  phi     = {phi:.6g} rad = {np.rad2deg(phi):.2f} deg  <-- Taper-Winkel Eq.(3)")

# =========================
# 2) Gewichte -> mechanische Δθ_i
# =========================
if np.any(weights < 0) or np.all(weights == 0):
    raise ValueError("Gewichte müssen > 0 sein und nicht alle 0.")

theta0_deg = np.rad2deg(theta0)
weights_norm = weights / weights.sum()
dtheta_mech_deg = weights_norm * theta0_deg          # mechanische Gelenkwinkel in Grad
dtheta_mech_rad = np.deg2rad(dtheta_mech_deg)        # in Rad
N = len(dtheta_mech_deg)

# Knotenwinkel (Stützstellen) durch kumulative Summe
thetas = np.concatenate([[0.0], np.cumsum(dtheta_mech_rad)])
# numerische Absicherung
thetas[-1] = theta0

print("\nMechanische Gelenkwinkel aus Gewichten (deg):")
print(np.round(dtheta_mech_deg, 3))
print(f"Summe = {dtheta_mech_deg.sum():.3f}°  (Soll: {theta0_deg:.3f}°)")

# =========================
# 3) Geometrie an den Stützstellen
# =========================
rc_vals    = rho_c(thetas, a, b_sol)
r_vals     = rho(thetas, a, b_sol)
delta_vals = delta_width(thetas, a, b_sol)

# Bogenlänge vom Tip:
s_vals = (np.sqrt(b_sol**2+1)/b_sol) * (rho_c(thetas, a, b_sol) - rho_c(0, a, b_sol))
Y_vals = L_check - s_vals            # flaches Mapping: Y=0 Basis, Y=L Spitze
w_vals = 0.5 * delta_vals            # Halbbreiten

# Geometrische (2D) Sekanten-Winkel (nur zur Info/Visualisierung)
sec_rad, sec_deg = segment_vector_angles(rc_vals, thetas)
if len(sec_deg) > 0:
    print("\nGeometrische Winkel zwischen Sekanten (deg) (Info):")
    print(np.round(sec_deg, 3))
    print(f"Mittelwert Sekanten-Winkel: {np.mean(sec_deg):.3f}°")

# =========================
# 4) Plots
# =========================
# 4a) Spiral (gewickelt) mit Stützstellen
theta_dense = np.linspace(0, theta0, 800)
r_dense = rho(theta_dense, a, b_sol)
x_dense = r_dense * np.cos(theta_dense)
y_dense = r_dense * np.sin(theta_dense)

plt.figure(figsize=(6,6))
plt.plot(x_dense, y_dense, label="Spiral r(θ)")
xk = rc_vals * np.cos(thetas)
yk = rc_vals * np.sin(thetas)
plt.scatter(xk, yk, s=15)
plt.axis("equal"); plt.title("Logarithmische Spirale (Δθ_i aus Gewichten)")
plt.legend(); plt.show()

# 4b) Breite δ(θ) an den Stützstellen
plt.figure(figsize=(6,4))
plt.plot(thetas, delta_vals)
plt.xlabel("θ [rad]"); plt.ylabel("Width δ(θ) [m]")
plt.title("Segmentbreite δ(θ) an den Stützstellen")
plt.show()

# 4c) Flachlayout (Uncurled)
plt.figure(figsize=(6,8))
for i in range(N):
    y0, y1 = Y_vals[i], Y_vals[i+1]
    w0, w1 = w_vals[i], w_vals[i+1]
    xs = np.array([-w0,  w0,  w1, -w1, -w0])   # zentriert um Mittelachse
    ys = np.array([ y0,  y0,  y1,  y1,  y0])
    plt.plot(xs, ys)
plt.gca().invert_yaxis()
plt.xlabel("x [m]"); plt.ylabel("Y (entrollt) [m]")
plt.title("Uncurled (flache) Anordnung der Segmente – Δθ_i aus Gewichten")
plt.show()
