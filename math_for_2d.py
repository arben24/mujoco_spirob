import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

# Zielgrößen wie zuvor
L_target = 0.25        # desired central-axis length [m]
tip_d = 0.005          # tip diameter [m]
base_d = 0.080         # base diameter [m]
Delta_theta = np.deg2rad(30)  # discretization step (30°)
N = int(np.ceil(np.log(base_d/tip_d) / (Delta_theta)))
print(N)
N = max(N, 4)
print(N)

# ---------- Helper functions ----------
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

# --- NEW: taper angle φ from Eq. (3)
def taper_angle_phi(b):
    num = b * (np.exp(2*np.pi*b) - 1.0)
    den = np.sqrt(b**2 + 1.0) * (np.exp(2*np.pi*b) + 1.0)
    return 2.0 * np.arctan(num / den)  # in radians

import numpy as np

def segment_vector_angles(rc, theta):
    """
    Berechnet den Winkel zwischen aufeinanderfolgenden Segmentvektoren
    auf einer logarithmischen Spirale.
    
    rc: Array der Radien (z. B. rho_c(theta, a, b))
    theta: Array der Winkel in Radiant (gleiche Länge)
    """
    # Punkte auf der Spirale
    x = rc * np.cos(theta)
    y = rc * np.sin(theta)
    
    # Segmentvektoren
    v = np.stack([np.diff(x), np.diff(y)], axis=1)  # (N,2)
    
    # Winkel zwischen aufeinanderfolgenden Segmentvektoren
    angles = []
    for i in range(len(v)-1):
        a = v[i]
        b = v[i+1]
        cos_phi = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        # numerische Rundungsfehler vermeiden
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        phi = np.arccos(cos_phi)
        angles.append(phi)
        
    return np.array(angles), np.rad2deg(angles)


# ---------- Step 2.1: Solve for b with scipy.optimize.bisect ----------
b_sol = bisect(f_of_b, 1e-4, 1.0, xtol=1e-12, rtol=1e-12, maxiter=100)
theta0 = theta0_from_ratio(b_sol, base_d, tip_d)
a = a_from_tip(b_sol, tip_d)
L_check = length_central(a, b_sol, theta0)
phi = taper_angle_phi(b_sol)  # NEW


print("Gelöste Parameter:")
print(f"  b      = {b_sol:.6g}")
print(f"  theta0 = {theta0:.6g} rad = {np.rad2deg(theta0):.2f} deg")
print(f"  a      = {a:.6g}")
print(f"  L_check= {L_check:.6g} m (Ziel {L_target} m)  |Δ|={abs(L_check-L_target):.3e}")
print(f"  phi     = {phi:.6g} rad = {np.rad2deg(phi):.2f} deg  <-- Taper-Winkel")


# ---------- Step 2.2/2.3: Discretization with equal Δθ ----------
thetas = np.linspace(0, theta0, N+1)
rc_vals = rho_c(thetas, a, b_sol)
r_vals = rho(thetas, a, b_sol)
delta_vals = delta_width(thetas, a, b_sol)
s_vals = (np.sqrt(b_sol**2+1)/b_sol) * (rho_c(thetas, a, b_sol) - rho_c(0, a, b_sol))
Y_vals = L_check - s_vals
w_vals = 0.5 * delta_vals

#compute angles between each element 
angles_rad, angles_deg = segment_vector_angles(rc_vals, thetas)
print(np.round(angles_deg, 3))
print(f"Mittelwert: {np.mean(angles_deg):.3f}°")


# ---------- Plots ----------
# Spiral
theta_dense = np.linspace(0, theta0, 800)
r_dense = rho(theta_dense, a, b_sol)
x_dense = r_dense * np.cos(theta_dense)
y_dense = r_dense * np.sin(theta_dense)

plt.figure(figsize=(6,6))
plt.plot(x_dense, y_dense, label="Spiral r(θ)")
plt.axis("equal"); plt.title("Logarithmische Spirale mit gelöstem b")
plt.legend(); plt.show()

# Breite δ(θ)
plt.figure(figsize=(6,4))
plt.plot(thetas, delta_vals)
plt.xlabel("θ [rad]"); plt.ylabel("Width δ(θ) [m]")
plt.title("Segmentbreite δ(θ)")
plt.show()

# Flachlayout (Uncurled)
plt.figure(figsize=(6,8))
for i in range(N):
    y0, y1 = Y_vals[i], Y_vals[i+1]
    w0, w1 = w_vals[i], w_vals[i+1]
    xs = np.array([-w0,  w0,  w1, -w1, -w0])
    ys = np.array([ y0,  y0,  y1,  y1,  y0])
    plt.plot(xs, ys)
plt.gca().invert_yaxis()
plt.xlabel("x [m]"); plt.ylabel("Y (entrollt) [m]")
plt.title("Uncurled (flache) Anordnung der Segmente")
plt.show()

