import numpy as np
from scipy.optimize import bisect
from pathlib import Path
import math_spirob.math_spirob as ms  # für Hilfsfunktionen

# do you want automatic xml formating of the output file? (reqires Mujoco, removes all comments but formats nicely)
auto_formating = True

if auto_formating:
    import mujoco as mj


# ==========================
# 1) Parameter & Mathematik
# ==========================
L_target   = 0.30               # gewünschte Mittelachsenlänge [m]
tip_d      = 0.01              # Spitzendurchmesser [m]
base_d     = 0.06              # Basisdurchmesser [m]
Delta_theta = np.deg2rad(30)    # Diskretisierungsschritt (30°)

#==========================

# Parameterobjekt
params = ms.SpiralParams(base_d=base_d, tip_d=tip_d, L_target=L_target)

# --- Solve b ---
f = ms.make_f_of_b(params)
b_sol = bisect(f, 1e-4, 1.0, xtol=1e-12, rtol=1e-12, maxiter=200)

# Abgeleitete Größen
theta0 = ms.theta0_from_ratio(b_sol, base_d=params.base_d, tip_d=params.tip_d)
a      = ms.a_from_tip(b_sol, tip_d=params.tip_d)
L_check = ms.length_central(a, b_sol, theta0)
phi_taper = ms.taper_angle_phi(b_sol)

# Diskretisierung
N_cont = (1.0 / (b_sol * Delta_theta)) * np.log(params.base_d / params.tip_d)
N = max(1, int(np.round(N_cont)))
new_Delta_theta = theta0 / N
thetas = np.linspace(0.0, theta0, N + 1)

# Breiten- und Längenfunktionen
delta_vals = ms.delta_width(thetas, a=a, b=b_sol)
factor = np.sqrt(b_sol**2 + 1.0) / b_sol
s_vals = factor * (ms.rho_c(thetas, a=a, b=b_sol) - ms.rho_c(0.0, a=a, b=b_sol))
Y_vals = L_check - s_vals
w_vals = 0.5 * delta_vals

seg_lengths = Y_vals[:-1] - Y_vals[1:]
seg_halfwidths = w_vals[:-1]

beta = np.exp(b_sol * new_Delta_theta)

# --- Alles schön zentral ausgeben ---
print("="*40)
print("SPIRALENPARAMETER & DISKRETISIERUNG")
print("="*40)
print(f"Ziel-Länge L_target       = {L_target:.6g} m")
print(f"Basis-Durchmesser          = {base_d:.6g} m")
print(f"Spitze-Durchmesser         = {tip_d:.6g} m")
print(f"Diskretisierungsschritt Δθ = {np.rad2deg(new_Delta_theta):.6f}° ({new_Delta_theta:.6g} rad)")
print(f"Anzahl Segmente N          = {N} (ungerundet: {N_cont:.6g})\n")

print("Gelöste Parameter:")
print(f"  b        = {b_sol:.6g}")
print(f"  theta0   = {theta0:.6g} rad = {np.rad2deg(theta0):.2f}°")
print(f"  a        = {a:.6g}")
print(f"  L_check  = {L_check:.6g} m (Ziel {L_target} m)  |Δ|={abs(L_check-L_target):.3e}")
print(f"  phi_taper= {np.rad2deg(phi_taper):.6f}°")
print(f"  beta     = {beta:.6g}\n")

# print("Segmentinformationen:")
# print(f"mittlere Segmentlänge = {np.mean(seg_lengths):.6g} m")
# print(f"min Segmentlänge      = {np.min(seg_lengths):.6g} m")
# print(f"max Segmentlänge      = {np.max(seg_lengths):.6g} m")
# print(f"mittlere Halbe Breite = {np.mean(seg_halfwidths):.6g} m")
# print(f"min Halbe Breite      = {np.min(seg_halfwidths):.6g} m")
# print(f"max Halbe Breite      = {np.max(seg_halfwidths):.6g} m\n")

# print("Zwischenwerte (theta, s, Y, delta, w):")
# print(f"{'theta(rad)':>10} | {'s(m)':>9} | {'Y(m)':>9} | {'delta(m)':>9} | {'w(m)':>9}")
# for i in range(len(thetas)):
#     print(f"{thetas[i]:10.6g} | {s_vals[i]:9.6g} | {Y_vals[i]:9.6g} | {delta_vals[i]:9.6g} | {w_vals[i]:9.6g}")
# print("\nSegmentwerte (Seg | seg_len | half_width):")
# print(f"{'Seg':>3} | {'seg_len(m)':>12} | {'half_width(m)':>14}")
# for i in range(N):
#     print(f"{i:3d} | {seg_lengths[i]:12.6g} | {seg_halfwidths[i]:14.6g}")
# print("="*40)

# ========================================
# MuJoCo-MJCF-Generator (Box-Kette)
# ========================================
def mjcf_header(model_name="spiral_chain"):
    return f'''<mujoco model="{model_name}">
  <compiler/>
  <option timestep="0.005" gravity="0 0 -9.81" impratio="10" iterations="50"/>
  <default>
    <joint damping="0.2" stiffness="0.01" limited="true" range="{-np.rad2deg(new_Delta_theta)+5} {np.rad2deg(new_Delta_theta)-5}" solimplimit="0.9 0.95 0.001" solreflimit="0.02 0.5" armature="0.01"/>
    <geom contype="1" conaffinity="0"/>
  </default>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="plane" size="2 2 0.1" rgba="0 0 1 0.6" contype="2" conaffinity="1"/>
      <!-- Der eigentliche Ketten-Root wird hier als Kind erzeugt -->
'''
def worldbody_footer():
    return '''    </body>
  </worldbody>
'''

def mjcf_footer():
    return '''
</mujoco>
'''

NUM_CABLES = 2

SITE_SIZE      = 0.001     # sichtbare Größe der Site-Kugeln

def body_block(i, seg_len, half_width, add_color=False, gap=0.002):

    half_vis_len = seg_len/2.0

    hx = float(half_width)
    hy = float(half_width)
    hz = float(half_vis_len)

    a = -np.tan(np.pi/2 - (phi_taper/2))
    b = -1
    c = np.tan(np.pi/2 - (phi_taper/2)) * ((half_width * beta) - 0.003)

    solv0 = ms.solve_for_points(a = a, b = b, c = c, theta_deg=np.rad2deg(new_Delta_theta))

    xC0, yC0 = solv0["C"]
    xD0, yD0 = solv0["D"]

    x_in = xD0
    y_in = 0
    z_in = yD0

    c = np.tan(np.pi/2 - (phi_taper/2)) * ((half_width ) - 0.003)

    solv1 = ms.solve_for_points(a = a, b = b, c = c, theta_deg=np.rad2deg(new_Delta_theta))

    xC1, yC1 = solv1["C"]
    xD1, yD1 = solv1["D"]

    x_out = xC1
    y_out = 0
    z_out = yC1 + seg_len

    rgba = '0.6 0.75 0.95 0.3' if add_color else '0.2 0.7 0.2 0.3'

#        <geom type="sphere" size="{r0}" rgba="0.9 0.0 0.0 0.2"/>  armature="0.00001"
    return f'''      <body name="seg_{i}" pos="0 0 0">
        <joint name="j_{i}" type="hinge" axis="0 1 0" pos="0 0 0" stiffness="0.05" damping="0.05" limited="true" range="{-np.rad2deg(new_Delta_theta)+0.1} {np.rad2deg(new_Delta_theta)-0.1}" solimplimit="0.9 0.95 0.001" solreflimit="0.01 0.5"/>

        <!-- sichtbare, kollisionslose box-->
        <geom name="g_{i}" type="box"
              size="{hx:.6g} {hy:.6g} {hz:.6g}"
              pos="0 0 {hz:.6g}"
              rgba="{rgba}" contype="1" conaffinity="0" density="1100"/>

        <!-- Tendon-Sites: unten (in) / oben (out) auf äußerster Kante ±x -->
        <site name="site_in_{i}_0"  pos="{x_in:.6g} {y_in:.6g} {z_in:.6g}"  size="{SITE_SIZE}" rgba="1 1 0 1"/>
        <site name="site_out_{i}_0" pos="{x_out:.6g} {y_out:.6g} {z_out:.6g}" size="{SITE_SIZE}" rgba="1 1 0 1"/>
        <site name="site_in_{i}_1"  pos="{-x_in:.6g} {y_in:.6g} {z_in:.6g}"  size="{SITE_SIZE}" rgba="1 1 0 1"/>
        <site name="site_out_{i}_1" pos="{-x_out:.6g} {y_out:.6g} {z_out:.6g}" size="{SITE_SIZE}" rgba="1 1 0 1"/>

        <!-- Attachment-Punkt für das nächste Segment am Segmentende: -->
        <body name="seg_{i}_end" pos="0 0 {seg_len:.6g}">
'''

def close_body_block():
    # Schließt seg_i_end und seg_i
    return '''        </body>
      </body>
'''

def tendons_xml(num_segments):
    parts = ['  <tendon>']
    for k in range(NUM_CABLES):
        parts.append(f'    <spatial name="tendon_{k}" width="0.001" rgba="1 0 0 1" frictionloss="0.1" stiffness="50">')
        for i in reversed(range(num_segments)):
            parts.append(f'      <site site="site_in_{i}_{k}"/>')
            parts.append(f'      <site site="site_out_{i}_{k}"/>')
        parts.append('    </spatial>')
    parts.append('  </tendon>\n')
    return "\n".join(parts)

def actuators_xml():
    lines = ['  <actuator>']
    for k in range(NUM_CABLES):
        lines.append(
          f'    <position name="tendon_act_{k}" tendon="tendon_{k}" '
          f'kp="200.0" forcerange="-200 0" ctrlrange="0.00 0.5"/>'
        )
    lines.append('  </actuator>\n')
    return "\n".join(lines)

def sensors_xml():
    lines = ['  <sensor>']
    for k in range(NUM_CABLES):
        lines.append(f'    <tendonpos name="tendon{k}_pos" tendon="tendon_{k}"/>')
        lines.append(f'    <tendonvel name="tendon{k}_vel" tendon="tendon_{k}"/>')
    lines.append('  </sensor>\n')
    return "\n".join(lines)



def build_chain_xml(seg_lengths, seg_halfwidths, model_name="spiral_chain"):
    """
    Erzeugt eine hierarchische Kette:
      base
        seg_0
          seg_0_end
            seg_1
              seg_1_end
                ...
    Joint i sitzt am Beginn von seg_i (Pivot).
    """
    assert len(seg_lengths) == len(seg_halfwidths)
    N = len(seg_lengths)

    xml = [mjcf_header(model_name)]

    for i in reversed(range(N)):
        add_color = (i % 2 == 0)
        xml.append(body_block(i, seg_lengths[i], seg_halfwidths[i], add_color=add_color))

    # und dann in der gleichen Reihenfolge wieder schließen:
    for i in reversed(range(N)):
        xml.append(close_body_block())

    xml.append(worldbody_footer())
    xml.append(tendons_xml(N))
    xml.append(actuators_xml())
    xml.append(sensors_xml())
    xml.append(mjcf_footer())

    full = "".join(xml)
    return full

# =============================
# 3) XML erzeugen & abspeichern
# =============================
xml_string = build_chain_xml(seg_lengths, seg_halfwidths, model_name="spiral_chain")
if auto_formating:
    spec = mj.MjSpec.from_string(xml_string)
    xml_string = spec.to_xml()
out_path = Path("spiral_chain.xml")
out_path.write_text(xml_string, encoding="utf-8")

print(f"\nMJCF exportiert nach: {out_path.resolve()}")
