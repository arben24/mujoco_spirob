#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import bisect
from pathlib import Path
import pandas as pd

# ==========================
# 1) Parameter & Mathematik
# ==========================
L_target   = 0.25               # gewünschte Mittelachsenlänge [m]
tip_d      = 0.01              # Spitzendurchmesser [m]
base_d     = 0.080              # Basisdurchmesser [m]
Delta_theta = np.deg2rad(20)    # Diskretisierungsschritt (30°)

# --- Hilfsfunktionen aus deiner Vorlage ---
def rho(theta, a, b):
    return a * np.exp(b * theta)

def rho_c(theta, a, b):
    return 0.5 * a * (np.exp(2*np.pi*b) + 1.0) * np.exp(b * theta)

def length_central(a, b, theta0):
    return (np.sqrt(b**2 + 1.0) / b) * (rho_c(theta0, a, b) - rho_c(0.0, a, b))

def delta_width(theta, a, b):
    # δ(θ) = a ( e^{b(θ+2π)} − e^{bθ} )
    return a * (np.exp(b*(theta + 2*np.pi)) - np.exp(b*theta))

def theta0_from_ratio(b, base_d, tip_d):
    return np.log(base_d / tip_d) / b

def a_from_tip(b, tip_d):
    return tip_d / (np.exp(2*np.pi*b) - 1.0)

def L_from_b(b, base_d, tip_d):
    th0 = theta0_from_ratio(b, base_d, tip_d)
    a_b = a_from_tip(b, tip_d)
    return length_central(a_b, b, th0)

def f_of_b(b):
    return L_from_b(b, base_d, tip_d) - L_target

def taper_angle_phi(b):
    num = b * (np.exp(2*np.pi*b) - 1.0)
    den = np.sqrt(b**2 + 1.0) * (np.exp(2*np.pi*b) + 1.0)
    return 2.0 * np.arctan(num / den)  # in radians

# --- Solve b with bisect ---
b_sol   = bisect(f_of_b, 1e-4, 1.0, xtol=1e-12, rtol=1e-12, maxiter=200)
theta0  = theta0_from_ratio(b_sol, base_d, tip_d)
print(theta0)
a       = a_from_tip(b_sol, tip_d)
L_check = length_central(a, b_sol, theta0)
phi_taper = taper_angle_phi(b_sol)

# Anzahl Segmente aus θ-Spanne mit Mindestanzahl
N = 1 / (b_sol * Delta_theta) * np.log(base_d / tip_d)

print("==========================")
print("Anzahl Segmente N =", N)
N = int(np.round(N))
print("Anzahl Segmente gerundet N =", N)
new_Delta_theta = theta0 / N
print("Neuer Diskretisierungsschritt Δθ =", np.rad2deg(new_Delta_theta), "°", f"({new_Delta_theta:.6g} rad)")
print("===========================")

#================================================
# Punkte
theta = np.linspace(0, theta0, 100)
x = a*np.exp(b_sol*theta)*np.cos(theta)
y = a*np.exp(b_sol*theta)*np.sin(theta)

# CSV
df = pd.DataFrame({'x [m]': x, 'y [m]': y})
#df.to_csv('spiral_xy.csv', sep='\t', index=False, float_format='%.6f')

#================================================

# Gleichmäßige Diskretisierung in θ
thetas     = np.linspace(0.0, theta0, N+1)
print(len(thetas))
delta_vals = delta_width(thetas, a, b_sol)
s_vals     = (np.sqrt(b_sol**2+1)/b_sol) * (rho_c(thetas, a, b_sol) - rho_c(0, a, b_sol))
Y_vals     = L_check - s_vals          # 0 an der Basis, L an der Spitze (wie bei dir)
w_vals     = 0.5 * delta_vals          # halbe Breite
print(thetas)
print(s_vals)
print(Y_vals)
print(w_vals)
# Segmentlängen in "entrollter" Darstellung:
seg_lengths = (Y_vals[:-1] - Y_vals[1:])  # positive Werte

# Querschnitts-Halbe (gemittelt pro Segment)
seg_halfwidths = w_vals[:-1]
print("=========================")
print(seg_halfwidths)




print("=========================")
print("Spiral-Parameter & Diskretisierung")
print("=========================")
print(f"  Ziel-Länge L_target = {L_target} m")
print(f"  Basis-Durchmesser    = {base_d} m")
print(f"  Spitze-Durchmesser   = {tip_d} m")
print(f"  Diskretisierungsschritt Δθ = {np.rad2deg(new_Delta_theta):.1f}°")
print(f"  Gelöste Parameter:")
print(f"    b       = {b_sol:.6g}")
print(f"    theta0  = {theta0:.6g} rad = {np.rad2deg(theta0):.2f} deg")
print(f"    a       = {a:.6g}")
print(f"    L_check = {L_check:.6g} m  (Ziel {L_target} m)  |Δ|={abs(L_check-L_target):.3e}")
print(f"  Diskretisierung in N={N} Segmente")
print(f"    mittlere Segmentlänge = {np.mean(seg_lengths):.6g} m")
print(f"    min Segmentlänge      = {np.min(seg_lengths):.6g} m")
print(f"    max Segmentlänge      = {np.max(seg_lengths):.6g} m")
print(f"    mittlere Halbe Breite = {np.mean(seg_halfwidths):.6g} m")
print(f"    min Halbe Breite      = {np.min(seg_halfwidths):.6g} m")
print(f"    max Halbe Breite      = {np.max(seg_halfwidths):.6g} m")
print("=========================")

print("Zwischenwerte")
print(" theta (rad) |   s (m)   |   Y (m)   | delta (m) |  w=delta/2 (m)")
for i in range(len(thetas)):
    print(f" {thetas[i]:10.6g} | {s_vals[i]:9.6g} | {Y_vals[i]:9.6g} | {delta_vals[i]:9.6g} | {w_vals[i]:9.6g}")
print("=========================")

print("Segmentwerte")
print(" seg |  seg_len (m) | half_width (m)")
for i in range(N):
    print(f" {i:3d} | {seg_lengths[i]:12.6g} | {seg_halfwidths[i]:14.6g}")
print("=========================")

# Kumulative Gelenkwinkel (Pose, die der Δθ-Auflösung folgt)
per_joint_angle = np.full(N, theta0 / N)   # gleichmäßig verteilte Winkel summieren sich zu theta0
qpos_angles = np.cumsum(per_joint_angle)   # kumuliert, nur für Keyframe (rein visuell/praktisch)

beta = np.exp(b_sol*new_Delta_theta)

print("Gelöste Parameter:")
print(f"  b      = {b_sol:.6g}")
print(f"  theta0 = {theta0:.6g} rad = {np.rad2deg(theta0):.2f} deg")
print(f"  a      = {a:.6g}")
print(f"  L_check= {L_check:.6g} m (Ziel {L_target} m)  |Δ|={abs(L_check-L_target):.3e}")
print(f"  beta   = {beta:.6g}")

# ========================================
# 2) MuJoCo-MJCF-Generator (Box-Kette)
# ========================================
def mjcf_header(model_name="spiral_chain"):
    return f'''<mujoco model="{model_name}">
  <compiler/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <size njmax="1000" nconmax="1000"/>
  <default>
    <joint damping="0.02" limited="true" range="{-np.rad2deg(new_Delta_theta)+1} {np.rad2deg(new_Delta_theta)-1}"/>
    <geom  friction="0.8 0.1 0.1" density="1200" rgba="0.7 0.7 0.8 1" contype="1" conaffinity="0"/>
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

# ---- feste Einstellungen für 2 Seile (±x) ----
SITE_SIZE      = 0.001     # sichtbare Größe der Site-Kugeln

def body_block(i, seg_len, half_width, add_color=False, gap=0.002):
    """
    Erzeugt:
      - body 'seg_i' (Pivot am Body-Ursprung, Joint um y)
      - verkürzte sichtbare Box mit Lücke 'gap'
      - 2x Sites pro Ende (in/out) für 2 Tendons (±x-Außenkante)
      - child-attachment 'seg_i_end' am Segmentende (z=seg_len)
    Kette entlang +z. Kinematik bleibt unverändert.
    """
    # sichtbare Halblänge so, dass zwischen Segmenten die Lücke = gap entsteht
    half_vis_len = seg_len/2.0#max(seg_len/2.0 - gap/2.0, 1e-6)

    hole_size = 0.005

    hx = float(half_width)
    hy = float(half_width)
    hz = float(half_vis_len)

    #d = half_width * beta -0.005

    d = half_width * beta - 0.003     #x position of the hole of an segment before the current one 

    x_in = d * np.cos(new_Delta_theta)

    z_in = d * np.sin(new_Delta_theta)
    y_in = 0

    #x_out = half_width - 0.003
    y_out = 0
    z_out = seg_len
    # x_test = 1
    # hyp = seg_len / np.cos(phi_taper)
    # x_out_z = hyp * np.sin(phi_taper)
    # x_out = d - x_out_z
    # print("-----------")
    # print(x_test)
    # print(d-x_in)
    x_out = half_width - 0.003
    #x_out = d - (hyp * np.sin(phi_taper))

    rgba = '0.6 0.75 0.95 0.3' if add_color else '0.2 0.7 0.2 0.3'

    return f'''      <body name="seg_{i}" pos="0 0 0">
        <joint name="j_{i}" type="hinge" axis="0 1 0" pos="0 0 0" stiffness="0.0" damping="0.06" limited="true" range="{-np.rad2deg(new_Delta_theta)+1} {np.rad2deg(new_Delta_theta)-1}"/>

        <!-- sichtbare, kollisionslose Box mit Lücke -->
        <geom name="g_{i}" type="box"
              size="{hx:.6g} {hy:.6g} {hz:.6g}"
              pos="0 0 {hz:.6g}"
              rgba="{rgba}" contype="1" conaffinity="0"/>

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
        parts.append(f'    <spatial name="tendon_{k}" width="0.001" rgba="1 0 0 1">')
        for i in reversed(range(num_segments)):
            parts.append(f'      <site site="site_in_{i}_{k}"/>')
            parts.append(f'      <site site="site_out_{i}_{k}"/>')
        parts.append('    </spatial>')
    parts.append('  </tendon>\n')
    return "\n".join(parts)

def actuators_xml():
    lines = ['  <actuator>']
    # Tipp: ctrlrange grob anpassen; hängt von deiner Kettenlänge/Lücke ab
    for k in range(NUM_CABLES):
        lines.append(
          f'    <position name="tendon_act_{k}" tendon="tendon_{k}" '
          f'kp="200.0" forcerange="-200 0" ctrlrange="0.05 0.35"/>'
        )
    lines.append('  </actuator>\n')
    return "\n".join(lines)

def sensors_xml():
    lines = ['  <sensor>']
    for k in range(NUM_CABLES):
        lines.append(f'    <tendonpos name="tendon{k}_pos" tendon="tendon_{k}"/>')
        lines.append(f'    <tendonvel name="tendon{k}_vel" tendon="tendon_{k}"/>')
        #lines.append(f'    <tendonfrc name="tendon{k}_frc" tendon="tendon_{k}"/>')
    lines.append('  </sensor>\n')
    return "\n".join(lines)



def build_chain_xml(seg_lengths, seg_halfwidths, per_joint_angle_rad=None, model_name="spiral_chain"):
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
    indent = ""  # wir verwenden fixe Blocks mit Einrückungen oben

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



    # Füge Keyframe vor </mujoco> ein:
    full = "".join(xml)
    #full = full.replace('</mujoco>', keyframe + '</mujoco>')
    return full

# =============================
# 3) XML erzeugen & abspeichern
# =============================
xml_string = build_chain_xml(seg_lengths, seg_halfwidths, per_joint_angle_rad=per_joint_angle, model_name="spiral_chain")
out_path = Path("spiral_chain.xml")
out_path.write_text(xml_string, encoding="utf-8")

print(f"\nMJCF exportiert nach: {out_path.resolve()}")
