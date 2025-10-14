import numpy as np
import matplotlib.pyplot as plt

def visualize_spiral_segment_angle(a=0.0024, b=0.2, Delta_theta=np.deg2rad(30), i_segment=2):
    """
    Zeigt zwei benachbarte Segmente (Vektoren) auf der logarithmischen Spirale
    und visualisiert ihren geometrischen Winkel.
    """
    # Spiralpunkte
    N = 10
    thetas = np.linspace(0, (N)*Delta_theta, N+1)
    rc = 0.5 * a * (np.exp(2*np.pi*b) + 1.0) * np.exp(b * thetas)
    x = rc * np.cos(thetas)
    y = rc * np.sin(thetas)
    
    # Segmentvektoren
    v1 = np.array([x[i_segment+1]-x[i_segment], y[i_segment+1]-y[i_segment]])
    v2 = np.array([x[i_segment+2]-x[i_segment+1], y[i_segment+2]-y[i_segment+1]])
    
    # Winkel zwischen den Vektoren
    cos_phi = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    cos_phi = np.clip(cos_phi, -1, 1)
    phi = np.arccos(cos_phi)
    
    # Plot
    plt.figure(figsize=(6,6))
    plt.plot(x, y, 'gray', lw=1, label='Spirale')
    plt.scatter(x, y, s=10, color='black')
    
    # Startpunkt für beide Vektoren
    P = np.array([x[i_segment+1], y[i_segment+1]])
    
    # Normierte Richtungen für Zeichnung
    v1u = v1 / np.linalg.norm(v1)
    v2u = v2 / np.linalg.norm(v2)
    scale = np.linalg.norm(v1) * 1.2
    
    # Zeichne Vektoren
    plt.arrow(P[0], P[1], -v1u[0]*scale, -v1u[1]*scale, 
              head_width=0.002, color='r', length_includes_head=True, label='v1')
    plt.arrow(P[0], P[1], v2u[0]*scale, v2u[1]*scale, 
              head_width=0.002, color='b', length_includes_head=True, label='v2')
    
    # Kleiner Bogen zur Winkelanzeige
    angle_points = np.linspace(0, phi, 50)
    r = np.linalg.norm(v1)*0.4
    arc_x = P[0] + r*np.cos(angle_points)
    arc_y = P[1] + r*np.sin(angle_points)
    plt.plot(arc_x, arc_y, 'k--')
    plt.text(P[0]+r*0.6, P[1]+r*0.3, f"{np.degrees(phi):.2f}°", fontsize=12, weight='bold')
    
    plt.axis('equal')
    plt.title("Geometrischer Winkel zwischen zwei Spiral-Segmenten")
    plt.legend()
    plt.show()

# Beispielaufruf
visualize_spiral_segment_angle()

