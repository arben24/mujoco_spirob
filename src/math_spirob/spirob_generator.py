import numpy as np
from scipy.optimize import bisect
from pathlib import Path
import mujoco as mj
import math_spirob.math_spirob as ms


# ===================================================================
# 1) High-Level API (öffentliche Funktionen)
# ===================================================================

def generate_xml_string(
    L_target: float,
    base_d: float,
    tip_d: float,
    Delta_theta_deg: float = 30.0,
    model_name: str = "spiral_chain",
    auto_format: bool = False,
):
    """
    Generate an MJCF XML string representing a discretized logarithmic spiral chain.

    This function solves the spiral parameters, computes the segment geometry,
    discretizes the chain, and constructs a complete MJCF model using ``XMLBuilder``.

    Parameters
    ----------
    L_target : float
        Desired total centerline length of the spiral.
    base_d : float
        Diameter at the base of the spiral (start).
    tip_d : float
        Diameter at the tip of the spiral (end).
    Delta_theta_deg : float, optional
        Target angular resolution per segment in degrees.
        The final angular resolution may differ slightly due to rounding.
    model_name : str, optional
        Name of the root model in the generated MJCF XML.
    auto_format : bool, optional
        If True, reformat the generated XML using MuJoCo’s ``MjSpec`` for
        consistent indentation and structure.

    Returns
    -------
    str
        The generated MJCF XML model as a string.
    """

    calc = SpiralCalculator(
        L_target=L_target,
        base_d=base_d,
        tip_d=tip_d,
        Delta_theta_deg=Delta_theta_deg,
    )

    geometry = calc.compute_geometry()

    xml = XMLBuilder(
        geometry.seg_lengths,
        geometry.seg_halfwidths,
        geometry.phi_taper,
        geometry.new_Delta_theta,
        geometry.beta,
    ).build(model_name=model_name)

    if auto_format:
        spec = mj.MjSpec.from_string(xml)
        xml = spec.to_xml()

    return xml


def generate_and_save_xml(
    filepath: str | Path,
    **kwargs,
):
    """
    Generate an MJCF XML file and save it to disk.

    This is a thin wrapper around :func:`generate_xml_string`.
    All keyword arguments are passed directly to that function.

    Parameters
    ----------
    filepath : str or Path
        Destination path where the XML file will be written.
    **kwargs
        Additional keyword arguments forwarded to :func:`generate_xml_string`.

    Returns
    -------
    pathlib.Path
        The final path of the saved XML file.
    """
    xml = generate_xml_string(**kwargs)
    filepath = Path(filepath)
    filepath.write_text(xml, encoding="utf-8")
    return filepath


# ===================================================================
# 2) Berechnungsteil (vormals Top-Level Code)
# ===================================================================

class SpiralGeometry:
    """
    Container class for all geometric quantities of the discretized spiral.

    Parameters
    ----------
    seg_lengths : array_like
        Length of each segment along the chain.
    seg_halfwidths : array_like
        Half-width of each segment (radius), controlling the box dimensions.
    phi_taper : float
        Tapering angle of the spiral body.
    new_Delta_theta : float
        Actual angular resolution used after discretization.
    beta : float
        Radial growth factor per segment.
    """

    def __init__(self, seg_lengths, seg_halfwidths, phi_taper, new_Delta_theta, beta):
        self.seg_lengths = seg_lengths
        self.seg_halfwidths = seg_halfwidths
        self.phi_taper = phi_taper
        self.new_Delta_theta = new_Delta_theta
        self.beta = beta


class SpiralCalculator:
    """
    Compute and discretize a logarithmic spiral for the soft robotic chain.

    This class encapsulates the complete mathematical handling:
    solving the spiral parameters, computing radii, angles, and
    discretizing the curve into mechanical segments.

    Parameters
    ----------
    L_target : float
        Desired total chain length.
    base_d : float
        Diameter at the base.
    tip_d : float
        Diameter at the tip.
    Delta_theta_deg : float
        Angular resolution in degrees for discretization.
    """

    def __init__(self, L_target, base_d, tip_d, Delta_theta_deg):
        self.L_target = L_target
        self.base_d = base_d
        self.tip_d = tip_d
        self.Delta_theta = np.deg2rad(Delta_theta_deg)

        self.params = ms.SpiralParams(
            base_d=self.base_d,
            tip_d=self.tip_d,
            L_target=self.L_target
        )

    def compute_geometry(self) -> SpiralGeometry:
        """
        Compute all geometric quantities of the logarithmic spiral.

        This includes solving the nonlinear equation for the spiral growth
        parameter ``b``, computing the spiral radius function, taper angle,
        centerline arc length, and discretizing the curve into segments.

        Returns
        -------
        SpiralGeometry
            A container holding all geometric values such as segment lengths,
            radii, taper angles, and discretization parameters.
        """

        f = ms.make_f_of_b(self.params)
        b_sol = bisect(f, 1e-4, 1.0, xtol=1e-12, rtol=1e-12, maxiter=200)

        theta0 = ms.theta0_from_ratio(
            b_sol,
            base_d=self.base_d,
            tip_d=self.tip_d
        )
        a = ms.a_from_tip(b_sol, tip_d=self.tip_d)
        L_check = ms.length_central(a, b_sol, theta0)
        phi_taper = ms.taper_angle_phi(b_sol)

        # Diskretisierung
        N_cont = (1.0 / (b_sol * self.Delta_theta)) * np.log(self.base_d / self.tip_d)
        N = max(1, int(np.round(N_cont)))
        new_Delta_theta = theta0 / N
        thetas = np.linspace(0.0, theta0, N + 1)

        delta_vals = ms.delta_width(thetas, a=a, b=b_sol)
        factor = np.sqrt(b_sol**2 + 1.0) / b_sol

        s_vals = factor * (ms.rho_c(thetas, a=a, b=b_sol) - ms.rho_c(0.0, a=a, b=b_sol))
        Y_vals = L_check - s_vals
        w_vals = 0.5 * delta_vals

        seg_lengths = (Y_vals[:-1] - Y_vals[1:])
        seg_halfwidths = w_vals[:-1]

        beta = np.exp(b_sol * new_Delta_theta)

        return SpiralGeometry(
            seg_lengths=seg_lengths,
            seg_halfwidths=seg_halfwidths,
            phi_taper=phi_taper,
            new_Delta_theta=new_Delta_theta,
            beta=beta
        )


# ===================================================================
# 3) XML-Erzeuger (vormals body_block, tendons_xml, ...)
# ===================================================================

# can also be build with Mujoco Spec API. Here, we keep it simple with string building.

class XMLBuilder:
    """
    Build a complete MJCF string for a discretized spiral soft robot.

    Parameters
    ----------
    seg_lengths : array_like
        Length of each segment.
    seg_halfwidths : array_like
        Half-width (radius) for each segment.
    phi_taper : float
        Taper angle of the spiral.
    Delta_theta : float
        Angular step between segments.
    beta : float
        Radial growth factor per segment.
    """
    NUM_CABLES = 2
    SITE_SIZE = 0.001

    def __init__(self, seg_lengths, seg_halfwidths, phi_taper, Delta_theta, beta):
        self.seg_lengths = seg_lengths
        self.seg_halfwidths = seg_halfwidths
        self.phi_taper = phi_taper
        self.Delta_theta = Delta_theta
        self.beta = beta
    # ---------- Hilfsblöcke ----------

    def mjcf_header(self, model_name):
        """
        Create the MJCF header including worldbody and base.

        Parameters
        ----------
        model_name : str
            Name of the root MJCF model.

        Returns
        -------
        str
            XML snippet.
        """
        rng = np.rad2deg(self.Delta_theta)
        return f'''<mujoco model="{model_name}">
  <compiler/>
  <option timestep="0.005" gravity="0 0 -9.81" impratio="10" iterations="50"/>
  <default>
    <joint damping="0.2" stiffness="0.01" limited="true" range="{-rng+5} {rng-5}" solimplimit="0.9 0.95 0.001" solreflimit="0.02 0.5" armature="0.01"/>
    <geom contype="1" conaffinity="0"/>
  </default>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="plane" size="2 2 0.1" rgba="0 0 1 0.6" contype="2" conaffinity="1"/>
      <!-- Der eigentliche Ketten-Root wird hier als Kind erzeugt -->
'''

    def mjcf_footer(self):
        """Return the closing MJCF tag."""
        return "</mujoco>\n"

    def worldbody_footer(self):
        """Return closing tags for worldbody."""
        return "    </body>\n  </worldbody>\n"

    # ---------- Körpersegment ----------


    def body_block(self, i, seg_len, half_width, add_color=False, gap=0.002):
        """
        Generate the MJCF XML block for a single chain segment.

        Parameters
        ----------
        i : int
            Segment index.
        seg_len : float
            Length of the segment.
        half_width : float
            Half-width (radius) of the segment.
        add_color : bool, optional
            Whether to use an alternating color scheme.
        gap : float, optional
            Visual spacing offset for tendon routing.

        Returns
        -------
        str
            XML snippet representing the segment.
        """
        half_vis_len = seg_len / 2.0
        hx, hy, hz = float(half_width), float(half_width), float(half_vis_len)
        
        a = -np.tan(np.pi/2 - (self.phi_taper/2))
        b = -1
        c = np.tan(np.pi/2 - (self.phi_taper/2)) * ((half_width * self.beta) - 0.003)

        solv0 = ms.solve_for_points(a=a, b=b, c=c, theta_deg=np.rad2deg(self.Delta_theta))
        #xC0, yC0 = solv0["C"]
        xD0, yD0 = solv0["D"]
        x_in, y_in, z_in = xD0, 0, yD0

        c = np.tan(np.pi/2 - (self.phi_taper/2)) * ((half_width) - 0.003)
        solv1 = ms.solve_for_points(a=a, b=b, c=c, theta_deg=np.rad2deg(self.Delta_theta))
        xC1, yC1 = solv1["C"]
        #xD1, yD1 = solv1["D"]
        x_out, y_out, z_out = xC1, 0, yC1 + seg_len

        rgba = '0.6 0.75 0.95 0.3' if add_color else '0.2 0.7 0.2 0.3'

        return f'''      <body name="seg_{i}" pos="0 0 0">
            <joint name="j_{i}" type="hinge" axis="0 1 0" pos="0 0 0" stiffness="0.05" damping="0.05"
                   limited="true" range="{-np.rad2deg(self.Delta_theta)+0.1} {np.rad2deg(self.Delta_theta)-0.1}"
                   solimplimit="0.9 0.95 0.001" solreflimit="0.01 0.5"/>
            <geom name="g_{i}" type="box" size="{hx:.6g} {hy:.6g} {hz:.6g}" pos="0 0 {hz:.6g}" 
                  rgba="{rgba}" contype="1" conaffinity="0" density="1100"/>
            <site name="site_in_{i}_0"  pos="{x_in:.6g} {y_in:.6g} {z_in:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
            <site name="site_out_{i}_0" pos="{x_out:.6g} {y_out:.6g} {z_out:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
            <site name="site_in_{i}_1"  pos="{-x_in:.6g} {y_in:.6g} {z_in:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
            <site name="site_out_{i}_1" pos="{-x_out:.6g} {y_out:.6g} {z_out:.6g}" size="{self.SITE_SIZE}" rgba="1 1 0 1"/>
            <body name="seg_{i}_end" pos="0 0 {seg_len:.6g}">
'''


    def close_body_block(self):
        """
        Erzeugt den schließenden MJCF-Block für ein Körpersegment.

        Dieser Block beendet sowohl das innere Hilfs-Body-Element
        (``seg_i_end``) als auch das äußere Segmentelement (``seg_i``).

        Returns
        -------
        str
            XML-Fragment, das die offenen ``<body>``-Tags schließt.
        """
        return "        </body>\n      </body>\n"

    # ---------- Tendons / Actuators ----------

    def tendons_xml(self, N):
        """
        Erzeugt die MJCF-Tendon-Definitionen für alle Seilzüge.

        Es werden für jedes der vorhandenen Kabel (`NUM_CABLES`)
        ``<spatial>``-Tendons erzeugt, deren Punkte über die zuvor
        generierten ``site_in``- und ``site_out``-Sites aller Segmente verlaufen.

        Parameters
        ----------
        N : int
            Anzahl der Segmente der Kette.

        Returns
        -------
        str
            XML-Block mit allen Tendon-Definitionen.
        """
        out = ["  <tendon>"]
        for k in range(self.NUM_CABLES):
            out.append(f'    <spatial name="tendon_{k}" width="0.001" rgba="1 0 0 1" frictionloss="0.1" stiffness="50">')
            for i in reversed(range(N)):
                out.append(f'      <site site="site_in_{i}_{k}"/>')
                out.append(f'      <site site="site_out_{i}_{k}"/>')
            out.append("    </spatial>")
        out.append("  </tendon>\n")
        return "\n".join(out)

    def actuators_xml(self):
        """
        Erzeugt die MJCF-Definition der Aktuatoren für die Tendons.

        Für jedes Kabel wird ein Position-Actuator erzeugt, der
        auf die zugehörige Tendon-Länge wirkt.

        Returns
        -------
        str
            XML-Block mit allen Aktuator-Definitionen.
        """
        out = ["  <actuator>"]
        for k in range(self.NUM_CABLES):
            out.append(
                f'    <position name="tendon_act_{k}" tendon="tendon_{k}" '
                f'kp="200" forcerange="-200 0" ctrlrange="0 0.5"/>'
            )
        out.append("  </actuator>\n")
        return "\n".join(out)

    def sensors_xml(self):
        """
        Erzeugt die MJCF-Sensordefinitionen für die Tendons.

        Für jedes Kabel werden Sensoren für Position und Geschwindigkeit
        der jeweiligen Tendon definiert.

        Returns
        -------
        str
            XML-Block mit allen Sensor-Definitionen.
        """
        out = ["  <sensor>"]
        for k in range(self.NUM_CABLES):
            out.append(f'    <tendonpos name="tendon{k}_pos" tendon="tendon_{k}"/>')
            out.append(f'    <tendonvel name="tendon{k}_vel" tendon="tendon_{k}"/>')
        out.append("  </sensor>\n")
        return "\n".join(out)

    # ---------- Gesamt-XML ----------

    def build(self, model_name="spiral_chain"):
        """
        Erzeugt das vollständige MJCF-XML-Modell.

        Der Builder setzt alle Teilkomponenten (Körpersegmente, Tendons,
        Aktuatoren, Sensoren und Header/Footer-Blöcke) zu einem vollständigen
        XML-Modell für Mujoco zusammen.

        Parameters
        ----------
        model_name : str, optional
            Name des MJCF-Modells (Default: ``"spiral_chain"``).

        Returns
        -------
        str
            Vollständiges MJCF-XML als String.
        """

        N = len(self.seg_lengths)
        xml = [self.mjcf_header(model_name)]

        # Körper generieren
        for i in reversed(range(N)):
            xml.append(
                self.body_block(
                    i,
                    self.seg_lengths[i],
                    self.seg_halfwidths[i],
                    add_color=(i % 2 == 0)
                )
            )

        # Körper schließen
        for _ in reversed(range(N)):
            xml.append(self.close_body_block())

        xml.append(self.worldbody_footer())
        xml.append(self.tendons_xml(N))
        xml.append(self.actuators_xml())
        xml.append(self.sensors_xml())
        xml.append(self.mjcf_footer())

        return "".join(xml)
