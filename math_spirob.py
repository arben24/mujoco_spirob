from __future__ import annotations
import math
from typing import Tuple
import numpy as np
from dataclasses import dataclass
from typing import Callable

"""
Werkzeuge für logarithmische Spiralen (SpiRob-Kontext).

Dieses Modul enthält reine, testbare Funktionen ohne globale Variablen,
sowie einen kleinen Parameter-Container und eine Factory für Root-Finding.
"""

__all__ = [
    "rho",
    "rho_c",
    "length_central",
    "delta_width",
    "theta0_from_ratio",
    "a_from_tip",
    "L_from_b",
    "f_of_b",
    "taper_angle_phi",
    "SpiralParams",
    "make_f_of_b",
]


# ----------------------------------------------------------------------
# Reine Funktions-Variante
# ----------------------------------------------------------------------

def rho(theta: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    """
    Berechnet den Radius der logarithmischen Spirale.

    Die Spirale ist definiert durch
    :math:`\\rho(\\theta) = a \\cdot e^{b\\,\\theta}`.

    Parameters
    ----------
    theta : float or np.ndarray
        Polarwinkel :math:`\\theta` in Radiant. Skalar oder Array.
    a : float
        Skalenfaktor :math:`a` (muss i. d. R. > 0 sein).
    b : float
        Wachstumsparameter :math:`b` (kann positiv oder negativ sein; nicht 0 für
        manche Folgeberechnungen, diese Funktion selbst toleriert 0).

    Returns
    -------
    float or np.ndarray
        Radius :math:`\\rho(\\theta)`; Form folgt der von ``theta``.

    Notes
    -----
    - Für :math:`b=0` ist :math:`\\rho(\\theta)=a` (Kreis).
    - Die Funktion ist vektorisiert für ``numpy``-Arrays.

    Examples
    --------
    >>> import numpy as np
    >>> rho(0.0, a=1.0, b=0.2)
    1.0
    >>> th = np.array([0.0, np.pi])
    >>> rho(th, a=1.0, b=0.2).shape == th.shape
    True
    """
    return a * np.exp(b * theta)


def rho_c(theta: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    """
    Hilfsgröße :math:`\\rho_c(\\theta)` zur Längenberechnung.

    Definiert als
    :math:`\\rho_c(\\theta) = \\tfrac{1}{2} a\\,(e^{2\\pi b}+1)\\,e^{b\\theta}`.

    Parameters
    ----------
    theta : float or np.ndarray
        Polarwinkel in Radiant. Skalar oder Array.
    a : float
        Skalenfaktor der Spirale.
    b : float
        Wachstumsparameter.

    Returns
    -------
    float or np.ndarray
        :math:`\\rho_c(\\theta)`; Form folgt der von ``theta``.

    Notes
    -----
    Diese Größe fällt bei geschlossenen Integralen für die Mittellinienlänge an.
    """
    return 0.5 * a * (np.exp(2 * np.pi * b) + 1.0) * np.exp(b * theta)


def length_central(a: float, b: float, theta0: float) -> float:
    """
    Länge der Mittellinie von :math:`\\theta=0` bis :math:`\\theta=\\theta_0`.

    Verwendet die geschlossene Form
    :math:`L = \\frac{\\sqrt{1+b^2}}{b}\\,[\\rho_c(\\theta_0)-\\rho_c(0)]`.

    Parameters
    ----------
    a : float
        Skalenfaktor der Spirale (typisch > 0).
    b : float
        Wachstumsparameter; **darf nicht 0** sein (Division durch 0).
    theta0 : float
        Endwinkel in Radiant (>= 0 empfohlen).

    Returns
    -------
    float
        Mittellinienlänge :math:`L(0\\to\\theta_0)`.

    Raises
    ------
    ValueError
        Wenn ``b`` nahe 0 ist (Division durch 0).

    Notes
    -----
    Für sehr kleine |b| ist die Formel numerisch schlecht konditioniert.

    Examples
    --------
    >>> length_central(a=0.01, b=0.2, theta0=np.pi) > 0
    True
    """
    if np.isclose(b, 0.0):
        raise ValueError("length_central: b darf nicht 0 sein (Division durch 0).")
    return (np.sqrt(b**2 + 1.0) / b) * (rho_c(theta0, a, b) - rho_c(0.0, a, b))


def delta_width(theta: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    """
    Differenz zweier radialer Spiralen nach einer vollen Umdrehung.

    Definiert als
    :math:`\\delta(\\theta) = a\\,(e^{b(\\theta+2\\pi)} - e^{b\\theta})`.

    Parameters
    ----------
    theta : float or np.ndarray
        Polarwinkel in Radiant.
    a : float
        Skalenfaktor.
    b : float
        Wachstumsparameter.

    Returns
    -------
    float or np.ndarray
        :math:`\\delta(\\theta)`; Form folgt der von ``theta``.

    Examples
    --------
    >>> delta_width(0.0, a=1.0, b=0.2) > 0
    True
    """
    return a * (np.exp(b * (theta + 2 * np.pi)) - np.exp(b * theta))


def theta0_from_ratio(b: float, base_d: float, tip_d: float) -> float:
    """
    Berechnet :math:`\\theta_0` aus dem Durchmesser-Verhältnis Basis/Spitze.

    Formel:
    :math:`\\theta_0 = \\frac{\\ln(\\tfrac{\\text{base\\_d}}{\\text{tip\\_d}})}{b}`.

    Parameters
    ----------
    b : float
        Wachstumsparameter; **darf nicht 0** sein.
    base_d : float
        Basisdurchmesser > 0.
    tip_d : float
        Spitzendurchmesser > 0.

    Returns
    -------
    float
        :math:`\\theta_0` (Radiant).

    Raises
    ------
    ValueError
        Wenn ``b`` nahe 0 ist oder Durchmesser nicht > 0 sind.

    Examples
    --------
    >>> theta0_from_ratio(0.2, base_d=0.06, tip_d=0.01) > 0
    True
    """
    if base_d <= 0 or tip_d <= 0:
        raise ValueError("theta0_from_ratio: base_d und tip_d müssen > 0 sein.")
    if np.isclose(b, 0.0):
        raise ValueError("theta0_from_ratio: b darf nicht 0 sein.")
    return np.log(base_d / tip_d) / b


def a_from_tip(b: float, tip_d: float) -> float:
    """
    Bestimmt :math:`a` aus dem Spitzendurchmesser :math:`\\text{tip\\_d}`.

    Formel:
    :math:`a = \\dfrac{\\text{tip\\_d}}{e^{2\\pi b}-1}`.

    Parameters
    ----------
    b : float
        Wachstumsparameter.
    tip_d : float
        Spitzendurchmesser > 0.

    Returns
    -------
    float
        Skalenfaktor :math:`a`.

    Raises
    ------
    ValueError
        Wenn ``tip_d`` nicht > 0 ist oder der Nenner numerisch 0 ist
        (typisch bei :math:`b \\approx 0`).

    Examples
    --------
    >>> a_from_tip(0.2, tip_d=0.01) > 0
    True
    """
    if tip_d <= 0:
        raise ValueError("a_from_tip: tip_d muss > 0 sein.")
    denom = np.exp(2 * np.pi * b) - 1.0
    if np.isclose(denom, 0.0):
        raise ValueError("a_from_tip: exp(2πb) - 1 ist 0 (b≈0).")
    return tip_d / denom


def L_from_b(b: float, base_d: float, tip_d: float) -> float:
    """
    Mittellinienlänge :math:`L(b)` für gegebene Durchmesser (Basis/Spitze).

    Intern:
    1) :math:`\\theta_0 = \\ln(base\\_d/tip\\_d)/b`
    2) :math:`a = \\text{tip\\_d}/(e^{2\\pi b}-1)`
    3) :math:`L = \\text{length\\_central}(a,b,\\theta_0)`

    Parameters
    ----------
    b : float
        Wachstumsparameter; darf nicht 0 sein.
    base_d : float
        Basisdurchmesser > 0.
    tip_d : float
        Spitzendurchmesser > 0.

    Returns
    -------
    float
        Mittellinienlänge :math:`L(b)`.

    Raises
    ------
    ValueError
        Bei ungültigen Parametern (siehe Teilfunktionen).

    Examples
    --------
    >>> L_from_b(0.2, base_d=0.06, tip_d=0.01) > 0
    True
    """
    th0 = theta0_from_ratio(b, base_d, tip_d)
    a_b = a_from_tip(b, tip_d)
    return length_central(a_b, b, th0)


def f_of_b(b: float, base_d: float, tip_d: float, L_target: float) -> float:
    """
    Zielfunktion :math:`f(b) = L(b) - L_{target}` für Root-Finding.

    Parameters
    ----------
    b : float
        Wachstumsparameter.
    base_d : float
        Basisdurchmesser > 0.
    tip_d : float
        Spitzendurchmesser > 0.
    L_target : float
        Ziel-Länge der Mittellinie > 0.

    Returns
    -------
    float
        Differenz :math:`f(b)`; Nullstelle entspricht Erfüllung des Längen-Ziels.

    Raises
    ------
    ValueError
        Wenn ``L_target`` nicht > 0 ist.

    Examples
    --------
    >>> f_of_b(0.2, base_d=0.06, tip_d=0.01, L_target=0.25)  # doctest:+ELLIPSIS
    ...
    """
    if L_target <= 0:
        raise ValueError("f_of_b: L_target muss > 0 sein.")
    return L_from_b(b, base_d, tip_d) - L_target


def taper_angle_phi(b: float) -> float:
    """
    Taper-Winkel :math:`\\varphi` (in Radiant).

    Formel:
    :math:`\\varphi = 2\\,\\arctan\\!\\left(\\dfrac{b\\,(e^{2\\pi b}-1)}{\\sqrt{1+b^2}\\,(e^{2\\pi b}+1)}\\right)`.

    Parameters
    ----------
    b : float
        Wachstumsparameter (reell).

    Returns
    -------
    float
        :math:`\\varphi` in Radiant.

    Notes
    -----
    - Für kleine :math:`|b|` wird der Winkel ebenfalls klein.
    - Der Ausdruck ist numerisch stabil im üblichen Arbeitsbereich.

    Examples
    --------
    >>> 0.0 <= taper_angle_phi(0.2) <= np.pi
    True
    """
    num = b * (np.exp(2 * np.pi * b) - 1.0)
    den = np.sqrt(b**2 + 1.0) * (np.exp(2 * np.pi * b) + 1.0)
    return 2.0 * np.arctan(num / den)


# ----------------------------------------------------------------------
# Komfort: Parameter-Container + Factory
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SpiralParams:
    """
    Parametercontainer für häufig gemeinsam verwendete Größen.

    Attributes
    ----------
    base_d : float
        Basisdurchmesser (> 0).
    tip_d : float
        Spitzendurchmesser (> 0).
    L_target : float
        Ziel-Länge der Mittellinie (> 0).
    """
    base_d: float
    tip_d: float
    L_target: float

    def validate(self) -> None:
        """
        Prüft die Gültigkeit der Parameter.

        Raises
        ------
        ValueError
            Falls ein Wert nicht > 0 ist.
        """
        if self.base_d <= 0 or self.tip_d <= 0 or self.L_target <= 0:
            raise ValueError(
                "SpiralParams: base_d, tip_d und L_target müssen > 0 sein."
            )


def make_f_of_b(params: SpiralParams) -> Callable[[float], float]:
    """
    Erzeugt eine eindimensionale Funktion :math:`f(b) = L(b) - L_{target}`.

    Praktisch für Root-Finding (z. B. ``scipy.optimize.brentq``).

    Parameters
    ----------
    params : SpiralParams
        Gültige Parameter (werden validiert).

    Returns
    -------
    Callable[[float], float]
        Funktion, die nur noch :math:`b` als Eingabe benötigt.

    Raises
    ------
    ValueError
        Wenn die Parameter ungültig sind.

    Examples
    --------
    >>> from math import isfinite
    >>> p = SpiralParams(base_d=0.06, tip_d=0.01, L_target=0.25)
    >>> f = make_f_of_b(p)  # doctest:+ELLIPSIS
    >>> isfinite(float(f(0.2)))
    True
    """
    params.validate()

    def _f(b: float) -> float:
        return f_of_b(b, base_d=params.base_d, tip_d=params.tip_d, L_target=params.L_target)

    return _f



def distance_point_line_from_origin(a: float, b: float, c: float) -> float:
    """
    Abstand des Ursprungs (0,0) zur Geraden :math:`a x + b y + c = 0`.

    Parameters
    ----------
    a, b, c : float
        Koeffizienten der Geraden in Normalform.

    Returns
    -------
    float
        Abstand :math:`d = |c| / \\sqrt{a^2 + b^2}`.

    Raises
    ------
    ValueError
        Wenn ``a = b = 0`` (keine gültige Gerade).

    Examples
    --------
    >>> distance_point_line_from_origin(0.0, 1.0, -2.0)  # y - 2 = 0
    2.0
    """
    norm = math.hypot(a, b)
    if norm == 0.0:
        raise ValueError("Ungültige Gerade: a=b=0.")
    return abs(c) / norm


def closest_point_on_line_to_origin(a: float, b: float, c: float) -> Tuple[float, float]:
    """
    Lotfußpunkt des Ursprungs auf die Gerade :math:`a x + b y + c = 0`.

    Formel
    ------
    :math:`M = -\\dfrac{c}{a^2 + b^2}\\,(a, b)`.

    Parameters
    ----------
    a, b, c : float
        Koeffizienten der Geraden in Normalform.

    Returns
    -------
    tuple[float, float]
        Koordinaten :math:`M=(x_M, y_M)`.

    Raises
    ------
    ValueError
        Wenn ``a = b = 0`` (keine gültige Gerade).

    Examples
    --------
    >>> closest_point_on_line_to_origin(0.0, 1.0, -2.0)  # y - 2 = 0
    (0.0, 2.0)
    """
    denom = a * a + b * b
    if denom == 0.0:
        raise ValueError("Ungültige Gerade: a=b=0.")
    return (-c * a / denom, -c * b / denom)


def line_unit_direction(a: float, b: float) -> Tuple[float, float]:
    """
    Einheits-Richtungsvektor einer Geraden :math:`a x + b y + c = 0`.

    Hinweise
    --------
    Ein Normalenvektor ist :math:`n=(a,b)`. Ein (rechtsdrehender) Richtungsvektor ist
    :math:`u_\\perp = (-b, a)`. Der ausgegebene Vektor wird normiert.

    Parameters
    ----------
    a, b : float
        Komponenten des Normalenvektors.

    Returns
    -------
    tuple[float, float]
        Einheits-Richtungsvektor :math:`\\hat{u}` entlang der Geraden.

    Raises
    ------
    ValueError
        Wenn ``a = b = 0`` (Richtung undefiniert).

    Examples
    --------
    >>> line_unit_direction(0.0, 1.0)  # Gerade parallel zur x-Achse
    (-1.0, 0.0)
    """
    norm = math.hypot(a, b)
    if norm == 0.0:
        raise ValueError("Ungültige Gerade: a=b=0.")
    return (-b / norm, a / norm)


def circle_radius_for_central_angle(d: float, theta_deg: float) -> float:
    """
    Kreisradius aus Abstand :math:`d` und Zentralwinkel :math:`\\theta` (in Grad).

    Formel
    ------
    :math:`r = \\dfrac{d}{\\cos(\\theta/2)}`  mit :math:`\\theta` in Grad.

    Parameters
    ----------
    d : float
        Abstand des Geraden-Lotfußes vom Ursprung.
    theta_deg : float
        Zentralwinkel in Grad.

    Returns
    -------
    float
        Kreisradius :math:`r`.

    Raises
    ------
    ValueError
        Falls :math:`\\cos(\\theta/2) \\le 0`, d. h. :math:`\\theta/2 \\ge 90^\\circ`.

    Examples
    --------
    >>> circle_radius_for_central_angle(1.0, 60.0)
    1.154700538...
    """
    theta_rad = math.radians(theta_deg)
    c = math.cos(theta_rad / 2.0)
    if c <= 0.0:
        raise ValueError("theta/2 muss < 90° sein (cos > 0).")
    return d / c


def intersection_points_circle_line(
    a: float, b: float, c: float, r: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Schnittpunkte zwischen Kreis :math:`x^2 + y^2 = r^2` und Gerade :math:`a x + b y + c = 0`.

    Geometrische Konstruktion
    -------------------------
    - :math:`M` ist der Lotfuß vom Ursprung auf die Gerade.
    - :math:`\\hat{u}` ist der Einheits-Richtungsvektor der Geraden.
    - :math:`d` ist der Abstand Ursprung→Gerade.
    - :math:`s = \\sqrt{r^2 - d^2}` (nichtnegativ, bei Tangente :math:`s=0`).
    - Schnittpunkte: :math:`C = M + s\\,\\hat{u}`, :math:`D = M - s\\,\\hat{u}`.

    Parameters
    ----------
    a, b, c : float
        Koeffizienten der Geraden in Normalform.
    r : float
        Kreisradius (>= 0).

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        Punkte :math:`C` und :math:`D` als Koordinatenpaare.

    Raises
    ------
    ValueError
        Wenn die Gerade vollständig außerhalb des Kreises liegt (kein Schnitt).

    Examples
    --------
    >>> intersection_points_circle_line(0.0, 1.0, 0.0, 1.0)  # x^2+y^2=1 und y=0
    ((1.0, 0.0), (-1.0, 0.0))
    """
    d = distance_point_line_from_origin(a, b, c)
    if d > r + 1e-12:
        raise ValueError("Keine Schnittpunkte: Gerade liegt außerhalb des Kreises.")
    Mx, My = closest_point_on_line_to_origin(a, b, c)
    ux, uy = line_unit_direction(a, b)
    s_sq = max(r * r - d * d, 0.0)  # numerisch nichtnegativ
    s = math.sqrt(s_sq)
    C = (Mx + s * ux, My + s * uy)
    D = (Mx - s * ux, My - s * uy)
    return C, D


def angle_between_rays(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    """
    Winkel zwischen den Strahlen :math:`OP` und :math:`OQ` (in Grad).

    Parameters
    ----------
    p, q : tuple[float, float]
        Endpunkte der Vektoren :math:`\\vec{OP}` und :math:`\\vec{OQ}`.

    Returns
    -------
    float
        Winkel :math:`\\angle(\\vec{OP},\\vec{OQ})` in Grad, im Bereich :math:`[0,180]`.

    Raises
    ------
    ValueError
        Wenn einer der Punkte im Ursprung liegt (Winkel undefiniert).

    Examples
    --------
    >>> angle_between_rays((1, 0), (0, 1))
    90.0
    """
    px, py = p
    qx, qy = q
    dp = px * qx + py * qy
    np_ = math.hypot(px, py)
    nq_ = math.hypot(qx, qy)
    if np_ == 0.0 or nq_ == 0.0:
        raise ValueError("Punkt liegt im Ursprung; Winkel undefiniert.")
    cosang = dp / (np_ * nq_)
    # numerisch einklammern
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def solve_for_points(
    a: float, b: float, c: float, theta_deg: float = 30.0
) -> dict[str, float | tuple[float, float]]:
    """
    Hauptfunktion: berechnet Kreis und Schnittpunkte zur gegebenen Gerade.

    Ablauf
    ------
    1. Abstand :math:`d` des Ursprungs zur Geraden.
    2. Radius :math:`r` aus gewünschtem Zentralwinkel ``theta_deg``.
    3. Schnittpunkte :math:`C, D` von Kreis und Gerade.
    4. Validierung: Zwischenwinkel :math:`\\angle COD` (sollte nahe ``theta_deg`` sein).

    Parameters
    ----------
    a, b, c : float
        Koeffizienten der Geraden in Normalform :math:`a x + b y + c = 0`.
    theta_deg : float, default 30.0
        Zentralwinkel in Grad, für den der Kreis konstruiert wird.

    Returns
    -------
    dict[str, float | tuple[float, float]]
        Wörterbuch mit Schlüsseln:
        - ``"d"`` : Abstand Ursprung→Gerade
        - ``"r"`` : Kreisradius
        - ``"C"`` : erster Schnittpunkt (x, y)
        - ``"D"`` : zweiter Schnittpunkt (x, y)
        - ``"winkel_deg"`` : Winkel :math:`\\angle COD` in Grad

    Raises
    ------
    ValueError
        Bei geometrisch unmöglichen Konfigurationen (z. B. kein Schnitt).

    Examples
    --------
    >>> res = solve_for_points(0.0, 1.0, -0.5, theta_deg=60.0)  # y-0.5=0
    >>> sorted(res.keys())
    ['C', 'D', 'd', 'r', 'winkel_deg']
    """
    d = distance_point_line_from_origin(a, b, c)
    r = circle_radius_for_central_angle(d, theta_deg)
    C, D = intersection_points_circle_line(a, b, c, r)
    w = angle_between_rays(C, D)
    return {"d": d, "r": r, "C": C, "D": D, "winkel_deg": w}



if __name__ == "__main__":
    # ===== Beispiel =====
    # Gerade in Normalform: a x + b y + c = 0
    # Beispiel: y = m x + b0  ->  m x - y + b0 = 0  => a=m, b=-1, c=b0
    # Hier z.B.: y = -1.5 x + 4  ->  a=-1.5, b=-1, c=4

    phi = 11
    phi_rad = np.radians(phi)
    d = 0.01

    a_line, b_line, c_line = -np.tan(np.pi/2 - phi_rad), -1.0, np.tan(np.pi/2 - phi_rad) * (d)
    theta = 30.0  # gewünschter Winkel zwischen den Strahlen OC und OD

    res = solve_for_points(a_line, b_line, c_line, theta)

    print("Gegebene Gerade: {:.6f} x + {:.6f} y + {:.6f} = 0".format(a_line, b_line, c_line))
    print("Gewünschter Winkel: {:.3f}°".format(theta))
    print("Abstand d zum Ursprung: {:.6f}".format(res["d"]))
    print("Benötigter Radius r:    {:.6f}".format(res["r"]))
    print("Schnittpunkt C:         ({:.6f}, {:.6f})".format(*res["C"]))
    print("Schnittpunkt D:         ({:.6f}, {:.6f})".format(*res["D"]))
    print("Prüfwinkel ∠COD:        {:.6f}°".format(res["winkel_deg"]))
