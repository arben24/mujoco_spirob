"""
SpiRob Torsionsfeder Systemidentifikation – Logarithmisches Dekrement
=====================================================================
Automatisierte Bestimmung von Steifigkeit (k) und Dämpfung (d) aus
Ausschwing-Messungen (Free Vibration / Logarithmic Decrement Method).

Verwendung:
    python torsion_spring_sysid.py

Konfiguration:
    JOINTS-Dict unten anpassen (sensor_id, J pro Gelenk).
"""

import csv
import glob
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# ============================================================================
# KONFIGURATION
# ============================================================================

# Pfad zum builds-Ordner (relativ zum Skript)
BUILDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'builds')

# Ergebnis-Ordner
RESULTS_DIR = os.path.join(BUILDS_DIR, 'results')

# Konfiguration pro Gelenk-Ordner
# - sensor_id: Welcher Sensor für die Auswertung verwendet werden soll
# - J: Massenträgheitsmoment des schwingenden Glieds in kg·m²
#
# ACHTUNG: Bitte die Werte anpassen! Die hier eingetragenen sind Platzhalter.
JOINTS = {
    #'data_joint_1':  {'sensor_id': 0,  'J': 0.060},
    #'data_joint_8':  {'sensor_id': 0,  'J': 0.001},
    'data_joint_11': {'sensor_id': 0, 'J': 0.001},
    # 'data_joint_14': {'sensor_id': 0, 'J': ??},  # Ordner derzeit leer
}

# Signalverarbeitung
LOWPASS_ENABLED = True       # Butterworth-Tiefpassfilter aktivieren
LOWPASS_CUTOFF_HZ = 50.0    # Grenzfrequenz in Hz
LOWPASS_ORDER = 4            # Filterordnung

# Peak-Erkennung
MIN_PEAKS_REQUIRED = 3       # Mindestanzahl Peaks für valide Auswertung
PEAK_PROMINENCE_FACTOR = 0.1 # Prominence = Faktor × max. Amplitude

# Ausreißer-Erkennung
OUTLIER_SIGMA = 2.0          # Messungen > 2σ vom Mittel werden markiert


# ============================================================================
# DATENKLASSEN
# ============================================================================

@dataclass
class MeasurementResult:
    """Ergebnis einer einzelnen Messung."""
    filename: str
    joint_name: str
    sensor_id: int
    J: float
    omega_d: float          # Gedämpfte Eigenkreisfrequenz [rad/s]
    omega_n: float          # Ungedämpfte Eigenkreisfrequenz [rad/s]
    f_d: float              # Gedämpfte Eigenfrequenz [Hz]
    f_n: float              # Ungedämpfte Eigenfrequenz [Hz]
    zeta: float             # Dämpfungsverhältnis [-]
    delta: float            # Logarithmisches Dekrement [-]
    k: float                # Steifigkeit [Nm/rad]
    d: float                # Dämpfung [Nm·s/rad]
    n_peaks: int            # Anzahl verwendeter Peaks
    is_outlier: bool = False


@dataclass
class JointSummary:
    """Zusammenfassung über alle Messungen eines Gelenks."""
    joint_name: str
    n_measurements: int
    n_valid: int
    n_outliers: int
    omega_d_mean: float
    omega_d_std: float
    omega_n_mean: float
    omega_n_std: float
    zeta_mean: float
    zeta_std: float
    k_mean: float
    k_std: float
    d_mean: float
    d_std: float


@dataclass
class AnalysisParams:
    """Alle einstellbaren Parameter der Ausschwing-Analyse.

    Defaults entsprechen den Modul-Konstanten, sodass sich das Verhalten
    ohne explizite Parameter nicht ändert (Rückwärtskompatibilität).
    """
    lowpass_enabled: bool = LOWPASS_ENABLED
    lowpass_cutoff_hz: float = LOWPASS_CUTOFF_HZ
    lowpass_order: int = LOWPASS_ORDER
    peak_prominence_factor: float = PEAK_PROMINENCE_FACTOR
    peak_min_distance_s: float = 0.02
    min_peaks_required: int = MIN_PEAKS_REQUIRED
    outlier_sigma: float = OUTLIER_SIGMA
    trigger_mode: str = 'onset'          # 'onset' = Schwingungsbeginn, 'jump' = größter Sprung
    onset_noise_factor: float = 4.0      # Empfindlichkeit der Onset-Erkennung


# ============================================================================
# SIGNALVERARBEITUNG
# ============================================================================

def apply_lowpass_filter(signal: np.ndarray, fs: float,
                         cutoff: float = LOWPASS_CUTOFF_HZ,
                         order: int = LOWPASS_ORDER) -> np.ndarray:
    """Butterworth-Tiefpassfilter auf das Signal anwenden."""
    nyq = 0.5 * fs
    if cutoff >= nyq:
        # Grenzfrequenz zu hoch für die Abtastrate → kein Filter
        return signal
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)


def estimate_sample_rate(t_s: np.ndarray) -> float:
    """Abtastrate aus den Zeitstempeln schätzen."""
    dt = np.diff(t_s)
    dt_positive = dt[dt > 0]
    if len(dt_positive) == 0:
        return 500.0  # Fallback
    return 1.0 / np.median(dt_positive)


# ============================================================================
# DATEN LADEN
# ============================================================================

def load_csv(filepath: str, sensor_id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    CSV-Datei laden und nach sensor_id filtern.

    Returns:
        t_s: Zeitachse in Sekunden (relativ, ab 0 = Trigger)
        signal: acc_sum_YZ Werte (Fallback: acc_mag_YZ für ältere Dateien)
    """
    t_us_list = []
    signal_list = []

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        signal_col = 'acc_sum_YZ' if 'acc_sum_YZ' in fieldnames else 'acc_mag_YZ'
        for row in reader:
            if int(row['sensor_id']) == sensor_id:
                t_us_list.append(int(row['t_us']))
                signal_list.append(float(row[signal_col]))

    if len(t_us_list) == 0:
        raise ValueError(f"Keine Daten für sensor_id={sensor_id} in {filepath}")

    t_us = np.array(t_us_list, dtype=np.float64)
    signal = np.array(signal_list, dtype=np.float64)

    # Zeitachse normalisieren: Erster Zeitstempel = 0
    t_s = (t_us - t_us[0]) / 1_000_000.0

    return t_s, signal


def find_trigger_index(t_s: np.ndarray, signal: np.ndarray) -> int:
    """
    Trigger-Zeitpunkt finden über den größten Sprung im Signal.

    Das Aufnahme-Skript speichert Pre-Trigger-Daten (ca. 0.5s vor dem Trigger).
    """
    if len(signal) < 10:
        return 0

    # Absoluter Unterschied zum Vorgänger
    diff = np.abs(np.diff(signal))

    # Suche den größten Sprung (= Trigger)
    trigger_idx = np.argmax(diff)

    return trigger_idx


def find_onset_index(t_s: np.ndarray, signal: np.ndarray,
                     noise_factor: float = 4.0) -> int:
    """
    Beginn der Schwingung finden (statt des größten Sprungs).

    Bestimmt eine Ruhe-Baseline aus dem Signalanfang und gibt den ersten
    Index zurück, an dem das Signal deutlich (noise_factor · σ_Rausch bzw.
    mind. 5 % der Maximalamplitude) davon abweicht. So werden auch die
    ersten Schwingungen erfasst, die vor dem größten Sprung liegen.
    """
    n = len(signal)
    if n < 10:
        return 0

    # Baseline aus dem ruhigen Anfang – höchstens bis zum größten Sprung
    jump = find_trigger_index(t_s, signal)
    n_base = max(5, min(int(jump), int(n * 0.2)))
    baseline = signal[:n_base]
    mean = float(np.mean(baseline))
    noise = float(np.std(baseline))

    centered = np.abs(signal - mean)
    thresh = max(noise_factor * noise, 0.05 * float(np.max(centered)))

    above = np.where(centered > thresh)[0]
    if len(above) == 0:
        return jump
    return int(above[0])


# ============================================================================
# KERN-ALGORITHMUS: LOGARITHMISCHES DEKREMENT
# ============================================================================

def analyze_free_vibration(t_s: np.ndarray, signal: np.ndarray,
                           fs: float,
                           params: Optional['AnalysisParams'] = None,
                           trigger_idx: Optional[int] = None) -> Optional[dict]:
    """
    Analysiert einen freien Ausschwingvorgang.

    Args:
        params: Analyse-Parameter. None → Standardwerte aus den Modul-Konstanten.
        trigger_idx: Fester Startindex (manuelle Vorgabe). None → automatische
            Erkennung gemäß ``params.trigger_mode`` ('onset' oder 'jump').

    Returns:
        dict mit omega_d, zeta, delta, n_peaks, peak_times, peak_amplitudes
        oder None falls nicht genug Peaks gefunden
    """
    if params is None:
        params = AnalysisParams()

    # --- 1. Trigger/Onset finden und Zeitachse zentrieren ---
    if trigger_idx is None:
        if params.trigger_mode == 'onset':
            trigger_idx = find_onset_index(t_s, signal, params.onset_noise_factor)
        else:
            trigger_idx = find_trigger_index(t_s, signal)
    else:
        trigger_idx = int(np.clip(trigger_idx, 0, len(signal) - 1))

    # Pre-Trigger-Phase für Offset-Bestimmung
    pre_trigger = signal[:max(trigger_idx, 1)]
    offset = np.mean(pre_trigger) if len(pre_trigger) > 5 else 0.0

    # Signal zentrieren
    signal_centered = signal - offset

    # Nur Post-Trigger verwenden (ab Trigger-Index)
    t_post = t_s[trigger_idx:] - t_s[trigger_idx]
    sig_post = signal_centered[trigger_idx:]

    if len(sig_post) < 20:
        return None

    # --- 2. Optionaler Tiefpassfilter ---
    if params.lowpass_enabled:
        sig_filtered = apply_lowpass_filter(sig_post, fs,
                                            params.lowpass_cutoff_hz,
                                            params.lowpass_order)
    else:
        sig_filtered = sig_post.copy()

    # --- 3. Peaks finden (Maxima) ---
    max_amp = np.max(np.abs(sig_filtered))
    if max_amp < 1e-6:
        return None

    prominence = params.peak_prominence_factor * max_amp
    # Minimaler Abstand zwischen Peaks (in Samples)
    min_distance = max(5, int(fs * params.peak_min_distance_s))

    # Positive Peaks (Maxima)
    pos_peak_idx, pos_props = find_peaks(
        sig_filtered,
        prominence=prominence,
        distance=min_distance
    )

    # Negative Peaks (Minima) → invertiertes Signal
    neg_peak_idx, neg_props = find_peaks(
        -sig_filtered,
        prominence=prominence,
        distance=min_distance
    )

    # Alle Peaks nach Zeit sortiert zusammenführen
    all_peak_idx = np.sort(np.concatenate([pos_peak_idx, neg_peak_idx]))
    all_peak_amplitudes = np.abs(sig_filtered[all_peak_idx])
    all_peak_times = t_post[all_peak_idx]

    if len(all_peak_idx) < params.min_peaks_required:
        return None

    # --- 4. Gedämpfte Eigenfrequenz ω_d ---
    # Aus aufeinanderfolgenden Peaks gleicher Polarität → volle Periode
    # Oder aus allen Peaks → halbe Periode
    # Wir verwenden alle Peaks (halbe Perioden) für bessere Statistik
    half_periods = np.diff(all_peak_times)
    half_periods = half_periods[half_periods > 0]

    if len(half_periods) == 0:
        return None

    # Volle Periode = 2 × Median der Halbperioden
    T_d = 2.0 * np.median(half_periods)
    omega_d = 2.0 * np.pi / T_d

    # --- 5. Logarithmisches Dekrement δ ---
    # Verwende aufeinanderfolgende Peaks (gleiche Polarität = volle Periode)
    # Dafür: jeden 2. Peak nehmen (alle geraden ODER alle ungeraden)
    # Besser: Nur positive Peaks ODER nur negative Peaks
    pos_amplitudes = sig_filtered[pos_peak_idx]
    neg_amplitudes = np.abs(sig_filtered[neg_peak_idx])

    # Wähle die Serie mit mehr Peaks
    if len(pos_amplitudes) >= len(neg_amplitudes):
        peak_amps_series = np.abs(pos_amplitudes)
        peak_times_series = t_post[pos_peak_idx]
    else:
        peak_amps_series = neg_amplitudes
        peak_times_series = t_post[neg_peak_idx]

    if len(peak_amps_series) < 2:
        return None

    # Logarithmisches Dekrement: δ = ln(A_i / A_{i+1}) über aufeinanderfolgende Peaks
    deltas = []
    for i in range(len(peak_amps_series) - 1):
        A_i = peak_amps_series[i]
        A_next = peak_amps_series[i + 1]
        if A_next > 1e-9 and A_i > A_next:  # Nur abklingende Paare
            deltas.append(math.log(A_i / A_next))

    if len(deltas) == 0:
        # Fallback: über alle Peaks (mit Faktor 2 weil Halbperioden)
        for i in range(len(all_peak_amplitudes) - 2):
            A_i = all_peak_amplitudes[i]
            A_next = all_peak_amplitudes[i + 2]  # Übernächster = gleiche Polarität
            if A_next > 1e-9 and A_i > A_next:
                deltas.append(math.log(A_i / A_next))

    if len(deltas) == 0:
        # Letzte Chance: Gesamtabklingrate über alle Peaks
        if len(all_peak_amplitudes) >= 2:
            A_first = all_peak_amplitudes[0]
            A_last = all_peak_amplitudes[-1]
            n_half_cycles = len(all_peak_amplitudes) - 1
            if A_last > 1e-9 and A_first > A_last:
                # Über n halbe Zyklen: δ_gesamt = ln(A_first/A_last)
                # Pro vollen Zyklus: δ = δ_gesamt / (n_half_cycles / 2)
                delta_total = math.log(A_first / A_last)
                delta = delta_total / (n_half_cycles / 2.0)
                deltas = [delta]

    if len(deltas) == 0:
        return None

    delta = np.mean(deltas)

    # --- 6. Dämpfungsverhältnis ζ ---
    zeta = delta / math.sqrt(4.0 * math.pi**2 + delta**2)

    return {
        'omega_d': omega_d,
        'zeta': zeta,
        'delta': delta,
        'n_peaks': len(all_peak_idx),
        # Für Plots:
        't_post': t_post,
        'sig_post': sig_post,
        'sig_filtered': sig_filtered,
        'all_peak_idx': all_peak_idx,
        'all_peak_times': all_peak_times,
        'all_peak_amplitudes': all_peak_amplitudes,
        'pos_peak_idx': pos_peak_idx,
        'neg_peak_idx': neg_peak_idx,
        'T_d': T_d,
        'offset': offset,
        'trigger_idx': trigger_idx,
    }


# ============================================================================
# ERGEBNIS-BERECHNUNG
# ============================================================================

def compute_parameters(omega_d: float, zeta: float, J: float) -> dict:
    """Berechne ω_n, k, d aus ω_d, ζ und J."""
    # ω_n = ω_d / sqrt(1 - ζ²)
    if zeta >= 1.0:
        # Überdämpft – sollte bei freier Schwingung nicht vorkommen
        omega_n = omega_d
    else:
        omega_n = omega_d / math.sqrt(1.0 - zeta**2)

    k = J * omega_n**2
    d = 2.0 * J * zeta * omega_n

    f_d = omega_d / (2.0 * math.pi)
    f_n = omega_n / (2.0 * math.pi)

    return {
        'omega_n': omega_n,
        'f_d': f_d,
        'f_n': f_n,
        'k': k,
        'd': d,
    }


# ============================================================================
# DIAGNOSE-PLOTS
# ============================================================================

def create_diagnostic_plot(result: MeasurementResult, analysis: dict,
                           output_path: str):
    """Erstellt einen Diagnoseplot für eine einzelne Messung."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    t = analysis['t_post']
    sig_raw = analysis['sig_post']
    sig_filt = analysis['sig_filtered']
    peak_idx = analysis['all_peak_idx']

    # --- Subplot 1: Signal mit Peaks und Einhüllender ---
    ax1 = axes[0]
    ax1.plot(t, sig_raw, color='#cccccc', linewidth=0.8, label='Rohsignal', alpha=0.6)
    ax1.plot(t, sig_filt, color='#2196F3', linewidth=1.2, label='Gefiltert')

    # Peaks markieren
    ax1.plot(t[analysis['pos_peak_idx']], sig_filt[analysis['pos_peak_idx']],
             'v', color='#E91E63', markersize=8, label='Maxima')
    ax1.plot(t[analysis['neg_peak_idx']], sig_filt[analysis['neg_peak_idx']],
             '^', color='#FF9800', markersize=8, label='Minima')

    # Einhüllende (Envelope) aus den Peaks
    if len(peak_idx) >= 2:
        peak_t = t[peak_idx]
        peak_a = np.abs(sig_filt[peak_idx])

        # Exponentieller Fit für die Einhüllende: A(t) = A0 * exp(-ζ*ωn*t)
        try:
            if result.zeta > 0 and result.omega_n > 0:
                decay_rate = result.zeta * result.omega_n
                t_envelope = np.linspace(0, t[-1], 200)
                A0 = peak_a[0]
                envelope = A0 * np.exp(-decay_rate * t_envelope)
                ax1.plot(t_envelope, envelope, '--', color='#4CAF50',
                         linewidth=1.5, label=f'Einhüllende (ζω_n={decay_rate:.1f})')
                ax1.plot(t_envelope, -envelope, '--', color='#4CAF50',
                         linewidth=1.5)
        except Exception:
            pass

    ax1.axhline(y=0, color='grey', linewidth=0.5, linestyle='-')
    ax1.axvline(x=0, color='red', linewidth=1, linestyle='--', alpha=0.5, label='Trigger')

    ax1.set_title(
        f"{result.joint_name} – {os.path.basename(result.filename)}\n"
        f"ω_d={result.omega_d:.2f} rad/s, ζ={result.zeta:.4f}, "
        f"k={result.k:.6f} Nm/rad, d={result.d:.6f} Nm·s/rad",
        fontsize=11
    )
    ax1.set_ylabel('acc_sum_YZ (g)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # --- Subplot 2: Peak-Amplituden (Abklingkurve) ---
    ax2 = axes[1]
    if len(peak_idx) >= 2:
        peak_numbers = np.arange(len(peak_idx))
        peak_amps = np.abs(sig_filt[peak_idx])
        ax2.bar(peak_numbers, peak_amps, color='#2196F3', alpha=0.7)
        ax2.set_xlabel('Peak-Nummer')
        ax2.set_ylabel('|Amplitude| (g)')
        ax2.set_title(f'Abklingende Peak-Amplituden (δ={result.delta:.4f})', fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# ZUSAMMENFASSUNG
# ============================================================================

def compute_joint_summary(joint_name: str,
                          results: list[MeasurementResult]) -> Optional[JointSummary]:
    """Berechne Zusammenfassung über alle Messungen eines Gelenks."""
    if not results:
        return None

    valid = [r for r in results if not r.is_outlier]

    if not valid:
        return JointSummary(
            joint_name=joint_name,
            n_measurements=len(results),
            n_valid=0, n_outliers=len(results),
            omega_d_mean=0, omega_d_std=0,
            omega_n_mean=0, omega_n_std=0,
            zeta_mean=0, zeta_std=0,
            k_mean=0, k_std=0,
            d_mean=0, d_std=0,
        )

    omega_d = np.array([r.omega_d for r in valid])
    omega_n = np.array([r.omega_n for r in valid])
    zeta = np.array([r.zeta for r in valid])
    k = np.array([r.k for r in valid])
    d = np.array([r.d for r in valid])

    return JointSummary(
        joint_name=joint_name,
        n_measurements=len(results),
        n_valid=len(valid),
        n_outliers=len(results) - len(valid),
        omega_d_mean=float(np.mean(omega_d)),
        omega_d_std=float(np.std(omega_d)),
        omega_n_mean=float(np.mean(omega_n)),
        omega_n_std=float(np.std(omega_n)),
        zeta_mean=float(np.mean(zeta)),
        zeta_std=float(np.std(zeta)),
        k_mean=float(np.mean(k)),
        k_std=float(np.std(k)),
        d_mean=float(np.mean(d)),
        d_std=float(np.std(d)),
    )


def mark_outliers(results: list[MeasurementResult],
                  sigma: float = OUTLIER_SIGMA) -> list[MeasurementResult]:
    """Markiere Messungen als Ausreißer wenn ω_d > sigma·σ vom Mittel abweicht."""
    if len(results) < 3:
        return results

    omega_d_values = np.array([r.omega_d for r in results])
    mean = np.mean(omega_d_values)
    std = np.std(omega_d_values)

    if std < 1e-9:
        return results

    for r in results:
        if abs(r.omega_d - mean) > sigma * std:
            r.is_outlier = True

    return results


# ============================================================================
# CSV EXPORT
# ============================================================================

def export_results_csv(all_results: list[MeasurementResult],
                       summaries: list[JointSummary],
                       output_dir: str):
    """Exportiere alle Ergebnisse als CSV-Dateien."""
    # --- Einzelergebnisse ---
    detail_path = os.path.join(output_dir, 'sysid_ergebnisse_detail.csv')
    headers = [
        'joint', 'datei', 'sensor_id', 'J_kgm2',
        'omega_d_rad_s', 'omega_n_rad_s', 'f_d_Hz', 'f_n_Hz',
        'zeta', 'delta', 'k_Nm_rad', 'd_Nms_rad',
        'n_peaks', 'ist_ausreisser'
    ]

    with open(detail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in all_results:
            writer.writerow([
                r.joint_name, os.path.basename(r.filename), r.sensor_id, r.J,
                f'{r.omega_d:.4f}', f'{r.omega_n:.4f}', f'{r.f_d:.4f}', f'{r.f_n:.4f}',
                f'{r.zeta:.6f}', f'{r.delta:.6f}', f'{r.k:.8f}', f'{r.d:.8f}',
                r.n_peaks, r.is_outlier
            ])

    print(f"  → Detail-CSV: {detail_path}")

    # --- Zusammenfassung ---
    summary_path = os.path.join(output_dir, 'sysid_ergebnisse_zusammenfassung.csv')
    headers = [
        'joint', 'n_messungen', 'n_valide', 'n_ausreisser',
        'omega_d_mean', 'omega_d_std',
        'omega_n_mean', 'omega_n_std',
        'zeta_mean', 'zeta_std',
        'k_mean_Nm_rad', 'k_std',
        'd_mean_Nms_rad', 'd_std'
    ]

    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for s in summaries:
            writer.writerow([
                s.joint_name, s.n_measurements, s.n_valid, s.n_outliers,
                f'{s.omega_d_mean:.4f}', f'{s.omega_d_std:.4f}',
                f'{s.omega_n_mean:.4f}', f'{s.omega_n_std:.4f}',
                f'{s.zeta_mean:.6f}', f'{s.zeta_std:.6f}',
                f'{s.k_mean:.8f}', f'{s.k_std:.8f}',
                f'{s.d_mean:.8f}', f'{s.d_std:.8f}',
            ])

    print(f"  → Zusammenfassungs-CSV: {summary_path}")


# ============================================================================
# KONSOLEN-AUSGABE
# ============================================================================

def print_results_table(results: list[MeasurementResult], joint_name: str):
    """Drucke eine formatierte Ergebnistabelle."""
    print(f"\n{'─' * 120}")
    print(f"  Gelenk: {joint_name}")
    print(f"{'─' * 120}")
    print(f"  {'Datei':<40} {'ω_d [rad/s]':>12} {'ω_n [rad/s]':>12} "
          f"{'ζ':>10} {'k [Nm/rad]':>14} {'d [Nm·s/rad]':>14} {'Peaks':>6} {'Status':>10}")
    print(f"  {'─' * 118}")

    for r in results:
        status = "⚠ AUSREISSER" if r.is_outlier else "✓ OK"
        basename = os.path.basename(r.filename)
        print(f"  {basename:<40} {r.omega_d:>12.4f} {r.omega_n:>12.4f} "
              f"{r.zeta:>10.6f} {r.k:>14.8f} {r.d:>14.8f} {r.n_peaks:>6} {status:>10}")


def print_summary(summary: JointSummary):
    """Drucke die Zusammenfassung eines Gelenks."""
    print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  ZUSAMMENFASSUNG: {summary.joint_name:<43}║")
    print(f"  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  Messungen gesamt:  {summary.n_measurements:<5}  "
          f"Valide: {summary.n_valid:<5}  Ausreißer: {summary.n_outliers:<5}  ║")
    print(f"  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  ω_d  = {summary.omega_d_mean:>10.4f} ± {summary.omega_d_std:<10.4f} rad/s           ║")
    print(f"  ║  ω_n  = {summary.omega_n_mean:>10.4f} ± {summary.omega_n_std:<10.4f} rad/s           ║")
    print(f"  ║  ζ    = {summary.zeta_mean:>10.6f} ± {summary.zeta_std:<10.6f}                 ║")
    print(f"  ║  k    = {summary.k_mean:>10.8f} ± {summary.k_std:<10.8f} Nm/rad       ║")
    print(f"  ║  d    = {summary.d_mean:>10.8f} ± {summary.d_std:<10.8f} Nm·s/rad     ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")


# ============================================================================
# ZUSAMMENFASSUNGS-PLOT
# ============================================================================

def create_summary_plot(summaries: list[JointSummary], output_dir: str):
    """Erstellt einen Vergleichsplot über alle Gelenke."""
    if not summaries or all(s.n_valid == 0 for s in summaries):
        return

    valid_summaries = [s for s in summaries if s.n_valid > 0]
    if not valid_summaries:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = [s.joint_name for s in valid_summaries]
    x = np.arange(len(names))

    # Steifigkeit k
    ax = axes[0]
    k_means = [s.k_mean for s in valid_summaries]
    k_stds = [s.k_std for s in valid_summaries]
    ax.bar(x, k_means, yerr=k_stds, capsize=5, color='#2196F3', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('k [Nm/rad]')
    ax.set_title('Steifigkeit k')
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')

    # Dämpfung d
    ax = axes[1]
    d_means = [s.d_mean for s in valid_summaries]
    d_stds = [s.d_std for s in valid_summaries]
    ax.bar(x, d_means, yerr=d_stds, capsize=5, color='#E91E63', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('d [Nm·s/rad]')
    ax.set_title('Dämpfung d')
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')

    # Dämpfungsverhältnis ζ
    ax = axes[2]
    z_means = [s.zeta_mean for s in valid_summaries]
    z_stds = [s.zeta_std for s in valid_summaries]
    ax.bar(x, z_means, yerr=z_stds, capsize=5, color='#4CAF50', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('ζ [-]')
    ax.set_title('Dämpfungsverhältnis ζ')
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')

    fig.suptitle('SpiRob Torsionsfeder – Systemidentifikation (Zusammenfassung)', fontsize=13)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'sysid_zusammenfassung.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  → Zusammenfassungs-Plot: {plot_path}")


# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

def process_joint(joint_name: str, config: dict,
                  results_dir: str,
                  analysis_params: Optional[AnalysisParams] = None) -> list[MeasurementResult]:
    """Verarbeite alle Messungen eines Gelenks."""
    if analysis_params is None:
        analysis_params = AnalysisParams()
    sensor_id = config['sensor_id']
    J = config['J']

    joint_dir = os.path.join(BUILDS_DIR, joint_name)
    if not os.path.isdir(joint_dir):
        print(f"\n⚠ Ordner {joint_dir} existiert nicht – überspringe {joint_name}")
        return []

    csv_files = sorted(glob.glob(os.path.join(joint_dir, 'spirob_messung_*.csv')))
    if not csv_files:
        print(f"\n⚠ Keine CSV-Dateien in {joint_dir} gefunden – überspringe {joint_name}")
        return []

    # Ergebnis-Unterordner für Plots
    joint_results_dir = os.path.join(results_dir, joint_name)
    os.makedirs(joint_results_dir, exist_ok=True)

    print(f"\n{'═' * 120}")
    print(f"  GELENK: {joint_name}  |  Sensor-ID: {sensor_id}  |  J = {J} kg·m²  |  {len(csv_files)} Messungen")
    print(f"{'═' * 120}")

    results = []

    for csv_file in csv_files:
        basename = os.path.basename(csv_file)

        try:
            # Daten laden
            t_s, signal = load_csv(csv_file, sensor_id)

            # Abtastrate schätzen
            fs = estimate_sample_rate(t_s)

            # Ausschwingvorgang analysieren
            analysis = analyze_free_vibration(t_s, signal, fs, analysis_params)

            if analysis is None:
                print(f"  ⚠ {basename}: Nicht genug Peaks gefunden – übersprungen")
                continue

            # Parameter berechnen
            params = compute_parameters(analysis['omega_d'], analysis['zeta'], J)

            # Ergebnis erstellen
            result = MeasurementResult(
                filename=csv_file,
                joint_name=joint_name,
                sensor_id=sensor_id,
                J=J,
                omega_d=analysis['omega_d'],
                omega_n=params['omega_n'],
                f_d=params['f_d'],
                f_n=params['f_n'],
                zeta=analysis['zeta'],
                delta=analysis['delta'],
                k=params['k'],
                d=params['d'],
                n_peaks=analysis['n_peaks'],
            )
            results.append(result)

            # Diagnoseplot erstellen
            plot_name = basename.replace('.csv', '_sysid.png')
            plot_path = os.path.join(joint_results_dir, plot_name)
            create_diagnostic_plot(result, analysis, plot_path)

        except Exception as e:
            print(f"  ✗ {basename}: Fehler – {e}")

    if results:
        # Ausreißer markieren
        results = mark_outliers(results, analysis_params.outlier_sigma)

        # Tabelle ausgeben
        print_results_table(results, joint_name)

    return results


def main():
    """Hauptprogramm."""
    print("\n" + "█" * 120)
    print("  SpiRob Torsionsfeder – Automatische Systemidentifikation")
    print("  Methode: Logarithmisches Dekrement aus Ausschwingvorgängen")
    print("█" * 120)

    # Ergebnis-Ordner erstellen
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results: list[MeasurementResult] = []
    summaries: list[JointSummary] = []

    params = AnalysisParams()

    for joint_name, config in JOINTS.items():
        results = process_joint(joint_name, config, RESULTS_DIR, params)
        all_results.extend(results)

        summary = compute_joint_summary(joint_name, results)
        if summary:
            summaries.append(summary)
            print_summary(summary)

    # CSV-Export
    if all_results:
        print(f"\n{'═' * 120}")
        print("  EXPORT")
        print(f"{'═' * 120}")
        export_results_csv(all_results, summaries, RESULTS_DIR)
        create_summary_plot(summaries, RESULTS_DIR)

    # Abschluss
    print(f"\n{'█' * 120}")
    print(f"  Fertig! {len(all_results)} Messungen ausgewertet.")
    print(f"  Ergebnisse in: {RESULTS_DIR}")
    print(f"{'█' * 120}\n")


if __name__ == '__main__':
    main()
