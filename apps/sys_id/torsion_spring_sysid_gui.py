"""
SpiRob Torsionsfeder Sys-ID – Interaktive GUI
==============================================
Grafische Oberfläche zum Auswerten von Ausschwing-Messungen (siehe
torsion_spring_sysid.py). Alle Erkennungs- und Filterparameter lassen sich
live einstellen; Signalverlauf, erkannte Peaks, Einhüllende und das aktuelle
Ergebnis (ω_d, ζ, k, d) werden sofort aktualisiert.

Funktionen:
  • Navigation durch alle Messungen eines Ordners (◀ / ▶)
  • Live-Auswertung der aktuellen Messung
  • Mittelung über alle Messungen des Ordners (mit Ausreißer-Markierung)
  • Speichern aller Einstellungen + Ergebnisse in <ordner>/sysid_settings.yaml
  • Beim Start werden vorhandene Einstellungen aus dieser YAML übernommen

Verwendung:
  uv run apps/sys_id/torsion_spring_sysid_gui.py [ordner]
  # ohne Argument: erster Gelenk-Ordner mit Messungen unter builds/
"""

import datetime
import glob
import os
import sys

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox

# torsion_spring_sysid liegt im selben Ordner → importierbar beim Direktstart
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torsion_spring_sysid as tss  # noqa: E402


BUILDS_DIR = tss.BUILDS_DIR
SETTINGS_FILENAME = 'sysid_settings.yaml'


def _f(x, n: int) -> float:
    """Nach nativem float runden (YAML kann keine numpy-Skalare serialisieren)."""
    return round(float(x), n)

# Standardeinstellungen (werden verwendet wenn keine YAML vorhanden ist)
DEFAULT_SETTINGS = {
    'sensor_id': 0,
    'J': 0.001,
    'lowpass_enabled': True,
    'lowpass_cutoff_hz': 50.0,
    'lowpass_order': 4,
    'peak_prominence_factor': 0.1,
    'peak_min_distance_s': 0.02,
    'min_peaks_required': 3,
    'outlier_sigma': 2.0,
    'trigger_mode': 'onset',
}


# ──────────────────────────────────────────────────────────────────────────────
# Ordner / Dateien
# ──────────────────────────────────────────────────────────────────────────────

def list_recording_folders() -> list:
    """Unterordner von builds/ die Messungen enthalten (ohne 'results')."""
    folders = []
    for entry in sorted(os.listdir(BUILDS_DIR)):
        full = os.path.join(BUILDS_DIR, entry)
        if not os.path.isdir(full) or entry == 'results':
            continue
        if glob.glob(os.path.join(full, 'spirob_messung_*.csv')):
            folders.append(full)
    return folders


def list_recordings(folder: str) -> list:
    """
    Alle Messungen eines Ordners. Falls zu einer Aufnahme eine per
    signal_editor erzeugte '*_bearbeitet.csv' existiert, wird diese bevorzugt.
    """
    raw = sorted(glob.glob(os.path.join(folder, 'spirob_messung_*.csv')))
    raw = [f for f in raw if '_bearbeitet' not in f]

    chosen = []
    for f in raw:
        base = os.path.splitext(f)[0]
        edited = f'{base}_bearbeitet.csv'
        chosen.append(edited if os.path.isfile(edited) else f)
    return chosen


def settings_path(folder: str) -> str:
    return os.path.join(folder, SETTINGS_FILENAME)


def load_settings(folder: str) -> tuple:
    """
    Einstellungen aus YAML laden (Defaults für fehlende Schlüssel).

    Returns:
        (settings: dict, start_overrides_by_name: dict[str, float])
    """
    settings = dict(DEFAULT_SETTINGS)
    overrides = {}
    path = settings_path(folder)
    if os.path.isfile(path):
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            loaded = data.get('settings', data)  # akzeptiere beide Layouts
            for key in DEFAULT_SETTINGS:
                if key in loaded and loaded[key] is not None:
                    settings[key] = loaded[key]
            overrides = loaded.get('start_overrides', {}) or {}
            print(f"Einstellungen geladen aus: {path}")
        except Exception as e:
            print(f"⚠ Konnte {path} nicht lesen ({e}) – nutze Standardwerte.")
    else:
        print("Keine sysid_settings.yaml gefunden – nutze Standardwerte.")
    return settings, overrides


# ──────────────────────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────────────────────

class SysIdGUI:
    def __init__(self, folder: str):
        self.folder = os.path.abspath(folder)
        self.folder_name = os.path.basename(self.folder.rstrip('/'))
        self.settings, overrides_by_name = load_settings(self.folder)

        # Alle Messungen einmalig laden und cachen (t_s, signal)
        self.recordings = []
        for path in list_recordings(self.folder):
            try:
                t_s, signal = tss.load_csv(path, int(self.settings['sensor_id']))
                self.recordings.append({'path': path, 't_s': t_s, 'signal': signal})
            except Exception as e:
                print(f"⚠ {os.path.basename(path)} übersprungen: {e}")

        if not self.recordings:
            raise RuntimeError(
                f"Keine auswertbaren Messungen in {self.folder} "
                f"(sensor_id={self.settings['sensor_id']})."
            )

        # Manuelle Startpunkt-Overrides (Dateipfad → Zeit in s); aus YAML übernehmen
        name2path = {os.path.basename(r['path']): r['path'] for r in self.recordings}
        self.start_overrides = {}
        for name, t in overrides_by_name.items():
            if name in name2path and t is not None:
                self.start_overrides[name2path[name]] = float(t)

        self.idx = 0
        self._loading = False          # unterdrückt Recompute während Bulk-Set
        self._delete_armed = False     # Zwei-Klick-Sicherung für Löschen
        self._last_results = []        # list[tss.MeasurementResult]
        self._last_summary = None
        self._n_failed = 0

        self._build_ui()
        self._apply_settings_to_widgets()
        self._recompute()

    # ── Parameter aus Widgets ─────────────────────────────────────────────────

    @property
    def params(self) -> 'tss.AnalysisParams':
        return tss.AnalysisParams(
            lowpass_enabled=self.chk_filter.get_status()[0],
            lowpass_cutoff_hz=float(self.s_cutoff.val),
            lowpass_order=int(round(self.s_order.val)),
            peak_prominence_factor=float(self.s_prom.val),
            peak_min_distance_s=float(self.s_dist.val) / 1000.0,  # ms → s
            min_peaks_required=int(round(self.s_minpk.val)),
            outlier_sigma=float(self.s_sigma.val),
            trigger_mode='onset' if self.chk_onset.get_status()[0] else 'jump',
        )

    @property
    def J(self) -> float:
        try:
            return float(self.tb_J.text)
        except ValueError:
            return float(self.settings['J'])

    @property
    def sensor_id(self) -> int:
        try:
            return int(float(self.tb_sensor.text))
        except ValueError:
            return int(self.settings['sensor_id'])

    def current_settings_dict(self) -> dict:
        p = self.params
        return {
            'sensor_id': int(self.sensor_id),
            'J': _f(self.J, 8),
            'lowpass_enabled': bool(p.lowpass_enabled),
            'lowpass_cutoff_hz': _f(p.lowpass_cutoff_hz, 4),
            'lowpass_order': int(p.lowpass_order),
            'peak_prominence_factor': _f(p.peak_prominence_factor, 4),
            'peak_min_distance_s': _f(p.peak_min_distance_s, 5),
            'min_peaks_required': int(p.min_peaks_required),
            'outlier_sigma': _f(p.outlier_sigma, 3),
            'trigger_mode': p.trigger_mode,
            'start_overrides': {
                os.path.basename(pth): _f(t, 5)
                for pth, t in sorted(self.start_overrides.items())
            },
        }

    # ── Trigger / Startpunkt ──────────────────────────────────────────────────

    def _auto_trigger_idx(self, t_s, signal, params) -> int:
        """Automatisch erkannter Startindex gemäß gewähltem Modus."""
        if params.trigger_mode == 'onset':
            return tss.find_onset_index(t_s, signal, params.onset_noise_factor)
        return tss.find_trigger_index(t_s, signal)

    def _trigger_idx_for(self, rec, params):
        """Startindex einer Messung: manueller Override oder None (= automatisch)."""
        ov = self.start_overrides.get(rec['path'])
        if ov is None:
            return None
        t_s = rec['t_s']
        return int(np.clip(np.searchsorted(t_s, ov), 0, len(t_s) - 1))

    def _effective_trigger_idx(self, rec, params) -> int:
        """Tatsächlich verwendeter Startindex (Override sonst Auto) – für die Anzeige."""
        ov_idx = self._trigger_idx_for(rec, params)
        if ov_idx is not None:
            return ov_idx
        return self._auto_trigger_idx(rec['t_s'], rec['signal'], params)

    # ── Analyse ───────────────────────────────────────────────────────────────

    def _compute_file(self, t_s, signal, params, J, trigger_idx=None):
        """Analyse einer Messung. Returns (fs, analysis|None, result|None)."""
        fs = tss.estimate_sample_rate(t_s)
        analysis = tss.analyze_free_vibration(t_s, signal, fs, params,
                                              trigger_idx=trigger_idx)
        if analysis is None:
            return fs, None, None
        p = tss.compute_parameters(analysis['omega_d'], analysis['zeta'], J)
        result = dict(analysis)
        result.update(p)
        result['fs'] = fs
        return fs, analysis, result

    def _compute_summary(self, params, J):
        """Alle Messungen auswerten → (results, summary, n_failed)."""
        results = []
        n_failed = 0
        for rec in self.recordings:
            tidx = self._trigger_idx_for(rec, params)
            _, _, res = self._compute_file(rec['t_s'], rec['signal'], params, J, tidx)
            if res is None:
                n_failed += 1
                continue
            results.append(tss.MeasurementResult(
                filename=rec['path'], joint_name=self.folder_name,
                sensor_id=self.sensor_id, J=J,
                omega_d=res['omega_d'], omega_n=res['omega_n'],
                f_d=res['f_d'], f_n=res['f_n'],
                zeta=res['zeta'], delta=res['delta'],
                k=res['k'], d=res['d'], n_peaks=res['n_peaks'],
            ))
        results = tss.mark_outliers(results, params.outlier_sigma)
        summary = tss.compute_joint_summary(self.folder_name, results)
        return results, summary, n_failed

    # ── UI aufbauen ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        plt.close('all')
        self.fig = plt.figure(figsize=(15.5, 9.0))
        self.fig.canvas.manager.set_window_title(
            f'Torsionsfeder Sys-ID – {self.folder_name}')

        # ── Plots ────────────────────────────────────────────────────────────
        self.ax_sig = self.fig.add_axes([0.055, 0.62, 0.56, 0.31])
        self.ax_sig.set_ylabel('acc_sum_YZ (zentriert)')
        self.ax_sig.grid(True, ls=':', alpha=0.5)

        self.ax_peaks = self.fig.add_axes([0.055, 0.45, 0.56, 0.12])
        self.ax_peaks.set_xlabel('Zeit (s)')
        self.ax_peaks.set_ylabel('|Peak| ')
        self.ax_peaks.grid(True, ls=':', alpha=0.5, axis='y')

        # ── Info-Panel (Text) ────────────────────────────────────────────────
        self.ax_info = self.fig.add_axes([0.655, 0.40, 0.325, 0.55])
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(
            0.0, 1.0, '', transform=self.ax_info.transAxes,
            va='top', ha='left', family='monospace', fontsize=10.5,
        )

        # ── Startpunkt-Slider (pro Messung) + Auto-Button ────────────────────
        self.s_start = Slider(self.fig.add_axes([0.14, 0.405, 0.38, 0.02]),
                              'Startpunkt (s)', 0.0, 1.0, valinit=0.0,
                              color='#E91E63', valfmt='%.3f')
        self.s_start.on_changed(self._on_start_change)
        self.btn_auto = Button(self.fig.add_axes([0.535, 0.40, 0.075, 0.03]),
                               'Auto', color='#E1BEE7', hovercolor='#CE93D8')
        self.btn_auto.on_clicked(self._on_auto_start)

        # ── Checkboxen: Onset-Erkennung & Tiefpass ───────────────────────────
        ax_onset = self.fig.add_axes([0.045, 0.355, 0.20, 0.045])
        ax_onset.set_facecolor('none')
        self.chk_onset = CheckButtons(ax_onset, ['Onset-Erkennung'],
                                      [self.settings['trigger_mode'] == 'onset'])
        self.chk_onset.on_clicked(self._on_change)

        ax_chk = self.fig.add_axes([0.045, 0.305, 0.20, 0.045])
        ax_chk.set_facecolor('none')
        self.chk_filter = CheckButtons(ax_chk, ['Tiefpass aktiv'], [True])
        self.chk_filter.on_clicked(self._on_change)

        # ── TextBoxes: J und sensor_id ───────────────────────────────────────
        self.tb_J = TextBox(self.fig.add_axes([0.115, 0.255, 0.085, 0.03]),
                            'J (kg·m²)  ', initial=str(self.settings['J']))
        self.tb_J.on_submit(self._on_change)
        self.tb_sensor = TextBox(self.fig.add_axes([0.115, 0.210, 0.085, 0.03]),
                                 'sensor_id  ', initial=str(self.settings['sensor_id']))
        self.tb_sensor.on_submit(self._on_sensor_change)

        # ── Slider (globale Parameter) ───────────────────────────────────────
        sx, sw, sh = 0.32, 0.30, 0.02
        self.s_cutoff = Slider(self.fig.add_axes([sx, 0.335, sw, sh]),
                               'Cutoff (Hz)', 1.0, 480.0, valinit=50.0, valfmt='%.0f')
        self.s_order  = Slider(self.fig.add_axes([sx, 0.298, sw, sh]),
                               'Ordnung', 1, 8, valinit=4, valstep=1, valfmt='%d')
        self.s_prom   = Slider(self.fig.add_axes([sx, 0.261, sw, sh]),
                               'Prominenz', 0.01, 0.6, valinit=0.1, valfmt='%.2f')
        self.s_dist   = Slider(self.fig.add_axes([sx, 0.224, sw, sh]),
                               'Peak-Abstand (ms)', 1, 80, valinit=20, valstep=1, valfmt='%d')
        self.s_minpk  = Slider(self.fig.add_axes([sx, 0.187, sw, sh]),
                               'Min. Peaks', 2, 15, valinit=3, valstep=1, valfmt='%d')
        self.s_sigma  = Slider(self.fig.add_axes([sx, 0.150, sw, sh]),
                               'Ausreißer σ', 0.5, 4.0, valinit=2.0, valfmt='%.1f')
        for s in (self.s_cutoff, self.s_order, self.s_prom,
                  self.s_dist, self.s_minpk, self.s_sigma):
            s.on_changed(self._on_change)

        # ── Buttons ──────────────────────────────────────────────────────────
        self.btn_prev = Button(self.fig.add_axes([0.045, 0.05, 0.095, 0.045]),
                               '◀ Vorherige', color='#607D8B', hovercolor='#90A4AE')
        self.btn_next = Button(self.fig.add_axes([0.150, 0.05, 0.095, 0.045]),
                               'Nächste ▶', color='#607D8B', hovercolor='#90A4AE')
        self.btn_delete = Button(self.fig.add_axes([0.295, 0.05, 0.15, 0.045]),
                                 'Aufnahme löschen', color='#FF8A65', hovercolor='#FF7043')
        self.btn_reset = Button(self.fig.add_axes([0.485, 0.05, 0.12, 0.045]),
                                'Standardwerte', color='#9E9E9E', hovercolor='#BDBDBD')
        self.btn_save = Button(self.fig.add_axes([0.655, 0.05, 0.20, 0.045]),
                               'Speichern (YAML)', color='#2196F3', hovercolor='#64B5F6')
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_delete.on_clicked(self._on_delete)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_save.on_clicked(self._on_save)

    def _apply_settings_to_widgets(self):
        """Übernimmt self.settings in die Widgets (ohne Zwischen-Recompute)."""
        self._loading = True
        s = self.settings
        self.s_cutoff.set_val(float(s['lowpass_cutoff_hz']))
        self.s_order.set_val(int(s['lowpass_order']))
        self.s_prom.set_val(float(s['peak_prominence_factor']))
        self.s_dist.set_val(float(s['peak_min_distance_s']) * 1000.0)
        self.s_minpk.set_val(int(s['min_peaks_required']))
        self.s_sigma.set_val(float(s['outlier_sigma']))
        # CheckButtons auf gewünschten Zustand bringen
        if self.chk_filter.get_status()[0] != bool(s['lowpass_enabled']):
            self.chk_filter.set_active(0)
        if self.chk_onset.get_status()[0] != (s['trigger_mode'] == 'onset'):
            self.chk_onset.set_active(0)
        self.tb_J.set_val(str(s['J']))
        self.tb_sensor.set_val(str(s['sensor_id']))
        self._loading = False

    # ── Recompute + Plot ────────────────────────────────────────────────────

    def _recompute(self):
        if self._loading or not self.recordings:
            return
        params = self.params
        J = self.J

        rec = self.recordings[self.idx]
        # Startpunkt-Slider auf aktuellen Wert bringen (Override oder Auto)
        self._sync_start_slider(rec, params)

        tidx = self._trigger_idx_for(rec, params)
        fs, analysis, result = self._compute_file(rec['t_s'], rec['signal'], params, J, tidx)

        self._draw_signal(rec, analysis, result, params)

        results, summary, n_failed = self._compute_summary(params, J)
        self._last_results = results
        self._last_summary = summary
        self._n_failed = n_failed

        self._update_info(rec, fs, result, summary, n_failed)
        self.fig.canvas.draw_idle()

    def _sync_start_slider(self, rec, params):
        """Startpunkt-Slider an die aktuelle Messung anpassen (ohne Override auszulösen)."""
        self._loading = True
        t_s = rec['t_s']
        t_max = float(t_s[-1]) if len(t_s) > 1 else 1.0
        self.s_start.valmax = t_max
        self.s_start.ax.set_xlim(0, t_max)
        ov = self.start_overrides.get(rec['path'])
        if ov is None:
            auto_idx = self._auto_trigger_idx(t_s, rec['signal'], params)
            val = float(t_s[auto_idx])
        else:
            val = float(ov)
        self.s_start.set_val(min(val, t_max))
        self._loading = False

    def _draw_signal(self, rec, analysis, result, params):
        ax, axp = self.ax_sig, self.ax_peaks
        ax.clear(); axp.clear()
        ax.grid(True, ls=':', alpha=0.5)
        axp.grid(True, ls=':', alpha=0.5, axis='y')
        ax.set_ylabel('acc_sum_YZ (zentriert)')
        axp.set_xlabel('Zeit (s)')
        axp.set_ylabel('|Peak|')

        t_s, signal = rec['t_s'], rec['signal']

        # Startindex konsistent zur Analyse bestimmen (Override oder Auto)
        is_manual = self.start_overrides.get(rec['path']) is not None
        trig = self._effective_trigger_idx(rec, params)

        # Basis-Darstellung auch wenn keine Peaks gefunden wurden
        fs = tss.estimate_sample_rate(t_s)
        pre = signal[:max(trig, 1)]
        offset = float(np.mean(pre)) if len(pre) > 5 else 0.0
        sig_post = (signal - offset)[trig:]
        t_abs = t_s[trig:]
        if params.lowpass_enabled:
            filt = tss.apply_lowpass_filter(sig_post, fs,
                                            params.lowpass_cutoff_hz, params.lowpass_order)
        else:
            filt = sig_post.copy()

        start_label = 'Start (manuell)' if is_manual else f'Start ({params.trigger_mode})'
        start_color = '#E91E63' if is_manual else 'red'
        ax.plot(t_s, signal - offset, color='#cccccc', lw=0.8,
                label='Roh (zentriert)', zorder=1)
        ax.plot(t_abs, filt, color='#2196F3', lw=1.3, label='Gefiltert', zorder=2)
        ax.axvline(t_s[trig], color=start_color, ls='--', lw=1.2, alpha=0.7,
                   label=start_label, zorder=3)
        ax.axhline(0, color='grey', lw=0.5)

        title = f'[{self.idx + 1}/{len(self.recordings)}]  {os.path.basename(rec["path"])}'

        if result is not None:
            pos = analysis['pos_peak_idx']
            neg = analysis['neg_peak_idx']
            allp = analysis['all_peak_idx']
            ax.plot(t_abs[pos], filt[pos], 'v', color='#E91E63', ms=7, label='Maxima')
            ax.plot(t_abs[neg], filt[neg], '^', color='#FF9800', ms=7, label='Minima')

            # Obere Hüllkurve durch die Peak-Maxima (Maxima = Referenz, liegen darunter).
            # Abklingrate aus log-linearem Fit der Peaks; Amplitude so verankert,
            # dass die Kurve das höchste Maximum berührt und alle Peaks darunter liegen.
            peak_t = analysis['t_post'][allp]
            peak_a = np.abs(filt[allp])
            fit_mask = peak_a > 1e-9
            if fit_mask.sum() >= 2:
                slope, _ = np.polyfit(peak_t[fit_mask], np.log(peak_a[fit_mask]), 1)
                decay = -slope  # Abklingrate [1/s]
                if decay <= 0:  # Fit klingt nicht ab (z.B. Anlaufphase) → physikalische Rate
                    decay = result['zeta'] * result['omega_n']
                # C = obere Schranke: env(t_i) = C·exp(-decay·t_i) ≥ jedes Maximum
                C = float(np.max(peak_a[fit_mask] * np.exp(decay * peak_t[fit_mask])))
                env = C * np.exp(-decay * analysis['t_post'])
                ax.plot(t_abs, env, '--', color='#4CAF50', lw=1.4,
                        label=f'Hüllkurve (Abkling {decay:.1f}/s)')
                ax.plot(t_abs, -env, '--', color='#4CAF50', lw=1.4)

            # Peak-Abkling-Balken
            amps = np.abs(filt[allp])
            axp.bar(t_abs[allp], amps, width=(t_abs[-1] - t_abs[0]) / max(len(allp) * 3, 1),
                    color='#2196F3', alpha=0.75)
        else:
            ax.text(0.5, 0.5, '⚠ Nicht genug Peaks – Parameter anpassen',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='#B00020',
                    bbox=dict(boxstyle='round', fc='#FFEBEE', ec='#B00020'))

        ax.set_title(title, fontsize=10)
        ax.legend(loc='upper right', fontsize=8, ncol=2)

    def _update_info(self, rec, fs, result, summary, n_failed):
        lines = []
        lines.append('━━ Aktuelle Messung ━━')
        lines.append(f'[{self.idx + 1}/{len(self.recordings)}] {os.path.basename(rec["path"])}')
        lines.append(f'fs ≈ {fs:.0f} Hz')
        lines.append('')
        if result is not None:
            lines.append(f"ω_d = {result['omega_d']:8.2f} rad/s   (f_d {result['f_d']:.2f} Hz)")
            lines.append(f"ω_n = {result['omega_n']:8.2f} rad/s   (f_n {result['f_n']:.2f} Hz)")
            lines.append(f"ζ   = {result['zeta']:8.4f}")
            lines.append(f"δ   = {result['delta']:8.4f}")
            lines.append(f"k   = {result['k']:10.5f} Nm/rad")
            lines.append(f"d   = {result['d']:10.6f} Nm·s/rad")
            lines.append(f"Peaks: {result['n_peaks']}")
        else:
            lines.append('⚠ Keine gültige Auswertung')
            lines.append('  (zu wenige Peaks)')

        lines.append('')
        lines.append('━━ Mittelung (alle Messungen) ━━')
        if summary is not None and summary.n_valid > 0:
            lines.append(f'Messungen: {summary.n_measurements}   '
                         f'valide: {summary.n_valid}   '
                         f'Ausreißer: {summary.n_outliers}')
            if n_failed:
                lines.append(f'ohne Peaks: {n_failed}')
            lines.append('')
            lines.append(f'ω_d = {summary.omega_d_mean:8.2f} ± {summary.omega_d_std:.2f} rad/s')
            lines.append(f'ω_n = {summary.omega_n_mean:8.2f} ± {summary.omega_n_std:.2f} rad/s')
            lines.append(f'ζ   = {summary.zeta_mean:8.4f} ± {summary.zeta_std:.4f}')
            lines.append(f'k   = {summary.k_mean:10.5f} ± {summary.k_std:.5f} Nm/rad')
            lines.append(f'd   = {summary.d_mean:10.6f} ± {summary.d_std:.6f} Nm·s/rad')
        else:
            lines.append('Keine valide Messung')
            if n_failed:
                lines.append(f'ohne Peaks: {n_failed}')

        self.info_text.set_text('\n'.join(lines))

    # ── Event-Handler ────────────────────────────────────────────────────────

    def _on_change(self, _=None):
        self._recompute()

    def _on_start_change(self, val):
        """Startpunkt manuell verschoben → Override für aktuelle Messung setzen."""
        if self._loading:
            return
        rec = self.recordings[self.idx]
        self.start_overrides[rec['path']] = float(val)
        self._recompute()

    def _on_auto_start(self, _):
        """Manuellen Startpunkt der aktuellen Messung verwerfen (zurück zu Auto)."""
        rec = self.recordings[self.idx]
        self.start_overrides.pop(rec['path'], None)
        self._recompute()

    def _on_sensor_change(self, _=None):
        """sensor_id geändert → CSVs mit neuer sensor_id neu laden."""
        if self._loading:
            return
        new_sid = self.sensor_id
        reloaded = []
        for rec in self.recordings:
            try:
                t_s, signal = tss.load_csv(rec['path'], new_sid)
                reloaded.append({'path': rec['path'], 't_s': t_s, 'signal': signal})
            except Exception:
                pass  # Datei ohne diese sensor_id → auslassen
        if reloaded:
            self.recordings = reloaded
            self.idx = min(self.idx, len(self.recordings) - 1)
        else:
            print(f'⚠ Keine Daten für sensor_id={new_sid} – Auswahl beibehalten.')
        self._recompute()

    def _on_prev(self, _):
        self._disarm_delete()
        if self.idx > 0:
            self.idx -= 1
            self._recompute()

    def _on_next(self, _):
        self._disarm_delete()
        if self.idx < len(self.recordings) - 1:
            self.idx += 1
            self._recompute()

    # ── Aufnahme löschen (Zwei-Klick-Sicherung) ──────────────────────────────

    def _recording_files(self, path: str) -> list:
        """Alle zu einer Aufnahme gehörenden Dateien (Roh, bearbeitet, Plot-PNG)."""
        if path.endswith('_bearbeitet.csv'):
            raw = path[:-len('_bearbeitet.csv')] + '.csv'
        else:
            raw = path
        base = os.path.splitext(raw)[0]                 # …/spirob_messung_<ts>
        edited = base + '_bearbeitet.csv'
        plot = base.replace('spirob_messung_', 'spirob_plot_') + '.png'
        return [f for f in (raw, edited, plot) if os.path.isfile(f)]

    def _disarm_delete(self):
        if self._delete_armed:
            self._delete_armed = False
            self.btn_delete.label.set_text('Aufnahme löschen')
            self.btn_delete.ax.set_facecolor('#FF8A65')
            self.fig.canvas.draw_idle()

    def _on_delete(self, _):
        # Erster Klick: nur scharf schalten
        if not self._delete_armed:
            self._delete_armed = True
            self.btn_delete.label.set_text('Wirklich löschen?')
            self.btn_delete.ax.set_facecolor('#E53935')
            self.fig.canvas.draw_idle()
            return

        # Zweiter Klick: tatsächlich löschen
        self._delete_armed = False
        self.btn_delete.label.set_text('Aufnahme löschen')
        self.btn_delete.ax.set_facecolor('#FF8A65')

        rec = self.recordings[self.idx]
        for f in self._recording_files(rec['path']):
            try:
                os.remove(f)
                print(f'Gelöscht: {f}')
            except OSError as e:
                print(f'⚠ Konnte {f} nicht löschen: {e}')

        self.start_overrides.pop(rec['path'], None)
        del self.recordings[self.idx]

        if not self.recordings:
            print('Keine Messungen mehr im Ordner – GUI wird geschlossen.')
            plt.close(self.fig)
            return

        self.idx = min(self.idx, len(self.recordings) - 1)
        self._recompute()

    def _on_reset(self, _):
        self.settings = dict(DEFAULT_SETTINGS)
        self._apply_settings_to_widgets()
        self._recompute()

    def _on_save(self, _):
        path = settings_path(self.folder)
        settings = self.current_settings_dict()

        results_block = None
        s = self._last_summary
        if s is not None:
            per_file = []
            for r in self._last_results:
                per_file.append({
                    'file': os.path.basename(r.filename),
                    'omega_d': _f(r.omega_d, 4),
                    'omega_n': _f(r.omega_n, 4),
                    'f_n': _f(r.f_n, 4),
                    'zeta': _f(r.zeta, 6),
                    'delta': _f(r.delta, 6),
                    'k': _f(r.k, 8),
                    'd': _f(r.d, 8),
                    'n_peaks': int(r.n_peaks),
                    'outlier': bool(r.is_outlier),
                })
            results_block = {
                'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                'folder': self.folder_name,
                'n_measurements': int(s.n_measurements),
                'n_valid': int(s.n_valid),
                'n_outliers': int(s.n_outliers),
                'n_failed': int(self._n_failed),
                'omega_d_mean': _f(s.omega_d_mean, 4),
                'omega_d_std': _f(s.omega_d_std, 4),
                'omega_n_mean': _f(s.omega_n_mean, 4),
                'omega_n_std': _f(s.omega_n_std, 4),
                'zeta_mean': _f(s.zeta_mean, 6),
                'zeta_std': _f(s.zeta_std, 6),
                'k_mean': _f(s.k_mean, 8),
                'k_std': _f(s.k_std, 8),
                'd_mean': _f(s.d_mean, 8),
                'd_std': _f(s.d_std, 8),
                'per_file': per_file,
            }

        doc = {'settings': settings}
        if results_block is not None:
            doc['results'] = results_block

        with open(path, 'w') as f:
            yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

        print(f'Gespeichert: {path}')
        self.ax_sig.set_title(f'✓ Gespeichert: {SETTINGS_FILENAME}',
                              color='#4CAF50', fontsize=10)
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Einstiegspunkt
# ──────────────────────────────────────────────────────────────────────────────

def resolve_folder(argv) -> str:
    if len(argv) > 1:
        folder = os.path.abspath(argv[1])
        if not os.path.isdir(folder):
            print(f'Ordner nicht gefunden: {folder}')
            sys.exit(1)
        return folder

    # Kein Argument: bevorzugt ein in JOINTS konfigurierter Ordner, sonst erster
    for joint_name in tss.JOINTS:
        cand = os.path.join(BUILDS_DIR, joint_name)
        if glob.glob(os.path.join(cand, 'spirob_messung_*.csv')):
            print(f'Kein Ordner angegeben – nutze JOINTS-Eintrag: {joint_name}')
            return cand

    folders = list_recording_folders()
    if not folders:
        print(f'Keine Messordner mit spirob_messung_*.csv unter {BUILDS_DIR}')
        sys.exit(1)
    print(f'Kein Ordner angegeben – nutze: {os.path.basename(folders[0])}')
    return folders[0]


def main():
    folder = resolve_folder(sys.argv)
    SysIdGUI(folder).run()


if __name__ == '__main__':
    main()
