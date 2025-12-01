import mujoco as mj
import mujoco.viewer as viewer
import time
import numpy as np
import math_spirob.spirob_generator as sg
from typing import Dict
from collections import defaultdict
import sys 

# Importiere PyQtGraph und das zugrundeliegende Qt-Framework
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, QMutex, QMutexLocker

# ---------------------------------------------------------
# CONFIG FLAGS
# ---------------------------------------------------------
USE_VIEWER = True          
REALTIME = True             
VIEWER_PASSIVE = True       # Muss True sein, damit das Skript die Kontrolle behält
SIM_TIME = 60.0             
# ---------------------------------------------------------
# Plot-Frequenz: Die Plots werden jetzt vom QTimer gesteuert.
PLOT_INTERVAL_MS = 50       # Aktualisiere Plots alle 50 ms (ca. 20 Hz)
MAX_HISTORY = 1000           # Wie viele Datenpunkte angezeigt werden sollen
# ---------------------------------------------------------
PLOT_ACC_DATA = True        
PLOT_GYRO_DATA = True       
PLOT_TENDONFRC_DATA = True  
PLOT_TENDONPOS_DATA = True  
PLOT_TENDOONVEL_DATA = True 
PLOT_JOINTPOS_DATA = True   
PLOT_JOINTVEL_DATA = True   
# ---------------------------------------------------------

# --- MUJOCO SETUP ---
xml_string = sg.generate_xml_string(
    L_target=0.30, base_d=0.06, tip_d=0.01, Delta_theta_deg=30,
    model_name="spiral_chain_plot", auto_format=True
)
spec = mj.MjSpec.from_string(xml_string)
model = spec.compile()
data = mj.MjData(model)

# --- DATENSTRUKTUREN ---
time_history = []
# Dies sind jetzt globale, synchronisierte Listen
global_data_history: Dict[str, Dict[str, list]] = {
    "acc": {}, "gyro": {}, "tendon_frc": {}, "tendon_pos": {}, 
    "tendon_vel": {}, "joint_pos": {}, "joint_vel": {}
}
sensor_map = {
    mj.mjtSensor.mjSENS_ACCELEROMETER: ("acc", PLOT_ACC_DATA),
    mj.mjtSensor.mjSENS_GYRO: ("gyro", PLOT_GYRO_DATA),
    mj.mjtSensor.mjSENS_TENDONACTFRC: ("tendon_frc", PLOT_TENDONFRC_DATA),
    mj.mjtSensor.mjSENS_TENDONPOS: ("tendon_pos", PLOT_TENDONPOS_DATA),
    mj.mjtSensor.mjSENS_TENDONVEL: ("tendon_vel", PLOT_TENDOONVEL_DATA),
    mj.mjtSensor.mjSENS_JOINTPOS: ("joint_pos", PLOT_JOINTPOS_DATA),
    mj.mjtSensor.mjSENS_JOINTVEL: ("joint_vel", PLOT_JOINTVEL_DATA),
}

for i in range(model.nsensor):
    sensor_type = model.sensor_type[i]
    sensor_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)
    for type_key, (data_dict_key, do_plot) in sensor_map.items():
        if sensor_type == type_key and do_plot:
            global_data_history[data_dict_key][sensor_name] = []


# ---------------------------------------------------------
# KLASSE FÜR DEN PLOT-MANAGER (QT-GUI)
# ---------------------------------------------------------

class PlotManager(QMainWindow):
    """Hauptfenster zur Verwaltung der PyQtGraph-Plots."""
    def __init__(self, history_data: Dict):
        super().__init__()
        self.setWindowTitle("Live MuJoCo Sensors (PyQtGraph)")
        self.setGeometry(100, 100, 1200, 800)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.history_data = history_data
        self.plot_widgets = defaultdict(dict)
        self.curves = defaultdict(dict)
        
        pg.setConfigOptions(antialias=True) # Schaltet Anti-Aliasing ein
        
        self._setup_plots()
        
        # Timer für die Plot-Aktualisierung
        self.timer = QTimer()
        self.timer.setInterval(PLOT_INTERVAL_MS)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start()

    def _setup_plots(self):
        """Erstellt alle PyQtGraph PlotWidgets basierend auf der Konfiguration."""
        
        for plot_key, sensor_dict in self.history_data.items():
            if not sensor_dict: continue

            # Dimension prüfen (hier vereinfacht: 3D vs. 1D)
            any_name = next(iter(sensor_dict))
            sensor_dim = model.sensor_dim[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, any_name)]
            is_vector = (sensor_dim == 3)
            
            if is_vector:
                # 3D Sensor (z.B. ACC, GYRO) -> 3 Subplots (X, Y, Z)
                labels = ["X-Achse", "Y-Achse", "Z-Achse"]
                for i, label in enumerate(labels):
                    plot_widget = pg.PlotWidget(title=f"{plot_key.upper()} - {label}")
                    plot_widget.addLegend()
                    self.layout.addWidget(plot_widget)
                    
                    for name in sensor_dict.keys():
                        # Erstelle die Kurve
                        curve = plot_widget.plot(pen=pg.intColor(len(self.curves[plot_key])))
                        curve.setData([], [])
                        self.curves[plot_key][(name, i)] = curve
                        
                    self.plot_widgets[plot_key][i] = plot_widget # Speichere das Widget

            else:
                # 1D Sensor (z.B. FRC, POS, VEL) -> 1 Plot
                plot_widget = pg.PlotWidget(title=f"{plot_key.upper()} Daten")
                plot_widget.addLegend()
                self.layout.addWidget(plot_widget)
                
                for name in sensor_dict.keys():
                    curve = plot_widget.plot(name=name, pen=pg.intColor(len(self.curves[plot_key])))
                    curve.setData([], [])
                    self.curves[plot_key][(name, 'scalar')] = curve
                    
                self.plot_widgets[plot_key]['scalar'] = plot_widget

    def update_plots(self):
        """Wird vom QTimer aufgerufen, um alle Kurven schnell zu aktualisieren."""
        
# NEU: Lock setzen, bevor Daten gelesen werden
        with QMutexLocker(MujocoThread.data_mutex):
            # Prüfen, ob die Daten synchron sind, bevor wir fortfahren
            if not time_history or len(global_data_history['acc'][next(iter(global_data_history['acc']))]) != len(time_history):
                 # Wenn die Längen nicht übereinstimmen, überspringen wir dieses Update
                 return

            current_time = time_history[-1] if time_history else 0
            
            for plot_key, sensor_dict in self.history_data.items():
                if not sensor_dict: continue

                any_name = next(iter(sensor_dict))
                sensor_dim = model.sensor_dim[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, any_name)]
                is_vector = (sensor_dim == 3)

                if is_vector:
                    # 3D Sensor
                    for i in range(3): # X, Y, Z
                        plot_widget = self.plot_widgets[plot_key][i]
                        
                        # Definiere X-Achse (Time History, limitiert)
                        t_data = np.array(time_history[-MAX_HISTORY:])
                        plot_widget.setXRange(max(0, current_time - (MAX_HISTORY * model.opt.timestep)), current_time)

                        for name in sensor_dict.keys():
                            curve = self.curves[plot_key][(name, i)]
                            
                            # Daten (limitiert und konvertiert)
                            y_data_list = sensor_dict[name]
                            if not y_data_list: continue
                            
                            y_data = np.array(y_data_list[-MAX_HISTORY:])[:, i]
                            
                            # Setzt nur die neuen Daten (extrem schnell!)
                            curve.setData(t_data, y_data)
                
                else:
                    # 1D Sensor
                    plot_widget = self.plot_widgets[plot_key]['scalar']
                    t_data = np.array(time_history[-MAX_HISTORY:])
                    plot_widget.setXRange(max(0, current_time - (MAX_HISTORY * model.opt.timestep)), current_time)

                    for name in sensor_dict.keys():
                        curve = self.curves[plot_key][(name, 'scalar')]
                        
                        y_data_list = sensor_dict[name]
                        if not y_data_list: continue

                        y_data = np.array(y_data_list[-MAX_HISTORY:]).squeeze()
                        curve.setData(t_data, y_data)


# ---------------------------------------------------------
# HAUPTSCHLEIFE FÜR MUJOCO
# ---------------------------------------------------------

class MujocoThread(QThread):
    """Führt die MuJoCo-Simulation in einem separaten Thread aus."""
    finished = pyqtSignal()

    # NEU: Der Mutex wird der Klasse hinzugefügt
    data_mutex = QMutex()
    
    def __init__(self, model, data, sim_time, viewer_handle):
        super().__init__()
        self.model = model
        self.data = data
        self.sim_time = sim_time
        self.v = viewer_handle
        self._running = True

    def stop(self):
        self._running = False
        
    def run(self):
        start_wall = time.time()
        print("MuJoCo Thread gestartet.")

        while self._running and self.v.is_running() and (self.data.time < self.sim_time):
            step_start = time.time()
            
            # --- Simulationsschritt ---
            mj.mj_step(self.model, self.data)
# NEU: Lock setzen, bevor Daten geändert werden
            # QMutexLocker hält den Lock, bis der Block beendet ist (thread-safe)
            with QMutexLocker(self.data_mutex):
                # --- Datensammlung (nur hier drinnen) ---
                time_history.append(self.data.time)

                for key, sensor_dict in global_data_history.items():
                    for name in sensor_dict.keys():
                        global_data_history[key][name].append(self.data.sensor(name).data.copy())
            # Viewer synchronisieren
            self.v.sync()

            # Echtzeit-Synchronisation
            if REALTIME:
                dt = self.model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)
        
        self.v.close()
        print(f"MuJoCo Simulation beendet. Dauer: {time.time() - start_wall:.2f} Sekunden")
        self.finished.emit()


if __name__ == '__main__':
    
    # 1. QT Application initialisieren
    app = QApplication(sys.argv)
    
    # 2. MuJoCo Viewer starten (passiv)
    launch_fn = viewer.launch_passive
    v = launch_fn(model, data)

    # 3. Plot Manager GUI starten
    plot_window = PlotManager(global_data_history)
    plot_window.show()

    # 4. MuJoCo Simulation in separatem Thread starten
    # Der Haupt-Thread (die Qt-App) wird NICHT blockiert!
    mujoco_thread = MujocoThread(model, data, SIM_TIME, v)
    
    # Sicherstellen, dass die App beendet wird, wenn der Sim-Thread fertig ist
    def close_app():
        plot_window.timer.stop()
        plot_window.close()
        app.quit()
        
    mujoco_thread.finished.connect(close_app)
    mujoco_thread.start()
    
    # 5. QT Application Loop starten (blockiert hier)
    sys.exit(app.exec())