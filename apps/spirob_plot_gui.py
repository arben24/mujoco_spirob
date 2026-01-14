#!/usr/bin/env python3
"""
GUI for plotting and analyzing MuJoCo SpiRob experiments.

This application provides a graphical interface to access all plotting and analysis
functions without writing Python code.
"""

import sys
import os
import subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QComboBox, QPushButton, QTextEdit, QGroupBox, QCheckBox,
    QLineEdit, QMessageBox, QSplitter, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QDesktopServices
from PyQt6.QtCore import QUrl

import matplotlib
matplotlib.use('QtAgg')  # Use Qt backend for matplotlib

from math_spirob.meta_analyzer import (
    crawl_experiments, load_summary_parquet,
    plot_force_distribution, plot_force_peak_usage
)
from math_spirob.plots import (
    load_run_metadata, filter_runs,
    plot_time_series, plot_comparison, plot_time_series_filtered, plot_comparison_filtered
)
from math_spirob.analyzer import load_experiment

class PlotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MuJoCo SpiRob Plot GUI")
        self.setGeometry(100, 100, 1200, 800)
        
        # Data
        self.all_runs = []
        self.filtered_runs = []
        self.available_sensors = []
        self.available_axes = ['X', 'Y', 'Z']
        self.available_metrics = ['raw', 'mean', 'std', 'min', 'max', 'skew', 'kurtosis']
        
        self.init_ui()
        self.load_data()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Run selection
        run_group = QGroupBox("Run Selection")
        run_layout = QVBoxLayout(run_group)
        
        self.run_list = QListWidget()
        self.run_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        run_layout.addWidget(QLabel("Available Runs:"))
        run_layout.addWidget(self.run_list)
        
        # Filters
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Controller:"))
        self.controller_filter = QLineEdit()
        self.controller_filter.textChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.controller_filter)
        
        filter_layout.addWidget(QLabel("Geom Type:"))
        self.geom_filter = QLineEdit()
        self.geom_filter.textChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.geom_filter)
        
        run_layout.addLayout(filter_layout)
        left_layout.addWidget(run_group)
        
        # Sensor/Metric selection
        sensor_group = QGroupBox("Data Selection")
        sensor_layout = QVBoxLayout(sensor_group)
        
        self.sensor_list = QListWidget()
        self.sensor_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        sensor_layout.addWidget(QLabel("Sensors:"))
        sensor_layout.addWidget(self.sensor_list)
        
        axes_layout = QHBoxLayout()
        axes_layout.addWidget(QLabel("Axes:"))
        self.axes_combo = QComboBox()
        self.axes_combo.addItems(self.available_axes)
        axes_layout.addWidget(self.axes_combo)
        
        axes_layout.addWidget(QLabel("Metric:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(self.available_metrics)
        axes_layout.addWidget(self.metric_combo)
        
        sensor_layout.addLayout(axes_layout)
        left_layout.addWidget(sensor_group)
        
        # Plot buttons
        plot_group = QGroupBox("Plot Actions")
        plot_layout = QVBoxLayout(plot_group)
        
        self.time_series_btn = QPushButton("Time Series")
        self.time_series_btn.clicked.connect(self.plot_time_series)
        plot_layout.addWidget(self.time_series_btn)
        
        self.comparison_btn = QPushButton("Comparison")
        self.comparison_btn.clicked.connect(self.plot_comparison)
        plot_layout.addWidget(self.comparison_btn)
        
        self.force_dist_btn = QPushButton("Force Distribution")
        self.force_dist_btn.clicked.connect(self.plot_force_distribution)
        plot_layout.addWidget(self.force_dist_btn)
        
        self.force_peak_btn = QPushButton("Force Peak Usage")
        self.force_peak_btn.clicked.connect(self.plot_force_peak_usage)
        plot_layout.addWidget(self.force_peak_btn)
        
        peak_metric_layout = QHBoxLayout()
        peak_metric_layout.addWidget(QLabel("Peak Metric:"))
        self.peak_metric_combo = QComboBox()
        self.peak_metric_combo.addItems(['force', 'share'])
        peak_metric_layout.addWidget(self.peak_metric_combo)
        plot_layout.addLayout(peak_metric_layout)
        
        self.open_video_btn = QPushButton("Open Video")
        self.open_video_btn.clicked.connect(self.open_selected_run_video)
        plot_layout.addWidget(self.open_video_btn)
        
        left_layout.addWidget(plot_group)
        
        # Status
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setPlainText("Ready")
        left_layout.addWidget(self.status_text)
        
        main_layout.addWidget(left_panel, 1)
        
        # Right panel: Info
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        info_group = QGroupBox("Run Information")
        info_layout = QVBoxLayout(info_group)
        
        self.run_info_text = QTextEdit()
        self.run_info_text.setReadOnly(True)
        info_layout.addWidget(self.run_info_text)
        
        right_layout.addWidget(info_group)
        main_layout.addWidget(right_panel, 1)
        
        # Connect signals
        self.run_list.itemSelectionChanged.connect(self.update_run_info)
        self.run_list.itemSelectionChanged.connect(self.update_sensors_for_selected_runs)
        
    def load_data(self):
        try:
            # Load runs
            self.all_runs = crawl_experiments()
            if not self.all_runs:
                raise FileNotFoundError("No experiment directories found in build/experiments")
            self.filtered_runs = self.all_runs.copy()
            self.update_run_list()
            
            # Load metadata for filters
            self.metadata_df = load_run_metadata()
            
            # Collect sensors from all runs
            self.collect_all_sensors()
            
            self.log_status("Data loaded successfully")
        except Exception as e:
            self.log_status(f"Error loading data: {e}")
            QMessageBox.warning(self, "Data Loading Error", f"Failed to load experiment data: {e}")
            # Set empty data
            self.all_runs = []
            self.filtered_runs = []
            self.available_sensors = []
            self.update_run_list()
            self.update_sensor_list()
    
    def update_run_list(self):
        self.run_list.clear()
        for run in self.filtered_runs:
            self.run_list.addItem(run)
    
    def update_sensor_list(self):
        self.sensor_list.clear()
        for sensor in self.available_sensors:
            self.sensor_list.addItem(sensor)
    
    def collect_all_sensors(self):
        """Collect all unique sensor names across all runs."""
        sensor_names = set()
        for run_id in self.all_runs:
            try:
                record, _ = load_experiment(run_id)
                for s in record.sensors:
                    sensor_names.add(s.name)
            except Exception as e:
                self.log_status(f"Warning: failed to load sensors for run {run_id}: {e}")
        self.available_sensors = sorted(sensor_names)
        self.update_sensor_list()
    
    def update_sensors_for_selected_runs(self):
        """Update sensor list based on currently selected runs."""
        runs = self.get_selected_runs()
        if not runs:
            # Default: all sensors across all runs
            self.collect_all_sensors()
            return
        
        sensor_names = set()
        for run_id in runs:
            try:
                record, _ = load_experiment(run_id)
                for s in record.sensors:
                    sensor_names.add(s.name)
            except Exception as e:
                self.log_status(f"Warning: failed to load sensors for run {run_id}: {e}")
        
        self.available_sensors = sorted(sensor_names)
        self.update_sensor_list()
    
    def apply_filters(self):
        filters = {}
        if self.controller_filter.text():
            filters['controller_info'] = self.controller_filter.text()
        if self.geom_filter.text():
            filters['geom_type'] = self.geom_filter.text()
        
        if filters:
            try:
                self.filtered_runs = filter_runs(self.metadata_df, filters)
            except Exception as e:
                self.log_status(f"Filter error: {e}")
                self.filtered_runs = self.all_runs.copy()
        else:
            self.filtered_runs = self.all_runs.copy()
        
        self.update_run_list()
        self.log_status(f"Filtered to {len(self.filtered_runs)} runs")
    
    def get_selected_runs(self):
        selected_items = self.run_list.selectedItems()
        return [item.text() for item in selected_items]
    
    def get_selected_sensors(self):
        selected_items = self.sensor_list.selectedItems()
        return [item.text() for item in selected_items]
    
    def update_run_info(self):
        selected_runs = self.get_selected_runs()
        if len(selected_runs) == 1:
            run_id = selected_runs[0]
            try:
                df = load_summary_parquet()
                run_data = df.filter(df['run_id'] == run_id)
                if not run_data.is_empty():
                    info = f"Run ID: {run_id}\n"
                    info += f"L_target: {run_data.select('L_target').item()}\n"
                    info += f"Base_d: {run_data.select('base_d').item()}\n"
                    info += f"Controller: {run_data.select('controller_info').item()}\n"
                    info += f"Geom Type: {run_data.select('geom_type').item()}\n"
                    self.run_info_text.setPlainText(info)
                else:
                    self.run_info_text.setPlainText("No summary data available")
            except Exception as e:
                self.run_info_text.setPlainText(f"Error loading info: {e}")
        else:
            self.run_info_text.setPlainText(f"{len(selected_runs)} runs selected")
    
    def plot_time_series(self):
        runs = self.get_selected_runs()
        sensors = self.get_selected_sensors()
        axes = [self.axes_combo.currentText()]
        metric = self.metric_combo.currentText()
        
        if not runs:
            QMessageBox.warning(self, "Selection Error", "Please select at least one run")
            return
        if not sensors:
            QMessageBox.warning(self, "Selection Error", "Please select at least one sensor")
            return
        
        # Check for missing columns in each run
        for run in runs:
            try:
                record, df = load_experiment(run)
                for sensor in sensors:
                    for ax in axes:
                        col = f"{sensor}_{ax}"
                        if col not in df.columns:
                            col = sensor
                        if col not in df.columns:
                            self.log_status(f"Warning: Column {col} not found in run {run}, plot may be incomplete")
            except Exception as e:
                self.log_status(f"Error checking columns for run {run}: {e}")
        
        try:
            plot_time_series(runs, sensors, axes, metric)
            self.log_status("Time series plot created")
        except Exception as e:
            self.log_status(f"Plot error: {e}")
            QMessageBox.critical(self, "Plot Error", f"Failed to create plot: {e}")
    
    def plot_comparison(self):
        runs = self.get_selected_runs()
        sensors = self.get_selected_sensors()
        if not sensors:
            QMessageBox.warning(self, "Selection Error", "Please select a sensor")
            return
        sensor = sensors[0]  # Take first
        axis = self.axes_combo.currentText()
        metric = self.metric_combo.currentText()
        
        if len(runs) < 2:
            QMessageBox.warning(self, "Selection Error", "Please select at least two runs for comparison")
            return
        
        # Check for missing columns in each run
        for run in runs:
            try:
                record, df = load_experiment(run)
                col = f"{sensor}_{axis}_{metric}"
                if col not in df.columns:
                    col = f"{sensor}_{metric}"
                if col not in df.columns:
                    self.log_status(f"Warning: Column {col} not found in run {run}, comparison may be incomplete")
            except Exception as e:
                self.log_status(f"Error checking columns for run {run}: {e}")
        
        try:
            plot_comparison(runs, sensor, axis, metric)
            self.log_status("Comparison plot created")
        except Exception as e:
            self.log_status(f"Plot error: {e}")
            QMessageBox.critical(self, "Plot Error", f"Failed to create plot: {e}")
    
    def plot_force_distribution(self):
        runs = self.get_selected_runs()
        if not runs:
            QMessageBox.warning(self, "Selection Error", "Please select a run")
            return
        run_id = runs[0]  # Take first
        
        try:
            plot_force_distribution(run_id)
            self.log_status("Force distribution plot created")
        except Exception as e:
            self.log_status(f"Plot error: {e}")
            QMessageBox.critical(self, "Plot Error", f"Failed to create plot: {e}")
    
    def plot_force_peak_usage(self):
        runs = self.get_selected_runs()
        if not runs:
            QMessageBox.warning(self, "Selection Error", "Please select a run")
            return
        run_id = runs[0]  # Take first
        metric = self.peak_metric_combo.currentText()
        
        try:
            plot_force_peak_usage(run_id, metric)
            self.log_status("Force peak usage plot created")
        except Exception as e:
            self.log_status(f"Plot error: {e}")
            QMessageBox.critical(self, "Plot Error", f"Failed to create plot: {e}")
    
    def get_video_path_for_run(self, run_id: str) -> str:
        """Get the video path for a given run ID."""
        base_dir = "build/experiments"
        return f"{base_dir}/{run_id}/video.mp4"
    
    def open_selected_run_video(self):
        """Open the video for the selected run."""
        selected_runs = self.get_selected_runs()
        if len(selected_runs) != 1:
            QMessageBox.warning(self, "Selection Error", "Please select exactly one run to open its video")
            return
        
        run_id = selected_runs[0]
        video_path = self.get_video_path_for_run(run_id)
        
        if not os.path.exists(video_path):
            QMessageBox.information(self, "Video not found", f"No video.mp4 found for run {run_id} at {video_path}")
            return
        
        try:
            # Use subprocess to open with system default application
            subprocess.Popen(['xdg-open', video_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.log_status(f"Opened video for {run_id}")
        except Exception as e:
            QMessageBox.critical(self, "Video Error", f"Failed to open video: {e}")
            self.log_status(f"Failed to open video for {run_id}: {e}")

    def log_status(self, message: str) -> None:
        """
        Append a status message to the status_text widget.
        """
        if hasattr(self, "status_text") and self.status_text is not None:
            self.status_text.append(message)
        else:
            print(message)


def main():
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')  # Modern style
        
        window = PlotGUI()
        window.show()
        
        sys.exit(app.exec())
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()