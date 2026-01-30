#!/usr/bin/env python3
"""
Spirob Aktuation GUI
Steuerung des dual-motor robotics actuation systems über serielle Schnittstelle.
"""

import tkinter as tk
from tkinter import scrolledtext
import customtkinter as ctk
import serial
import serial.tools.list_ports
import threading
import time
from queue import Queue
from typing import Optional, List
import logging

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FIRMWARE CONFIGURATION (basierend auf main.cpp)
# ============================================================================

FIRMWARE_CONFIG = {
    "baudrate": 115200,
    "delimiter": "\n",
    "num_actuators": 2,
    "max_force": 10.0,  # Maximale Kraft für GUI (N)
    "min_force": 0.0,
    "default_force": 0.0,
    "rate_limit_ms": 50,  # Max. Befehl alle 50ms
}

# ============================================================================
# SERIAL COMMUNICATION THREAD
# ============================================================================

class SerialReaderThread(threading.Thread):
    """
    Liest kontinuierlich von der seriellen Schnittstelle und verarbeitet
    eingehende Daten in einem separaten Thread.
    """
    
    def __init__(self, output_queue: Queue):
        super().__init__(daemon=True)
        self.serial_port: Optional[serial.Serial] = None
        self.output_queue = output_queue
        self.running = False
        self.stop_event = threading.Event()
    
    def connect(self, port: str, baudrate: int) -> bool:
        """Verbinde mit dem seriellen Port."""
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=1
            )
            self.running = True
            logger.info(f"Verbunden mit {port} @ {baudrate} baud")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Verbinden: {e}")
            return False
    
    def disconnect(self):
        """Trenne die serielle Verbindung."""
        self.running = False
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            logger.info("Verbindung getrennt")
    
    def send_command(self, command: str) -> bool:
        """Sende einen Befehl an den ESP32."""
        if not self.serial_port or not self.serial_port.is_open:
            logger.error("Keine Verbindung aktiv")
            return False
        
        try:
            # Stelle sicher, dass der Befehl mit Newline endet
            if not command.endswith("\n"):
                command += "\n"
            
            self.serial_port.write(command.encode())
            logger.debug(f"Befehl gesendet: {command.strip()}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Senden: {e}")
            return False
    
    def run(self):
        """Hauptschleife für das Lesen von der seriellen Schnittstelle."""
        while not self.stop_event.is_set():
            if not self.running or not self.serial_port:
                time.sleep(0.1)
                continue
            
            try:
                if self.serial_port.in_waiting > 0:
                    # Lese eine komplette Zeile
                    line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self.output_queue.put(line)
            except Exception as e:
                logger.error(f"Fehler beim Lesen: {e}")
                self.running = False
            
            time.sleep(0.01)  # Kleine Verzögerung zur CPU-Entlastung


# ============================================================================
# MAIN GUI APPLICATION
# ============================================================================

class SpirobGUI(ctk.CTk):
    """
    Hauptanwendung für die Steuerung des Spirob Aktuation Systems.
    """
    
    def __init__(self):
        super().__init__()
        
        self.title("Spirob Aktuation Control")
        self.geometry("900x700")
        self.resizable(True, True)
        
        # Serielle Kommunikation
        self.output_queue = Queue()
        self.serial_thread = SerialReaderThread(self.output_queue)
        self.serial_thread.start()
        
        # State tracking
        self.is_connected = False
        self.last_slider_send = [0, 0]  # Zeitmessungen für Rate Limiting
        self.slider_values = [0.0, 0.0]  # Aktuelle Schieberegler-Werte
        
        # UI Setup
        self._setup_ui()
        
        # Starte den Feedback-Loop
        self.after(100, self._check_serial_output)
    
    def _setup_ui(self):
        """Baue die UI auf."""
        
        # ===== TOP FRAME: Verbindung =====
        connection_frame = ctk.CTkFrame(self, fg_color="gray20")
        connection_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            connection_frame,
            text="🔌 Verbindung",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # COM-Port Auswahl
        port_frame = ctk.CTkFrame(connection_frame)
        port_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(port_frame, text="COM-Port:").pack(side="left", padx=5)
        self.port_dropdown = ctk.CTkComboBox(
            port_frame,
            values=self._get_available_ports(),
            width=150
        )
        self.port_dropdown.pack(side="left", padx=5)
        
        # Refresh-Button für COM-Ports
        ctk.CTkButton(
            port_frame,
            text="🔄 Aktualisieren",
            command=self._refresh_ports,
            width=120
        ).pack(side="left", padx=5)
        
        # Connect/Disconnect Buttons
        button_frame = ctk.CTkFrame(connection_frame)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        self.connect_button = ctk.CTkButton(
            button_frame,
            text="✅ Verbinden",
            fg_color="green",
            command=self._connect,
            width=150
        )
        self.connect_button.pack(side="left", padx=5)
        
        self.disconnect_button = ctk.CTkButton(
            button_frame,
            text="❌ Trennen",
            fg_color="red",
            command=self._disconnect,
            width=150,
            state="disabled"
        )
        self.disconnect_button.pack(side="left", padx=5)
        
        # Verbindungsstatus
        self.status_label = ctk.CTkLabel(
            connection_frame,
            text="Status: Getrennt",
            text_color="red"
        )
        self.status_label.pack(anchor="w", padx=10, pady=(5, 10))
        
        # ===== MIDDLE FRAME: Slider für Motoren =====
        control_frame = ctk.CTkFrame(self, fg_color="gray20")
        control_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            control_frame,
            text="⚙️  Motor Steuerung (Kraft in Newton)",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Erstelle Slider für jeden Motor
        self.sliders = []
        self.slider_labels = []
        
        for i in range(FIRMWARE_CONFIG["num_actuators"]):
            motor_frame = ctk.CTkFrame(control_frame)
            motor_frame.pack(fill="x", padx=10, pady=10)
            
            # Motor Label
            label = ctk.CTkLabel(
                motor_frame,
                text=f"Motor {i} (Index {i}, Servo ID {i+1})",
                font=("Arial", 12, "bold")
            )
            label.pack(anchor="w", pady=(0, 5))
            
            # Slider Frame
            slider_inner = ctk.CTkFrame(motor_frame)
            slider_inner.pack(fill="x")
            
            # Slider
            slider = ctk.CTkSlider(
                slider_inner,
                from_=FIRMWARE_CONFIG["min_force"],
                to=FIRMWARE_CONFIG["max_force"],
                number_of_steps=100,
                command=lambda value, motor_id=i: self._on_slider_change(motor_id, value)
            )
            slider.pack(side="left", fill="x", expand=True, padx=(0, 10))
            self.sliders.append(slider)
            
            # Wert-Display
            value_label = ctk.CTkLabel(
                slider_inner,
                text=f"0.0 N",
                width=80,
                font=("Arial", 11, "bold")
            )
            value_label.pack(side="left", padx=5)
            self.slider_labels.append(value_label)
            
            # Buttons: Start/Stop für diesen Motor
            button_inner = ctk.CTkFrame(motor_frame)
            button_inner.pack(fill="x", pady=(5, 0))
            
            ctk.CTkButton(
                button_inner,
                text=f"Start M{i}",
                fg_color="green",
                command=lambda motor_id=i: self._start_motor(motor_id),
                width=100
            ).pack(side="left", padx=5)
            
            ctk.CTkButton(
                button_inner,
                text=f"Stop M{i}",
                fg_color="orange",
                command=lambda motor_id=i: self._stop_motor(motor_id),
                width=100
            ).pack(side="left", padx=5)
        
        # ===== EMERGENCY STOP =====
        emergency_frame = ctk.CTkFrame(control_frame)
        emergency_frame.pack(fill="x", padx=10, pady=15)
        
        self.emergency_button = ctk.CTkButton(
            emergency_frame,
            text="🛑 NOTFALL - ALLE STOPPEN",
            fg_color="#FF0000",
            text_color="white",
            font=("Arial", 14, "bold"),
            command=self._emergency_stop,
            height=50
        )
        self.emergency_button.pack(fill="x")
        
        # ===== BOTTOM FRAME: Feedback =====
        feedback_frame = ctk.CTkFrame(self, fg_color="gray20")
        feedback_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            feedback_frame,
            text="📨 Feedback vom ESP32",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Textfeld für Ausgabe (nutze tkinter.scrolledtext, da customtkinter begrenzt ist)
        self.feedback_text = scrolledtext.ScrolledText(
            feedback_frame,
            height=8,
            width=100,
            bg="black",
            fg="lime",
            font=("Courier", 10),
            wrap=tk.WORD
        )
        self.feedback_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Clear Button
        ctk.CTkButton(
            feedback_frame,
            text="🧹 Feedback löschen",
            command=self._clear_feedback,
            width=150
        ).pack(anchor="e", padx=10, pady=(0, 10))
    
    def _get_available_ports(self) -> List[str]:
        """Finde verfügbare COM-Ports."""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        return ports if ports else ["Keine Ports gefunden"]
    
    def _refresh_ports(self):
        """Aktualisiere die Liste der COM-Ports."""
        ports = self._get_available_ports()
        self.port_dropdown.configure(values=ports)
        if ports:
            self.port_dropdown.set(ports[0])
        self._log_feedback(f"✓ COM-Ports aktualisiert: {', '.join(ports)}")
    
    def _connect(self):
        """Verbinde mit dem ESP32."""
        port = self.port_dropdown.get()
        
        if not port or "Keine Ports" in port:
            self._log_feedback("❌ Kein COM-Port ausgewählt!")
            return
        
        if self.serial_thread.connect(port, FIRMWARE_CONFIG["baudrate"]):
            self.is_connected = True
            self.connect_button.configure(state="disabled")
            self.disconnect_button.configure(state="normal")
            self.port_dropdown.configure(state="disabled")
            self.status_label.configure(
                text=f"Status: Verbunden ({port})",
                text_color="green"
            )
            self._log_feedback(f"✅ Verbunden mit {port}")
            
            # Sende initial 'status' Befehl
            self.serial_thread.send_command("status")
            self.serial_thread.send_command("help")
        else:
            self._log_feedback(f"❌ Fehler beim Verbinden mit {port}")
    
    def _disconnect(self):
        """Trenne die Verbindung."""
        self.serial_thread.disconnect()
        self.is_connected = False
        self.connect_button.configure(state="normal")
        self.disconnect_button.configure(state="disabled")
        self.port_dropdown.configure(state="normal")
        self.status_label.configure(
            text="Status: Getrennt",
            text_color="red"
        )
        self._log_feedback("⏹️ Verbindung getrennt")
    
    def _on_slider_change(self, motor_id: int, value: float):
        """
        Callback wenn ein Slider bewegt wird.
        Implementiert Rate Limiting: max. alle 50ms senden.
        """
        self.slider_values[motor_id] = value
        self.slider_labels[motor_id].configure(text=f"{value:.1f} N")
        
        # Rate Limiting
        current_time = time.time() * 1000  # millisekunden
        if current_time - self.last_slider_send[motor_id] >= FIRMWARE_CONFIG["rate_limit_ms"]:
            if self.is_connected:
                command = f"set {motor_id} {value:.1f}"
                self.serial_thread.send_command(command)
                self.last_slider_send[motor_id] = current_time
                logger.debug(f"Slider command sent: {command}")
    
    def _start_motor(self, motor_id: int):
        """Starte einen Motor."""
        if self.is_connected:
            command = f"start {motor_id}"
            self.serial_thread.send_command(command)
            self._log_feedback(f"→ Befehl: {command}")
    
    def _stop_motor(self, motor_id: int):
        """Stoppe einen Motor."""
        if self.is_connected:
            command = f"stop {motor_id}"
            self.serial_thread.send_command(command)
            self._log_feedback(f"→ Befehl: {command}")
    
    def _emergency_stop(self):
        """Notfall-Stop: Alle Motoren stoppen."""
        if self.is_connected:
            command = "stop all"
            self.serial_thread.send_command(command)
            self._log_feedback(f"🛑 NOTFALL STOP: {command}")
            
            # Setze alle Slider auf 0
            for slider in self.sliders:
                slider.set(0.0)
    
    def _check_serial_output(self):
        """
        Prüfe kontinuierlich auf neue Ausgaben von der seriellen Schnittstelle.
        Diese Methode wird alle 100ms aufgerufen.
        """
        while not self.output_queue.empty():
            try:
                message = self.output_queue.get_nowait()
                self._log_feedback(message)
            except:
                pass
        
        # Plane nächsten Check
        self.after(100, self._check_serial_output)
    
    def _log_feedback(self, message: str):
        """Schreibe eine Nachricht ins Feedback-Textfeld."""
        self.feedback_text.insert("end", message + "\n")
        self.feedback_text.see("end")  # Scrolle zum Ende
    
    def _clear_feedback(self):
        """Lösche das Feedback-Textfeld."""
        self.feedback_text.delete("1.0", "end")
    
    def on_closing(self):
        """Cleanup bei Programmende."""
        self.serial_thread.disconnect()
        self.serial_thread.stop_event.set()
        self.destroy()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Setze das Theme
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    app = SpirobGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
