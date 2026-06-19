"""
SpiRob Sensor Aufzeichnungsskript - Hochfrequent & YZ-Magnitude
===============================================================
- Verwendet den Hardware-Zeitstempel (t_us) für perfekte zeitliche Synchronisation.
- Berechnet die YZ-Magnitude: sqrt(AccY^2 + AccZ^2).
- Optimierter serieller Leseprozess für maximale Abtastrate (ohne time.sleep).
"""

import serial
import struct
import time
import os
import collections
import csv
from datetime import datetime
import math
import matplotlib.pyplot as plt

# ================== EINSTELLUNGEN ==================
SERIAL_PORT = '/dev/ttyUSB1'  # Anpassen!
BAUD_RATE = 1000000

# Aufnahme-Einstellungen
TRIGGER_THRESHOLD_G = 0.5     # Änderung in 'g' zwischen zwei Messungen, die den Trigger auslöst
PRE_RECORD_TIME_S = 0.5       # Wie viele Sekunden VOR dem Trigger gespeichert werden sollen
RECORD_TIME_S = 2.0           # Wie viele Sekunden NACH dem Trigger aufgenommen werden sollen
EXPECTED_FPS = 500            # Sehr großzügig geschätzt für die Puffergröße

# ================== BINÄR FORMAT ==================
FRAME_HDR_0 = 0xAA
FRAME_HDR_1 = 0x55
FRAME_FIXED_SIZE = 2 + 4 + 4 + 1
SENSOR_PACKET_SIZE = 1 + 6 * 4
MAX_SENSORS_PER_FRAME = 16

def ensure_builds_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    builds_dir = os.path.join(script_dir, 'builds')
    os.makedirs(builds_dir, exist_ok=True)
    return builds_dir

def parse_serial_stream(ser, callback):
    """Hochfrequentes Einlesen des binären Streams ohne künstliche Delays."""
    buffer = bytearray()
    
    # ESP32 ggf. in richtigen Modus versetzen und Puffer leeren
    ser.reset_input_buffer()
    ser.write(b'b')
    
    while True:
        try:
            # Blockierendes Lesen: max(1, in_waiting) sorgt dafür, dass sofort 
            # gelesen wird, wenn was da ist, aber die CPU nicht 100% rödelt, wenn nichts da ist.
            bytes_to_read = max(1, ser.in_waiting)
            new_data = ser.read(bytes_to_read)
            if new_data:
                buffer.extend(new_data)

            while True:
                idx = buffer.find(bytes([FRAME_HDR_0, FRAME_HDR_1]))
                if idx == -1:
                    if len(buffer) > 1:
                        buffer = buffer[-1:]
                    break

                if idx > 0:
                    del buffer[:idx]

                if len(buffer) < FRAME_FIXED_SIZE:
                    break

                frame_id, t_us, n = struct.unpack_from("<IIB", buffer, 2)

                if n == 0 or n > MAX_SENSORS_PER_FRAME:
                    del buffer[:2]
                    continue

                total_len = FRAME_FIXED_SIZE + n * SENSOR_PACKET_SIZE
                if len(buffer) < total_len:
                    break

                payload = buffer[FRAME_FIXED_SIZE:total_len]
                del buffer[:total_len]

                offset = 0
                for _ in range(n):
                    pkt = payload[offset:offset + SENSOR_PACKET_SIZE]
                    sensor_id = pkt[0]
                    accX, accY, accZ, magX, magY, magZ = struct.unpack("<ffffff", pkt[1:])
                    
                    callback(t_us, frame_id, sensor_id, accX, accY, accZ, magX, magY, magZ)
                    offset += SENSOR_PACKET_SIZE
                    
        except KeyboardInterrupt:
            print("\nManuell abgebrochen.")
            break
        except Exception as e:
            print(f"Lesefehler: {e}")
            break

def main():
    builds_dir = ensure_builds_dir()
    
    max_history = int(PRE_RECORD_TIME_S * EXPECTED_FPS * 4) 
    pre_trigger_buffer = collections.deque(maxlen=max_history)
    
    recorded_data = []
    
    is_recording = False
    trigger_t_us = 0      # Speichert den *Hardware*-Zeitstempel des Triggers
    last_acc = {}

    print("=" * 50)
    print(" Warte auf serielle Verbindung...")
    print("=" * 50)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"Verbunden. Warte auf Bewegung (Schwellwert {TRIGGER_THRESHOLD_G}g)...")
    except Exception as e:
        print(f"Fehler beim Öffnen des Ports: {e}")
        return

    def on_data_received(t_us, frame_id, sensor_id, accX, accY, accZ, magX, magY, magZ):
        nonlocal is_recording, trigger_t_us, last_acc, pre_trigger_buffer, recorded_data
        
        # YZ-Magnitude berechnen
        acc_mag_YZ = accY + accZ
        
        row = (t_us, frame_id, sensor_id, accX, accY, accZ, acc_mag_YZ, magX, magY, magZ)
        
        if not is_recording:
            pre_trigger_buffer.append(row)
            
            # Schwellwert-Erkennung über die Magnitude von Y und Z (oder allen 3)
            # Da X als Schwerkraft konstant bleibt (wenn wir uns nur um X drehen), 
            # betrachten wir hier nur die Änderung in Y und Z für den sauberen Trigger.
            if sensor_id in last_acc:
                l_y, l_z = last_acc[sensor_id]
                delta = math.sqrt((accY - l_y)**2 + (accZ - l_z)**2)
                
                if delta >= TRIGGER_THRESHOLD_G:
                    print(f"\n[!] TRIGGER AUSGELÖST bei Sensor {sensor_id}! Delta = {delta:.2f}g")
                    is_recording = True
                    trigger_t_us = t_us  # Hardware-Zeitpunkt des Triggers speichern!
                    
                    recorded_data.extend(pre_trigger_buffer)
                    print(f"Start der Aufzeichnung für {RECORD_TIME_S} Sekunden...")
            
            last_acc[sensor_id] = (accY, accZ)
            
        else:
            recorded_data.append(row)
            
            # Beende die Aufnahme nach RECORD_TIME_S, basierend auf der echten Hardware-Zeit!
            if (t_us - trigger_t_us) >= (RECORD_TIME_S * 1_000_000):
                raise StopIteration("Aufnahme abgeschlossen")

    try:
        parse_serial_stream(ser, on_data_received)
    except StopIteration:
        pass
    finally:
        ser.close()
        print("Serielle Verbindung geschlossen.")

    if not recorded_data:
        print("Keine Daten aufgezeichnet.")
        return

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(builds_dir, f"spirob_messung_{timestamp_str}.csv")
    plot_filename = os.path.join(builds_dir, f"spirob_plot_{timestamp_str}.png")

    print(f"\nSpeichere {len(recorded_data)} Datenpunkte...")

    # CSV Schreiben (inklusive der YZ Magnitude)
    headers = ['t_us', 'frame_id', 'sensor_id', 'accX', 'accY', 'accZ', 'acc_mag_YZ', 'magX', 'magY', 'magZ']
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(recorded_data)

    # Plot erstellen
    sensors = {}
    for row in recorded_data:
        t_us, fid, sid, ax, ay, az, amag_yz, mx, my, mz = row
        if sid not in sensors:
            sensors[sid] = {'t': [], 'ay': [], 'az': [], 'amag_yz': []}
        
        # Wir zentrieren die X-Achse so, dass der Trigger exakt bei 0.0 Sekunden liegt
        t_relative_s = (t_us - trigger_t_us) / 1_000_000.0
        
        # Filtern: Wir wollen nur den Bereich [-PRE_RECORD_TIME_S bis RECORD_TIME_S]
        if t_relative_s >= -PRE_RECORD_TIME_S:
            sensors[sid]['t'].append(t_relative_s)
            sensors[sid]['ay'].append(ay)
            sensors[sid]['az'].append(az)
            sensors[sid]['amag_yz'].append(amag_yz)

    num_sensors = len(sensors)
    fig, axes = plt.subplots(num_sensors, 1, figsize=(10, 3.5 * num_sensors), sharex=True)
    if num_sensors == 1:
        axes = [axes]
    
    fig.suptitle(f"SpiRob Gelenk-Ausschwingen (YZ-Magnitude)", fontsize=14)

    for idx, (sid, sdata) in enumerate(sensors.items()):
        ax = axes[idx]
        
        ax.plot(sdata['t'], sdata['ay'], label='Acc Y', linewidth=1.0, alpha=0.5)
        ax.plot(sdata['t'], sdata['az'], label='Acc Z', linewidth=1.0, alpha=0.5)
        ax.plot(sdata['t'], sdata['amag_yz'], label='Magnitude YZ', linewidth=1.5, color='black')
        
        # Markiere den Zeitpunkt des Triggers (exakt bei 0)
        ax.axvline(x=0.0, color='r', linestyle='--', label='Trigger')
        
        ax.set_title(f"Sensor ID: {sid}")
        ax.set_ylabel("Beschleunigung (g)")
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Zeit ab Trigger (s)")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    
    print(f" -> CSV & Plot unter {builds_dir} gespeichert!")

if __name__ == '__main__':
    main()