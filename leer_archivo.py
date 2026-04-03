import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os
DB_PATH = "dat" #nombre de la carpeta donde esta la info

def leer_archivo(record_name="sz01", mostrar= True):

    # Leer el registro ECG 
    ruta = os.path.join(DB_PATH, record_name)
    record = wfdb.rdrecord(ruta)
    ecg = record.p_signal[:, 0]
    fs = record.fs

    #leer latidos
    ann = wfdb.rdann(ruta, 'ari')
    r_peaks = ann.sample
    t = np.arange(len(ecg)) / fs

    record = wfdb.rdrecord(os.path.join(DB_PATH, record_name))
    ecg = record.p_signal[:, 0]   # ECG monoderivación
    fs = record.fs                # frecuencia de muestreo (200 Hz)

    # Vector de tiempo
    t = np.arange(len(ecg)) / fs

    ## obtengo tiempos del episodio
    seizure_times = load_seizure_times(os.path.join(DB_PATH, "times.seize"), record_name)

    print("Seizures encontradas:", seizure_times)
    if mostrar:

        #ECG c/ latiodos
        plt.figure(figsize=(12, 4))
        plt.plot(t, ecg, label="ECG", linewidth=1)
        plt.plot(r_peaks / fs, ecg[r_peaks],'ro',markersize=4, label="R-peaks (.ari)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud")
        plt.title("ECG con latidos detectados (archivo .ari)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
         # ECG
        plt.figure(figsize=(15, 4))
        plt.plot(t, ecg, color="blue", linewidth=0.8, label="ECG")

        # Marcar cada seizure
        for i, (start, end) in enumerate(seizure_times):
            plt.axvspan(start,end, color="red", alpha=0.3, label="Seizure" if i == 0 else None)

        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud ECG")
        plt.title(f"ECG completo – {record_name} (seizures marcadas)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        for i, (start, end) in enumerate(seizure_times):

            mask = (t >= start - int(start / 200)) & (t <= end + int(end / 200))
            
            plt.figure(figsize=(12, 4))
            plt.plot(t[mask], ecg[mask])
            plt.axvspan(start, end, color="red", alpha=0.3, label="Seizure" if i == 0 else None)
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Amplitud ECG")
            plt.title(f"{record_name} – Seizure {i+1}")
            plt.grid(True)
            plt.show()

    return ecg, t, fs, seizure_times, r_peaks

def hms_to_seconds(hms): # de hh:mm:ss a entero
    h, m, s = hms.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def load_seizure_times(filepath, record_name):
    seizures = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and parts[0] == record_name:
                start = hms_to_seconds(parts[1])
                end = hms_to_seconds(parts[2])
                seizures.append((start, end))
    return seizures






