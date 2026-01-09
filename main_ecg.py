# %% librerias + imports
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import pandas as pd

fs = 200  # Hz

# %% Lectura  + Graficos ECG sin filtrar


def graficar_archivo(path, name):
    #Lectura de archivos
    raw = np.fromfile(path, dtype=np.int16)
    gain = 25
    baseline = 26

    ecg_mV = (raw - baseline) / gain
    
    t = np.arange(len(ecg_mV)) / fs

#Graficos ECG

    plt.figure(figsize=(12, 4))
    plt.plot(t, ecg_mV, linewidth=1)
    #plt.xlim([0, 10])
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title(f"ECG ({name})")
    plt.grid(True)
    plt.show()
    
    return ecg_mV



# %% Correlacion
# 👉 un gran pico en lag = 0
#👉 picos secundarios en lag = RR
#👉 repetición de estos picos cada múltiplo del RR


# =========================
#  Funciones de análisis
# =========================

def _autocorrelacion_normalizada(path, name):
    """
    Calcula la autocorrelación unilateral (lags >= 0),
    con normalización no sesgada y acf[0] = 1.
    """
    #Lectura de archivos
    raw = np.fromfile(path, dtype=np.int16)
    gain = 25
    baseline = 26

    ecg_mV = (raw - baseline) / gain
    
    x = np.asarray(ecg_mV).astype(float)
    x = x - np.mean(x)  # quitar componente DC
    N = len(x)
    acf_full = signal.correlate(x, x, method="direct")
    acf = acf_full[N-1:]  # lags no-negativos

    # Normalización "unbiased": dividir por el número de términos en cada lag
    norm = np.arange(N, 0, -1)
    acf_unbiased = acf / norm

    # Normalizar para que acf[0] = 1
    acf_norm = acf_unbiased / acf_unbiased[0] if acf_unbiased[0] != 0 else acf_unbiased
    
    plt.figure(figsize=(12, 4))
    plt.plot( acf_norm, linewidth=1)
    #plt.xlim([0, 10])
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title(f"Autocorrelacion normalizada ({name})")
    plt.grid(True)
    plt.show()
    
    
    return acf_norm



#%% Filtrado lineal

# %% Filtrado no lineal

# %% main

def main():
    directory_path = Path('data')
    for file_path in directory_path.iterdir():  # Para cada archivo en la carpeta "data"
        graficar_archivo(file_path, file_path.name)  # Graficar
        _autocorrelacion_normalizada(file_path, file_path.name)

if __name__=="__main__":
    main()
    


    
    