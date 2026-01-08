# %% librerias + imports
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def graficar_archivo(path):
    #%%Lectura de archivos
    raw = np.fromfile(path, dtype=np.int16)
    gain = 25
    baseline = 26

    ecg_mV = (raw - baseline) / gain

    fs = 200  # Hz
    t = np.arange(len(ecg_mV)) / fs

    #%%Graficos ECG

    plt.figure(figsize=(12, 4))
    plt.plot(t, ecg_mV, linewidth=1)
    plt.xlim([0, 10])
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title("ECG (decoded correctly)")
    plt.grid(True)
    plt.show()
    
    return ecg_mV


def main():
    directory_path = Path('data')
    for file_path in directory_path.iterdir():  # Para cada archivo en la carpeta "data"
        graficar_archivo(file_path)  # Graficar

if __name__=="__main__":
    main()