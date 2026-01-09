# %% librerias + imports
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal as sig
import pandas as pd
from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio

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
 


#%% Filtrado lineal
# 1) PLANTILLA DE DISEÑO - PASABANDA DIGITAL

wp = (0.8, 35)
ws = (0.1, 40)

# Atenuaciones (divididas entre 2 porque se filtra forward + backward)
alpha_p = 0.5     # dB
alpha_s = 20      # dB

# 2) DISEÑO IIR (BUTTER - CHEBY1 - CHEBY2 - CAUER)

mi_sos_butt  = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s, ftype='butter', fs=fs, output='sos')

mi_sos_cauer = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s, ftype='ellip', fs=fs, output='sos')

mi_sos_cheb1 = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s, ftype='cheby1', fs=fs, output='sos')

# mi_sos_cheb2 = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s, ftype='cheby2', fs=fs, output='sos')


# Elegimos dos para graficar
mi_sos1 = mi_sos_cauer
tipo1 = "Cauer"

mi_sos2 = mi_sos_cheb1
tipo2 = "Cheby 1"

# Respuesta en frecuenia (uso sosfreqz)
w1, h1 = sig.sosfreqz(mi_sos1, worN=2048, fs=fs)
w2, h2 = sig.sosfreqz(mi_sos2, worN=2048, fs=fs)

# Fase
fase1 = np.unwrap(np.angle(h1))
w_rad1 = w1 / (fs/2) * np.pi
gd1 = -np.diff(fase1) / np.diff(w_rad1)

fase2 = np.unwrap(np.angle(h2))
w_rad2 = w2 / (fs/2) * np.pi
gd2 = -np.diff(fase2) / np.diff(w_rad2)

# %% GRAFICOS 
# CAUER
plt.figure(figsize=(12,10))

plt.subplot(3,1,1)
plt.plot(w1, 20*np.log10(abs(h1)))
plot_plantilla('bandpass', wp, alpha_p*2, ws, alpha_s*2, fs)
plt.title(f'IIR {tipo1} - Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(ω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

plt.subplot(3,1,2)
plt.plot(w1, fase1)
plt.title(f'IIR {tipo1} - Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid(True, which='both', ls=':')
plt.legend()

plt.subplot(3,1,3)
plt.plot(w1[1:], gd1)
plt.title(f'IIR {tipo1} - Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')
plt.legend()

# CHEBY 1
plt.figure(figsize=(12,10))

plt.subplot(3,1,1)
plt.plot(w2, 20*np.log10(abs(h2)))
plot_plantilla('bandpass', wp, alpha_p*2, ws, alpha_s*2, fs)
plt.title(f'IIR {tipo2} - Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(ω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

plt.subplot(3,1,2)
plt.plot(w2, fase2)
plt.title(f'IIR {tipo2} - Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid(True, which='both', ls=':')
plt.legend()

plt.subplot(3,1,3)
plt.plot(w2[1:], gd2)
plt.title(f'IIR {tipo2} - Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')
plt.legend()

# %% 2) DISEÑO FIR (FIRWIN2 y FIRLS)

frecuencias = np.array([0, 0.1, 0.8, 35, 35.7, fs/2])
deseado     = np.array([0, 0, 1, 1, 0, 0])

# FIR con ventana rectangular
numtaps = 1001
fir_win = sig.firwin2(numtaps, frecuencias, deseado, fs=fs, window='boxcar')
retardo = (numtaps - 1)//2

# FIR Least-Squares
numtaps_ls = 2001  # impar
fir_ls = sig.firls(numtaps_ls, frecuencias, deseado, fs=fs)
retardo_ls = (numtaps_ls - 1)//2

# Respuesta en frecuencia
w_fir, h_fir = sig.freqz(fir_win, worN=2048, fs=fs)
fase_fir = np.unwrap(np.angle(h_fir))
gd_fir = -np.diff(fase_fir) / np.diff(w_fir/fs*np.pi)

w_ls, h_ls = sig.freqz(fir_ls, worN=2048, fs=fs)
fase_ls = np.unwrap(np.angle(h_ls))
gd_ls = -np.diff(fase_ls) / np.diff(w_ls/fs*np.pi)

# %% GRAFICOS 
# VENTANA RECTANGULAR
plt.figure(figsize=(12,10))

plt.subplot(3,1,1)
plt.plot(w_fir, 20*np.log10(abs(h_fir)))
plot_plantilla('bandpass', wp, alpha_p*2, ws, alpha_s*2, fs)
plt.title('FIR ventana rectangular - Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(ω)| [dB]')
plt.xlim(0, 500)
plt.grid(True, which='both', ls=':')
plt.legend()

plt.subplot(3,1,2)
plt.plot(w_fir, fase_fir)
plt.title('FIR ventana rectangular - Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.xlim(0, 500)
plt.grid(True, which='both', ls=':')
plt.legend()

plt.subplot(3,1,3)
plt.plot(w_fir[1:], gd_fir)
plt.title('FIR ventana rectangular - Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.xlim(0, 65)
plt.grid(True, which='both', ls=':')
plt.legend()

# CUADRADOS MINIMOS (LEAST - SQUARES)
plt.figure(figsize=(12,10))

plt.subplot(3,1,1)
plt.plot(w_ls, 20*np.log10(abs(h_ls)))
plot_plantilla('bandpass', wp, alpha_p*2, ws, alpha_s*2, fs)
plt.title('FIR cuadrados minimos - Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(ω)| [dB]')
plt.xlim(0, 500)
plt.grid(True, which='both', ls=':')
plt.legend()

plt.subplot(3,1,2)
plt.plot(w_ls, fase_ls)
plt.title('FIR cuadrados minimos - Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.xlim(0, 500)
plt.grid(True, which='both', ls=':')
plt.legend()

plt.subplot(3,1,3)
plt.plot(w_ls[1:], gd_ls)
plt.title('FIR cuadrados minimos - Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.xlim(0, 65)
plt.grid(True, which='both', ls=':')
plt.legend()


# %% Filtrado no lineal

# %% main

def main():
    directory_path = Path('data')
    for file_path in directory_path.iterdir():  # Para cada archivo en la carpeta "data"
        graficar_archivo(file_path, file_path.name)  # Graficar

if __name__=="__main__":
    main()
    