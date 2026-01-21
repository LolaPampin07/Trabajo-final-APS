# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 18:36:22 2026

@author: lolyy
"""
# %% LIBRERIAS + DECLARACIONES


import matplotlib.pyplot as plt
from os import listdir
from scipy import signal as sig
from scipy.signal import periodogram , get_window
import pandas as pd
import scipy.io as sio
import wave
from numpy.fft import fft


import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

fs=200 #Hz
f1=5
f2=15

rr_min = 0.3 #[s]  (200 bpm)
rr_max = 2#[s]  (30 bpm)

# %% Lectura  + Graficos ECG sin filtrar

def leer_archivo(path, start, stop):
    #Lectura de archivos
    start *= 12000
    stop *= 12000
    raw = np.fromfile('data/' + path, dtype=np.int16)
    gain = 25
    baseline = 26

    ecg_mV = (raw - baseline) / gain
    
    t = np.arange(len(ecg_mV)) / fs
    
    return ecg_mV[start:stop],t[start:stop]
    

def graficar_archivo(ecg_mV, t, name):
    #Graficos ECG
    
    plt.figure(figsize=(12, 4))
    plt.plot(t, ecg_mV, linewidth=1)
    #plt.xlim([0, 10])
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title(f"ECG ({name})")
    plt.grid(True)
    plt.show()
    
    return

# %% Deteccion de latidos

def detect_rpeaks(ecg, min_hr=30, max_hr=220):
    # Filtrado para QRS
    b, a = butter(2, [f1, f2], btype='band', fs=fs)
    xf = filtfilt(b, a, ecg) #filtrado bidereccional

    # Altura mínima relativa y distancia mínima entre picos
    # Distancia mínima según HR máxima permitida
    min_dist = int(fs * 60.0 / max_hr)

    # Altura mínima: percentil relativo (ajustalo según tu señal)
    h = np.percentile(xf, 99) * 0.30

    peaks, _ = find_peaks(xf, distance=min_dist, height=h)

    # Filtrar picos demasiado cercanos según HR fisiológico
    # (opcional, pero evita falsos positivos)
    rr = np.diff(peaks) / fs
    rr_ok = (rr > 60.0/max_hr) & (rr < 60.0/min_hr)
    keep = np.insert(rr_ok, 0, True)
    peaks = peaks[keep]

    return peaks

def prueba_grafico(ecg,t):
    """
    Funcion para comprobar la deteccion de latidos graficamente
    """
    plt.figure(figsize=(12,4))
    peaks= detect_rpeaks(ecg)
    plt.plot(t, ecg, label='ECG')
    plt.plot(t[peaks], ecg[peaks], 'r x', label='R-peaks')
    plt.legend(); plt.xlabel('Tiempo (s)'); plt.ylabel('mV')
    plt.title('Detección de picos R')
    plt.show()
    
    return peaks
    
# %% Construccion temporal de RR

def const_RR (idx_r):
    
    idx_r = np.asarray(idx_r, dtype=float)
    t_r = idx_r / fs  # tiempos de ocurrencia de cada latido (s)
    rr = np.diff(t_r)   # en segundos

    good = (rr >= rr_min) & (rr <= rr_max) #compruebo eliminar valores absurdos

    rr_clean = rr[good]
    t_r_clean = t_r[1:][good]   # tiempos asociados al RR

    plt.figure(figsize=(10,4))
    plt.plot(t_r_clean, rr_clean, '-o', markersize=3)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('RR (s)')
    plt.title('Serie de intervalos RR')
    plt.grid(True)
    plt.show()
    
    hr = 60/rr_clean
    
    return hr, t_r_clean

# %% main    

def main():
    files = [file for file in listdir('data')]
    paciente1 = files[0]
    ecg, t = leer_archivo(paciente1, 14, 17)
    graficar_archivo(ecg, t, "Paciente 1")
    latidos= prueba_grafico(ecg,t)
    hr, tr= const_RR(latidos)
    
    

if __name__=="__main__":
    main()
