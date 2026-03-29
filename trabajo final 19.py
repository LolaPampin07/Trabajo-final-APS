# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 10:25:50 2026

@author: lolyy
"""

# %% LIBRERIAS

import matplotlib.pyplot as plt
from os import listdir
from scipy.signal import welch, windows, sosfiltfilt, find_peaks, iirdesign, sosfreqz, lfilter, correlate
from numpy.fft import fft
from scipy.interpolate import interp1d
import math
import numpy as np
import scipy.io as sio

# --------------Variables globales------------------
fs=200 #Hz --> establecida en paper 
F_QRS=[5,15] # [Hz] rango de frecuencia complejo QRS

fs_hr = 4.0  # Hz frecuencia para muestreo uniforme de HR para aplicacion de la FFT

## Frecuencia cardiaca minima y maxima (umbral fisiologico) ##
MIN_HR=20 #latidos/min
MAX_HR=220

RR_MIN = 0.3 #[s]  (200 bpm)
rr_max = 2#[s]  (30 bpm)

PRE_ICT1 = 14
POS_ICT1= 16

# %% Lectura  + Graficos ECG

def leer_archivo(path, start, stop):
    #Lectura de archivos --> devuelve el ECG SIN FILTRAR
    start = round(start * 12000)
    stop = round(stop * 12000)
    raw = np.fromfile('data/' + path, dtype=np.int16)
    gain = 25
    baseline = 26

    ecg_mV = (raw - baseline) / gain
    
    t = np.arange(len(ecg_mV)) / fs
    
    return ecg_mV[start:stop],t[start:stop]
    

def graficar_ecg(ecg_f, ecg_mV, t, r_locs, name, tz, banda_color="#2ca02c", dpi=150):
    """
    Grafica ECG crudo, ECG filtrado y picos R; resalta la banda de tiempo del episodio.

    Parámetros
    ----------
    ecg_f : array
        ECG filtrado (mismas muestras que ecg_mV)
    ecg_mV : array
        ECG crudo (en mV)
    t : array
        Vector tiempo en segundos (puede empezar en t[0] != 0)
    r_locs : array (int)
        Índices de muestra de picos R sobre los vectores ecg_f/ecg_mV
    name : str
        Texto para el título
    tz : tuple(float, float)
        (t_lo, t_hi) en segundos para sombrear el episodio (mismas unidades que t)
    fs : float
        Frecuencia de muestreo [Hz]
    banda_color : str
        Color de la banda de episodio
    dpi : int
        DPI de la figura

    Retorna
    -------
    fig, ax : objetos de Matplotlib
    """
    # Estilo y figura
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)

    # Curvas
    ax.plot(t, ecg_mV, label='ECG crudo', alpha=0.4)
    ax.plot(t, ecg_f,  label='ECG filtrado', linewidth=1.0)

    # Picos R en el mismo origen temporal que t
    t0 = t[0]
    t_r = t0 + (np.asarray(r_locs, dtype=int) / fs)
    # Filtrar por seguridad posibles índices fuera de rango
    r_mask = (r_locs >= 0) & (r_locs < len(ecg_f))
    t_r = t0 + (np.asarray(r_locs[r_mask]) / fs)

    ax.scatter(t_r, ecg_f[r_locs[r_mask]], color='r', s=20, label='R-peaks', zorder=3)

    # Banda de episodio (en segundos)
    f_lo, f_hi = tz
    ax.axvspan(f_lo, f_hi, color=banda_color, alpha=0.15,
               label=f"Episodio {f_lo:.2f}–{f_hi:.2f} s")
    ax.axvline(f_lo, color=banda_color, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(f_hi, color=banda_color, linestyle="--", linewidth=1.2, alpha=0.8)

    # Decoración
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel('Voltaje [mV]')
    ax.set_title(f"ECG ({name})")
    ax.legend(loc='best')
    ax.grid(True)

    fig.tight_layout()
    return 