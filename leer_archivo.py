# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 10:25:50 2026

@author: lolyy

REVISADO
"""

# %% LIBRERIAS

import matplotlib.pyplot as plt
import numpy as np
import variables_globales



# %% Lectura  + Graficos ECG

def abrir_archivo(path, start, stop):
    #Lectura de archivos --> devuelve el ECG SIN FILTRAR
    
    #paso el tiempo pasado por parametro de hora a segundos
    start = round(start * 12000) 
    stop = round(stop * 12000)
    
    raw = np.fromfile('data/' + path, dtype=np.int16)
    gain = 25
    baseline = 26

    ecg_mV = (raw - baseline) / gain
    
    t = np.arange(len(ecg_mV)) / variables_globales.fs
    
    return ecg_mV[start:stop],t[start:stop]
    

def graficar_ecg(ecg_f, ecg_mV, t, r_locs, name, tz, banda_color="#2ca02c",  picos= False):
    # Estilo y figura
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))

    # Curvas
    ax.plot(t, ecg_mV, label='ECG crudo', alpha=0.4)
    ax.plot(t, ecg_f,  label='ECG filtrado', linewidth=1.0)

    if picos:
    # Picos R en el mismo origen temporal que t
        t0 = t[0]
        t_r = t0 + (np.asarray(r_locs, dtype=int) / variables_globales.fs)
        # Filtrar por seguridad posibles índices fuera de rango
        r_mask = (r_locs >= 0) & (r_locs < len(ecg_f))
        t_r = t0 + (np.asarray(r_locs[r_mask]) / variables_globales.fs)
    
        ax.scatter(t_r, ecg_f[r_locs[r_mask]], color='r', s=20, label='R-peaks', zorder=3)

    # Banda de episodio (en segundos)
    f_lo, f_hi = tz
    ax.axvspan(f_lo, f_hi, color=banda_color, alpha=0.15,
               label=f"Episodio {f_lo:.2f}–{f_hi:.2f} s")
    ax.axvline(f_lo, color=banda_color, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(f_hi, color=banda_color, linestyle="--", linewidth=1.2, alpha=0.8)

    # Titulos de eje y grafico
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel('Voltaje [mV]')
    ax.set_title(f"ECG ({name})")
    ax.legend(loc='best')
    ax.grid(True)

    fig.tight_layout()
    return 