# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 10:42:10 2026

@author: lolyy

# IDEA: 
    filtrado lineal para mitigar ruido muscular y contaminacion externa
    filtrado no lineal para linea de base
    filtro adaptado (no lineal) p/ deteccion de latidos
"""

#%% librerias

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
from scipy.signal import sosfiltfilt,  iirdesign, sosfreqz
import variables_globales
import scipy.io as sio
import deteccion_picos

# %%Filtrado LINEAL ECG
def ecg_filter(ecg, t, mostrar= True):

    fs = variables_globales.fs

    # ---------- Parámetros de diseño ----------
    wp_band = np.array([1, 35])      # banda pasante [Hz]
    ws_band = np.array([0.1, 40])    # banda de rechazo [Hz]

    # Frecuencias límite (escalares, para gráficos)
    fp1, fp2 = wp_band
    fs1, fs2 = ws_band

    # Atenuaciones (forward + backward → mitad por etapa)
    alpha_p = 0.5    # ripple pasante [dB]
    alpha_s = 20     # atenuación stopband [dB]

    # ---------- Diseño IIR Butterworth ----------
    sos_butt = iirdesign(wp=wp_band, ws=ws_band, gpass=alpha_p, gstop=alpha_s, ftype='butter', fs=fs, output='sos')

    # ---------- Respuesta en frecuencia ----------
    n_fft = 4096
    w, h = sosfreqz(sos_butt, worN=n_fft, fs=fs)
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))

    # ---------- Fase y retardo de grupo ----------
    fase = np.unwrap(np.angle(h))
    w_rad = w / (fs / 2) * np.pi
    gd = -np.diff(fase) / np.diff(w_rad)

    # =================================================
    #                   GRÁFICOS
    # =================================================

    # ---------- Magnitud ----------
    plt.figure(figsize=(9, 4))
    plt.plot(w, mag_db, label='Filtro digital', color='C0')

    # Banda pasante
    plt.fill_between(
        [fp1, fp2],
        -alpha_p, 5,
        color='lightgreen',
        alpha=0.5,
        label='Banda pasante'
    )

    # Banda de rechazo baja
    plt.fill_between(
        [0, fs1],
        -100, -alpha_s,
        color='lightgreen',
        alpha=0.5
    )

    # Banda de rechazo alta
    plt.fill_between(
        [fs2, fs / 2],
        -100, -alpha_s,
        color='lightgray',
        alpha=0.5,
        label='Banda de rechazo'
    )

    # Líneas de especificación (ESCALARES)
    plt.axvline(fp1, linestyle='--', color='k')
    plt.axvline(fp2, linestyle='--', color='k')
    plt.axvline(fs1, linestyle='--', color='k')
    plt.axvline(fs2, linestyle='--', color='k')
    plt.axhline(-alpha_s, linestyle='--', color='k')

    plt.title('Filtro IIR Butterworth – ECG (Pasa Banda)')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('|H(jω)| [dB]')
    plt.xlim([0, fs / 2])
    plt.ylim([-100, 10])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------- Fase ----------
    plt.figure(figsize=(9, 4))
    plt.plot(w, fase)
    plt.title('Filtro IIR Butterworth – Fase')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Fase [rad]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------- Retardo de grupo ----------
    plt.figure(figsize=(9, 4))
    plt.plot(w[1:], gd)
    plt.title('Filtro IIR Butterworth – Retardo de Grupo')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('τg [muestras]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------- Filtrado ----------
    y = sosfiltfilt(sos_butt, ecg)
    
    if mostrar:
        plt.figure(figsize=(12, 6))
        # ECG crudo
        plt.subplot(2, 1, 1)
        plt.plot(t, ecg, color='tab:gray')
        plt.xlim(850)
        plt.title('ECG crudo')
        plt.ylabel('Amplitud [u.a.]')
        plt.grid(True)

        # ECG filtrado
        plt.subplot(2, 1, 2)
        plt.plot(t, y, color='tab:blue')
        plt.xlim(850)
        plt.title('ECG filtrado (IIR Butterworth pasa-banda)')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud [u.a.]')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    return y


# %% Filtrado no lineal

# %%---------  FILTRO DE MEDIANA --------------

# eliminacion de picos
def filt_mediana (ecg):
    filtrada = signal.medfilt(ecg, 11)
    filtrada_600 = signal.medfilt(filtrada, 31)
    
    plt.figure(figsize=(15,5))
    plt.title('Estimación')
    plt.plot(ecg, label='ECG original')
    plt.plot(filtrada, label='med_200')
    plt.plot(filtrada_600, label='Estimador b')
    plt.xlabel('Muestras')
    plt.legend()
    plt.grid(True)
    plt.show()
    

    
    plt.figure(figsize=(15, 6))
    plt.title('Filtrado')
    plt.plot(ecg, label='ECG original')
    plt.plot(ecg-filtrada_600, label='ECG filtrada')
    plt.xlabel('Muestras')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # # Subplot 1: ECG original
    # plt.subplot(2, 1, 1)
    # plt.plot(ecg, label='ECG original', color='tab:gray')
    # plt.xlim (10000,12250)
    # plt.title('ECG original')
    # plt.ylabel('Amplitud')
    # plt.grid(True)
    # plt.legend()
    
    # # Subplot 2: ECG filtrado
    # plt.subplot(2, 1, 2)
    # plt.plot(ecg - filtrada_600, label='ECG filtrada', color='tab:blue')
    # plt.title('ECG filtrada')
    # plt.xlabel('Muestras')
    # plt.ylabel('Amplitud')
    # plt.grid(True)
    # plt.legend()
    
    # plt.tight_layout()
    # plt.show()

    return

# %% ---------  SPLINES CUBICOS --------------

def splines_cubicos(ecg):

    qrs_detections = deteccion_picos.matched_filter_ecg(ecg)
    M = len(qrs_detections)

    puntos_spline = np.array(qrs_detections - 110, dtype=int).flatten()

    plt.figure(figsize=(15,5))
    plt.title('Puntos Clave')
    plt.plot(ecg, label='ECG')
    plt.plot(qrs_detections, ecg[qrs_detections.astype(int)], 'ro', label='Pico QRS')
    plt.plot(puntos_spline, ecg[puntos_spline], 'go', label='Puntos Isoeléctricos')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Muestras')
    plt.show()
    
    plt.figure(figsize=(15,5))
    plt.title('Vista en detalle')
    plt.plot(ecg, label='ECG')
    plt.plot(qrs_detections, ecg[qrs_detections.astype(int)], 'ro', label='Pico QRS')
    plt.plot(puntos_spline, ecg[puntos_spline], 'go', label='Puntos Isoeléctricos')
    plt.legend()
    plt.xlim(10000,15000)
    plt.grid(True)
    plt.xlabel('Muestras')
    plt.show()
    return qrs_detections