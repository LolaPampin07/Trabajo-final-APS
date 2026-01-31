# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 18:36:22 2026

@author: lolyy
"""
# %% LIBRERIAS


import matplotlib.pyplot as plt
from os import listdir
from scipy import signal as sig
from scipy.signal import welch, windows
import pandas as pd
import scipy.io as sio
import wave
from numpy.fft import fft
from scipy.interpolate import interp1d
import math

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# %% Variables globales

FS=200 #Hz --> establecida en paper 
F_QRS=[5,15] # [Hz] rango de frecuencia complejo QRS

FS_hr = 4.0  # Hz frecuencia para muestreo uniforme de HR para aplicacion de la FFT

## Frecuencia cardiaca minima y maxima (umbral fisiologico) ##
MIN_HR=30 #latidos/min
MAX_HR=220

RR_MIN = 0.3 #[s]  (200 bpm)
rr_max = 2#[s]  (30 bpm)

PRE_ICT1 = 14
POS_ICT1= 16


"""
####Tiempos de episodios segun pacientes####
sz01 = ("00:14:36", "00:16:12")
sz02_01 = ("01:02:43", "01:03:43")
sz02_02 = ("02:55:51", "02:56:16")  
sz03_01 = ("01:24:34", "01:26:22")
sz03_02 = ("02:34:27", "02:36:17")  
sz04 = ("00:20:10", "00:21:55")
sz05 = ("00:24:07", "00:25:30")
sz06_01 = ("00:51:25", "00:52:19")
sz06_02 = ("02:04:45", "02:06:10") 
sz07 = ("01:08:02", "01:09:31")
"""


# %% Lectura  + Graficos ECG sin filtrar

def leer_archivo(path, start, stop):
    #Lectura de archivos
    start *= 12000
    stop *= 12000
    raw = np.fromfile('data/' + path, dtype=np.int16)
    gain = 25
    baseline = 26

    ecg_mV = (raw - baseline) / gain
    
    t = np.arange(len(ecg_mV)) / FS
    
    return ecg_mV[start:stop],t[start:stop]
    

def graficar_archivo(ecg_mV, t, name):
    #Graficos ECG
    plt.figure(figsize=(12, 4))
    plt.plot(t, ecg_mV, linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [mV]")
    plt.title(f"ECG ({name})")
    plt.grid(True)
    plt.show()
    return

# %% Deteccion de latidos

def detect_rpeaks(ecg,t):
    # Filtrado para QRS
    b, a = butter(2, F_QRS, btype='band', fs=FS) # filtro digital tipo butter pasabanda en rango de frecuencia del complejo QRS de orden 2
    xf = filtfilt(b, a, ecg) #filtrado bidereccional

    # Altura mínima relativa y distancia mínima entre picos
    # Distancia mínima según HR máxima permitida
    min_dist = int(FS * 60.0 / MAX_HR) #distancia que tendrian los latidos si hr=220 lat/min

    # Altura mínima: percentil relativo --> umbral adaptativo
    med = np.median(xf) #promedio robusto
    mad = np.median(np.abs(xf - med)) # MAD = Median Absolute Deviation --> a cada elemento le resto la media y calculo la media del modulo de resta
    k=4 # define un umbral minimo de altura
    
    h = med + k * 1.4826 * mad   # 1.4826: factor que convierte la MAD en una desviacion estandar (distribucion gaussiana)
    #mediana + 4 veces el estimador de desviacion estandar
    
    peaks, _ = find_peaks(xf, distance=min_dist, height=h)

    # Filtrar picos demasiado cercanos según HR fisiológico
    rr = np.diff(peaks) / FS
    rr_ok = (rr > 60.0/MAX_HR) & (rr < 60.0/MIN_HR)
    keep = np.insert(rr_ok, 0, True)
    peaks = peaks[keep]

    return peaks

def prueba_latidos(ecg,t, peaks):
    #Funcion para comprobar la deteccion de latidos graficamente
    
    plt.figure(figsize=(12,4))
    plt.plot(t, ecg, label='ECG')
    plt.plot(t[peaks], ecg[peaks], 'r x', label='R-peaks')
    plt.legend(); plt.xlabel('Tiempo [s]'); plt.ylabel('Voltaje [mV]')
    plt.title('Detección de picos R')
    plt.show()
    
    return

    
# %% Construccion temporal de RR

def const_RR (latidos):
    
    #latidos = np.asarray(latidos, dtype=float)
    t_r = latidos / FS  # tiempos de ocurrencia de cada latido (s)
    rr = np.diff(t_r)   # en segundos --> np.diff resta el actual con el anterior == armo los intervalor RR
    

    good = (rr >= RR_MIN) & (rr <= rr_max) #compruebo eliminar valores absurdos

    rr_clean = rr[good]
    t = t_r[1:][good]   # tiempos asociados a cada RR
    
    hr = 60/rr_clean # frecuencia cardiaca  

#### INTERPOLACION
    
    t0 = t[0]
    t1 = t[-1] # ultimo elemento del array
    t_u = np.arange(t0, t1, 1.0/FS_hr)   # eje de tiempo uniforme
    
    # Interpolación lineal (suele ser suficiente para HRV lenta)
    f = interp1d(t, hr, kind='linear', fill_value='extrapolate', bounds_error=False)
    hr_u = f(t_u)   # HR(t) uniforme, en bpm
    
    # Ajuste polinomial de grado 4 (paper) sobre HR_u vs t_u
    p = np.polyfit(t_u, hr_u, deg=4)
    trend = np.polyval(p, t_u)
    
    hr_detr = hr_u - trend   # señal de HR sin tendencia (bpm)

    N = len(hr_detr)
    
    # Ventana rectangular (implícita); FFT de una sola cara
    HR = np.fft.rfft(hr_detr, n=N)
    freqs = np.fft.rfftfreq(N, d=1.0/FS_hr)
    
    # PSD no normalizada (proporcional)
    PSD = (np.abs(HR)**2) / N
    
    # Recorte de banda 0.01–0.10 Hz
    f_lo, f_hi = 0.01, 0.10
    band = (freqs >= f_lo) & (freqs <= f_hi) #array de booleanos
    freq_band = freqs[band]
    psd_band  = PSD[band]
    
    # Pico espectral dentro de 0.01–0.10 Hz
    if np.any(band):
        kmax = np.argmax(psd_band)
        f_peak = freq_band[kmax]
        p_peak = psd_band[kmax]
    else:
        f_peak = np.nan
        p_peak = np.nan
    
    print(f"Pico en banda 0.01–0.10 Hz: {f_peak:.4f} Hz (potencia relativa={p_peak:.3g})")
    return hr_detr, hr_u, t_u, trend

def grafico_hr_det(t_u, hr_u, trend):
    
    hr_detr = hr_u - trend 
    
    
    # HR(t) cruda vs detrendida
    plt.figure(figsize=(12,4))
    plt.plot(t_u, hr_u, label='HR interpolada (bpm)', alpha=0.6)
    plt.plot(t_u, hr_detr, label='HR sin tendencia (bpm)', color='r')
    plt.plot(t_u, trend, label='Tendencia polinomial grado 4', color='k', lw=2)
    plt.xlabel('Tiempo [s]'); plt.ylabel('Latidos por minuto [#bpm]'); plt.title('HR y tendencia')
    plt.legend(); plt.grid(True); plt.show()  
    return
def transformada_rapida(x, name):
    X = fft(x)    
    PDS = np.abs(X)**2
    
    N= len(x)
    df= df = FS_hr / N #resolucion espectral = [[1/(s*muestras)]
    
    
    Ff=np.arange(N)*df #mi eje x en hz
    
    Ff = Ff[:N//2]
    PDS = (np.abs(X)**2) / (N * FS_hr)
    PDS = PDS[:N//2]

    # Gráfico
    plt.figure(figsize=(20, 10))
    plt.plot(Ff, 10 * np.log10(PDS + 1e-20), 'x-', label='FFT')
    plt.xlim([0, FS_hr/2])
    plt.title("PDS [dB] ")
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PDS [dB]')
    plt.title(f'PSD (FFT) - {name}')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return Ff, PDS

# %% PSD POR WELCH
def welch_psd(x, name="PSD WELCH",
              seg_len_sec=60,        # 60–120 s va bien para 0.03–0.1 Hz
              overlap=0.5,           # 50% de solapamiento
              detrend_already_done=True):
    """
    Calcula PSD con Welch usando ventana de Hann PERIÓDICA (sym=False),
    adecuada para análisis espectral con FFT.
    
    ecg_uniform: señal ya remuestreada a fs (muestras uniformes)
    fs: frecuencia de muestreo [Hz]
    """
    
    n = len(x)
    
    #control de error
    if n < 4:
        raise ValueError("La señal es demasiado corta.")
        

    # --- nperseg desde segundos (clave para baja frecuencia) ---
    nperseg = int(seg_len_sec * FS_hr)
    # Asegurar límites razonables
    nperseg = max(64, min(nperseg, n))  # al menos 64 muestras y no más que n
    
    # --- ventana Hann PERIÓDICA ---
    win = windows.hann(nperseg, sym=False)  # sym=False -> versión periódica (mejor para FFT/Welch)

    # --- solapamiento ---
    noverlap = int(nperseg * overlap)
    
    # --- nfft (potencia de 2 >= nperseg, útil para eficiencia y resolución) ---
    

    def next_pow2(x):
        return 2 ** math.ceil(math.log2(max(1, int(x))))
    nfft = max(nperseg, next_pow2(nperseg)) #si nperseg no es potencia de 2, entonces nfft se sube a la siguiente potencia de 2.

    # --- Welch ---
    f, Pxx = welch(x,#senal
                   fs=FS_hr, #frecuencia de muestreo
                   window=win, 
                   nperseg=nperseg,
                   noverlap=noverlap,
                   nfft=nfft,
                   return_onesided=True,
                   average= 'mean', #promedio robusto
                   scaling='density')  # PSD en unidades/Hz

    # --- Gráfico ---
    eps = 1e-20
    plt.figure(figsize=(14, 6))
    plt.plot(f, 10*np.log10(Pxx + eps), label=f'PSD (Welch) - {name}')
    plt.xlim(0,1)
    plt.title(f'PSD (Welch) - {name}')
    plt.ylabel('Densidad Espectral de Potencia [dB/Hz]')
    plt.xlabel('Frecuencia [Hz]')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()

    return f, Pxx



# %% main    

def main():
    
####### PACIENTE 1#############
    

    files = [file for file in listdir('data')]
    paciente1 = files[0]
    ecg, t = leer_archivo(paciente1, 12 , 25)
    #graficar_archivo(ecg, t, "Paciente 1") #grafica el ECG
    latidos = detect_rpeaks(ecg,t)
    prueba_latidos(ecg,t, latidos)
    hr_d, hr_u, t_u, trend = const_RR(latidos)
    grafico_hr_det(t_u, hr_u, trend)
    Ff,PSD= transformada_rapida(hr_d, "HR ENTERO PAC 1")

    ecg_pre,t_pre= leer_archivo(paciente1, 12, 14)
    latidos_pre_01= detect_rpeaks(ecg_pre,t_pre)
    hr_pre_01, tr_pre_01, _ , _= const_RR(latidos_pre_01)
    Ff_pre,PSD_pre= transformada_rapida(hr_pre_01, "PRE ICTAL paciente 1")
    f_hr_pre_01,pxx_pre_01= welch_psd(hr_pre_01, "PRE ICTAL paciente 1")
    
    ecg_post, t_post= leer_archivo(paciente1, 17, 25)
    latidos_post_01= detect_rpeaks(ecg_post,t_post)
    hr_post_01, tr_01,  _ , _ = const_RR(latidos_post_01)
    Ff_post_01,PSD_post_01= transformada_rapida(hr_post_01, "POST ICTAL paciente 1")
    f_hr_post_01,pxx_post_01= welch_psd(hr_post_01, "POST ICTAL paciente 1")
    
    
####### PACIENTE 7#############
  

    #sz07 = ("01:08:02", "01:09:31")
    files = [file for file in listdir('data')]
    paciente7 = files[1]
    ecg_07, t = leer_archivo(paciente7, 66 , 72)
    #graficar_archivo(ecg, t, "Paciente 1") #grafica el ECG
    latidos_07 = detect_rpeaks(ecg_07,t)
    prueba_latidos(ecg_07,t, latidos_07)
    hr_07, hr_u_07, t_u_07, trend_07 = const_RR(latidos_07)
    grafico_hr_det(t_u_07, hr_u_07, trend_07)
    Ff,PSD= transformada_rapida(hr_d, "HR ENTERO PAC 7")


    ecg_pre_07,t_pre_07= leer_archivo(paciente7, 66, 68)
    latidos_pre_07= detect_rpeaks(ecg_pre_07,t_pre_07)
    hr_pre_07, tr_pre_07, _ , _= const_RR(latidos_pre_07)
    Ff_pre_07,PSD_pre_07= transformada_rapida(hr_pre_07, "PRE ICTAL paciente 7")
    f_hr_pre_07,pxx_pre_07= welch_psd(hr_pre_07, "PRE ICTAL paciente 7")
    
    ecg_post_07, t_post_07= leer_archivo(paciente1, 70, 72)
    latidos_post_07= detect_rpeaks(ecg_post_07,t_post_07)
    hr_post_07, tr_07,  _ , _ = const_RR(latidos_post_07)
    Ff_post_07,PSD_post_07= transformada_rapida(hr_post_07, "POST ICTAL paciente 7")
    f_hr_post_07,pxx_post_07= welch_psd(hr_post_07, "POST ICTAL paciente 7")
    

if __name__=="__main__":
    main()
