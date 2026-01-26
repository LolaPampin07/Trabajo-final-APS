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


import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# %% Variables globales

FS=200 #Hz
F1=5
F2=15
FS_hr = 4.0  # Hz frecuencia para muestreo uniforme para aplicacion de la FFT

## Frecuencia cardiaca minima y maxima (umbral fisiologico) ##
MIN_HR=30 #latidos/min
MAX_HR=220

RR_MIN = 0.3 #[s]  (200 bpm)
rr_max = 2#[s]  (30 bpm)

PRE_ICT1 = 14
POS_ICT1= 16


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
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title(f"ECG ({name})")
    plt.grid(True)
    plt.show()
    return

# %% Deteccion de latidos

def detect_rpeaks(ecg,t):
    # Filtrado para QRS
    b, a = butter(2, [F1, F2], btype='band', fs=FS)
    xf = filtfilt(b, a, ecg) #filtrado bidereccional

    # Altura mínima relativa y distancia mínima entre picos
    # Distancia mínima según HR máxima permitida
    min_dist = int(FS * 60.0 / MAX_HR) #distancia que tendrian los latidos si hr=220 lat/min

    # Altura mínima: percentil relativo
    med = np.median(xf)
    mad = np.median(np.abs(xf - med))
    k=4
    h = med + k * 1.4826 * mad   # k típico: 3–6
    peaks, _ = find_peaks(xf, distance=min_dist, height=h)

    # Filtrar picos demasiado cercanos según HR fisiológico
    rr = np.diff(peaks) / FS
    rr_ok = (rr > 60.0/MAX_HR) & (rr < 60.0/MIN_HR)
    keep = np.insert(rr_ok, 0, True)
    peaks = peaks[keep]
    
    prueba_latidos(ecg,t, peaks)

    return peaks

def prueba_latidos(ecg,t, peaks):
    #Funcion para comprobar la deteccion de latidos graficamente
    
    plt.figure(figsize=(12,4))
    plt.plot(t, ecg, label='ECG')
    plt.plot(t[peaks], ecg[peaks], 'r x', label='R-peaks')
    plt.legend(); plt.xlabel('Tiempo (s)'); plt.ylabel('mV')
    plt.title('Detección de picos R')
    plt.show()
    
    return

    
# %% Construccion temporal de RR

def const_RR (latidos):
    
    idx_r = np.asarray(latidos, dtype=float)
    t_r = latidos / FS  # tiempos de ocurrencia de cada latido (s)
    rr = np.diff(t_r)   # en segundos
    

    good = (rr >= RR_MIN) & (rr <= rr_max) #compruebo eliminar valores absurdos

    rr_clean = rr[good]
    t = t_r[1:][good]   # tiempos asociados al RR

    plt.figure(figsize=(10,4))
    plt.plot(t, rr_clean, '-o', markersize=3)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('RR (s)')
    plt.title('Serie de intervalos RR')
    plt.grid(True)
    plt.show()
    
    hr = 60/rr_clean
    

#### INTERPOLACION
    
    t0 = t[0]
    t1 = t[-1]
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
    
    # PSD no normalizada (proporcional); podés normalizar por N, FS, etc. si querés comparar entre sujetos
    PSD = (np.abs(HR)**2) / N
    
    # Recorte de banda 0.01–0.10 Hz
    f_lo, f_hi = 0.01, 0.10
    band = (freqs >= f_lo) & (freqs <= f_hi)
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
    

    # HR(t) cruda vs detrendida
    plt.figure(figsize=(12,4))
    plt.plot(t_u, hr_u, label='HR interpolada (bpm)', alpha=0.6)
    plt.plot(t_u, hr_detr, label='HR sin tendencia (bpm)', color='r')
    plt.plot(t_u, trend, label='Tendencia polinomial grado 4', color='k', lw=2)
    plt.xlabel('Tiempo (s)'); plt.ylabel('bpm'); plt.title('HR y tendencia')
    plt.legend(); plt.grid(True); plt.show()
    
    # plt.figure(figsize=(12,4))
    # plt.plot(t_u, hr_detr, label='HR sin tendencia (bpm)')
    # plt.xlabel('Tiempo [s]'); plt.ylabel('HR [latidos por minuto]'); plt.title('Ritmo cardiaco')
    # plt.axhline(0, color='k', lw=0.8)
    # plt.legend(); plt.grid(True); plt.show()
    
    
    
    # Convertir PSD a dB
    PSD_dB = 10 * np.log10(PSD)
    
    # Gráfico de la PSD en dB
    plt.figure(figsize=(10,4))
    plt.plot(freqs, PSD_dB, label='PSD(HR detrendida) [dB] - ventana rectangular')
    plt.axvspan(f_lo, f_hi, color='orange', alpha=0.2, label='0.01–0.10 Hz')
    if not np.isnan(f_peak):
        plt.axvline(f_peak, color='r', ls='--', label=f'Pico {f_peak:.3f} Hz')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Potencia (dB)')
    plt.title('Espectro HR (FFT)')
    plt.legend()
    plt.grid(True, which='both')
    plt.show()
    
    
    plt.figure(figsize=(10,4)) 
    f = np.fft.rfftfreq(len(trend), d=1/FS)
    plt.plot(freqs, np.abs(HR))
    plt.xlim(0, 0.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title('PRUEBA')

 
    return hr, t

def transformada_rapida(x):
    X = fft(x)    
    PDS = np.abs(X)**2
    
    N= len(x)
    df= df = FS / N #resolucion espectral = [[1/(s*muestras)]
    
    
    Ff=np.arange(N)*df #mi eje x en hz
    
    Ff = Ff[:N//2]
    PDS = (np.abs(X)**2) / (N * FS)
    PDS = PDS[:N//2]

    # Gráfico
    plt.figure(figsize=(20, 10))
    plt.plot(Ff, 10 * np.log10(PDS + 1e-20), 'x', label='FFT')
    #plt.xlim([0, fs/2])
    plt.title("PDS [dB] ")
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PDS [dB]')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return Ff, PDS

# %% PSD POR WELCH
def welch_psd(ecg_uniform, name="ECG (uniforme)",
              seg_len_sec=60,        # 60–120 s va bien para 0.03–0.1 Hz
              overlap=0.5,           # 50% de solapamiento
              use_median=True,       # promedio robusto
              detrend_already_done=True):
    """
    Calcula PSD con Welch usando ventana de Hann PERIÓDICA (sym=False),
    adecuada para análisis espectral con FFT.
    
    ecg_uniform: señal ya remuestreada a fs (muestras uniformes)
    fs: frecuencia de muestreo [Hz]
    """

    x = np.asarray(ecg_uniform)
    n = len(x)
    if n < 4:
        raise ValueError("La señal es demasiado corta.")

    # --- nperseg desde segundos (clave para baja frecuencia) ---
    nperseg = int(seg_len_sec * FS)
    # Asegurar límites razonables
    nperseg = max(64, min(nperseg, n))  # al menos 64 muestras y no más que n
    
    # --- ventana Hann PERIÓDICA ---
    # sym=False -> versión periódica (mejor para FFT/Welch)
    win = windows.hann(nperseg, sym=False)

    # --- solapamiento ---
    noverlap = int(nperseg * overlap)
    
    # --- nfft (potencia de 2 >= nperseg, útil para eficiencia y resolución) ---
    def _next_pow2(x):
        return 1 << int(np.ceil(np.log2(max(1, int(x)))))
    nfft = max(nperseg, _next_pow2(nperseg))

    # --- detrend ---
    # Si ya detrendaste antes (p. ej. polinomio), evitá detrendarlo de nuevo aquí
    detrend_arg = False if detrend_already_done else 'constant'

    # --- promedio ---
    avg = 'median' if use_median else 'mean'

    # --- Welch ---
    f, Pxx = welch(x,
                   fs=FS,
                   window=win,
                   nperseg=nperseg,
                   noverlap=noverlap,
                   nfft=nfft,
                   detrend=detrend_arg,
                   return_onesided=True,
                   average=avg,
                   scaling='density')  # PSD en unidades/Hz

    # --- Gráfico ---
    eps = 1e-20
    plt.figure(figsize=(14, 6))
    plt.plot(f, 10*np.log10(Pxx + eps), 'x', label=f'PSD (Welch) - {name}')
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
    files = [file for file in listdir('data')]
    paciente1 = files[0]
    ecg, t = leer_archivo(paciente1, 12 , 25)
    graficar_archivo(ecg, t, "Paciente 1")
    latidos= detect_rpeaks(ecg,t)
    hr, tr= const_RR(latidos)

    ecg_pre,t_pre= leer_archivo(paciente1, 12, 14)
    latidos= detect_rpeaks(ecg_pre,t_pre)
    hr_pre, tr_pre= const_RR(latidos)
    Ff_pre,PSD_pre= transformada_rapida(ecg_pre)
    f_ecg_pre,pxx_pre= welch_psd(ecg_pre, "Paciente 1")
    
    ecg_post, t_post= leer_archivo(paciente1, 17, 25)
    latidos= detect_rpeaks(ecg_post,t_post)
    hr, tr= const_RR(latidos)
    Ff_post,PSD_post= transformada_rapida(ecg_post)
    f_ecg_post,pxx_post= welch_psd(ecg_post, "Paciente 1")
    

if __name__=="__main__":
    main()
