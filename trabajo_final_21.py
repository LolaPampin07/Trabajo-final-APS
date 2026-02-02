# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 18:36:22 2026

@author: lolyy
"""
# %% LIBRERIAS


import matplotlib.pyplot as plt
from os import listdir
from scipy.signal import welch, windows
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
    start = round(start * 12000)
    stop = round(stop * 12000)
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
    #plt.xlim(0,60)
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
    t_u = np.arange(t[0], t[-1], 1.0/FS_hr)   # eje de tiempo uniforme
    
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

def grafico_hr_det(t_u, hr_u, trend, name):
    
    hr_detr = hr_u - trend 
    
    
    # HR(t) cruda vs detrendida
    plt.figure(figsize=(12,4))
    plt.plot(t_u, hr_u, label='HR interpolada (bpm)', alpha=0.6)
    plt.plot(t_u, hr_detr, label='HR sin tendencia (bpm)', color='r')
    plt.plot(t_u, trend, label='Tendencia polinomial grado 4', color='k', lw=2)
    plt.xlabel('Tiempo [s]'); plt.ylabel('Latidos por minuto [#bpm]'); plt.title(f'HR - {name}')
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
    # plt.figure(figsize=(20, 10))
    # plt.plot(Ff, 10 * np.log10(PDS + 1e-20), 'x-', label='FFT')
    # plt.xlim([0, FS_hr/2])
    # plt.title("PDS [dB] ")
    # plt.xlabel('Frecuencia [Hz]')
    # plt.ylabel('PDS [dB]')
    # plt.title(f'PSD (FFT) - {name}')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    
    return Ff, PDS

# %% PSD POR WELCH
def welch_psd(x, name="PSD WELCH",
              seg_len_sec=60,
              overlap=0.5,
              detrend_already_done=True):

    n = len(x)
    if n < 4:
        raise ValueError("La señal es demasiado corta.")

    nperseg = int(seg_len_sec * FS_hr)
    nperseg = max(64, min(nperseg, n))
    win = windows.hann(nperseg, sym=False)
    noverlap = int(nperseg * overlap)

    def next_pow2(x):
        return 2 ** math.ceil(math.log2(max(1, int(x))))
    nfft = max(nperseg, next_pow2(nperseg))

 
    detrend_arg = False if detrend_already_done else 'constant'

    f, Pxx = welch(x,
                   fs=FS_hr,
                   window=win,
                   nperseg=nperseg,
                   noverlap=noverlap,
                   nfft=nfft,
                   detrend=detrend_arg,   # <- clave
                   return_onesided=True,
                   average='median',
                   scaling='density')
    return f, Pxx


def presentacion_datos(
    f_pre, pxx_pre, 
    f_post, pxx_post, 
    name="PSD HR",
    banda=(0.01, 0.1),          # banda a resaltar [Hz]
    xlim=(0,0.5),
    dpi=150
):

    eps = 1e-20  # estabilidad numérica para log10

    # Estilo base
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)

    # Curvas PSD en dB
    ax.plot(f_pre, 10*np.log10(pxx_pre + eps), 
            label="PRE ICTAL", 
            color="#1f77b4", linewidth=2.2)
    ax.plot(f_post, 10*np.log10(pxx_post + eps), 
            label="POST ICTAL", 
            color="#d62728", linewidth=2.2)

    # Sombreado de banda
    f_lo, f_hi = banda
    banda_color = "#2ca02c"
    ax.axvspan(f_lo, f_hi, color=banda_color, alpha=0.15, label=f"Banda {f_lo}–{f_hi} Hz")

    # Línea/guías opcionales en los bordes de la banda (estético)
    ax.axvline(f_lo, color=banda_color, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(f_hi, color=banda_color, linestyle="--", linewidth=1.2, alpha=0.8)

    # Etiquetas y título
    ax.set_xlabel("Frecuencia [Hz]", fontsize=12)
    ax.set_ylabel("PSD [dB]", fontsize=12)
    ax.set_title(f"PSD ritmo cardíaco – {name}", fontsize=14)

    # Límites opcionales
    if xlim is not None:
        ax.set_xlim(*xlim)


    # Leyenda y cuadricula
    ax.legend(loc="best", frameon=True)
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.8)

    # Ajuste de márgenes
    plt.tight_layout()


    plt.show()

    return 


def analizar_paciente(indice, nombre, tiempos, archivos):
    
    paciente = archivos[indice]
    ecg, t = leer_archivo(paciente, tiempos[0] , tiempos[-1])
    #graficar_archivo(ecg, t, "Paciente 1") #grafica el ECG
    latidos = detect_rpeaks(ecg,t)
    prueba_latidos(ecg,t, latidos)
    hr_d, hr_u, t_u, trend = const_RR(latidos)
    grafico_hr_det(t_u, hr_u, trend, f"HR ENTERO {nombre}")
    #Ff,PSD= transformada_rapida(hr_d, f"HR ENTERO {nombre}")

    ecg_pre,t_pre= leer_archivo(paciente, tiempos[0], tiempos[1])
    latidos_pre= detect_rpeaks(ecg_pre,t_pre)
    hr_pre, tr_pre, _ , _= const_RR(latidos_pre)
    Ff_pre,PSD_pre= transformada_rapida(hr_pre, f"PRE ICTAL {nombre}")
    f_hr_pre,pxx_pre= welch_psd(hr_pre, f"PRE ICTAL {nombre}")
    
    ecg_post, t_post= leer_archivo(paciente, tiempos[2], tiempos[3])
    latidos_post= detect_rpeaks(ecg_post,t_post)
    hr_post, tr,  _ , _ = const_RR(latidos_post)
    Ff_post,PSD_post= transformada_rapida(hr_post, f"POST ICTAL {nombre}")
    f_hr_post,pxx_post= welch_psd(hr_post, f"POST ICTAL {nombre}")
    
    
    presentacion_datos(f_hr_pre,pxx_pre,f_hr_post,pxx_post, nombre, xlim=(0, 2))
    #presentacion_datos(Ff_pre,PSD_pre,Ff_post,PSD_post, nombre, ylim=(min(PSD_pre), max(PSD_post)))

# %% main    

def main():
    
# %%
####### PACIENTES#############


    files = [file for file in listdir('data')]
    # Paciente 1  (sz01: 00:14:36 → 00:16:12)
    analizar_paciente(0, 'Paciente 1', [8.6, 12.600, 16.500, 20.500], files)
    
    # Paciente 2  (sz02_01: 01:02:43 → 01:03:43)
    analizar_paciente(1, 'Paciente 2 - Episodio 1', [56.717, 60.717, 64.217, 68.217], files)    
    analizar_paciente(1, 'Paciente 2 – Episodio 2', [169.850, 173.850, 176.767, 180.767], files)

    
    # Paciente 3  (sz03_01: 01:24:34 → 01:26:22)
    analizar_paciente(2, 'Paciente 3 - Episodio 1', [78.567, 82.567, 86.867, 90.867], files)
    analizar_paciente(2, 'Paciente 3 – Episodio 2', [148.450, 152.450, 156.783, 160.783], files)
    
    
    # Paciente 4  (sz04: 00:20:10 → 00:21:55)
    analizar_paciente(3, 'Paciente 4', [14.167, 18.167, 22.417, 26.417], files)
    
    # Paciente 5  (sz05: 00:24:07 → 00:25:30)
    analizar_paciente(4, 'Paciente 5', [18.117, 22.117, 26.000, 30.000], files)
    
    # Paciente 6  (sz06_01: 00:51:25 → 00:52:19)
    analizar_paciente(5, 'Paciente 6 - Episodio 1', [45.417, 49.417, 52.817, 56.817], files)
    analizar_paciente(5, 'Paciente 6 – Episodio 2', [118.750, 122.750, 126.667, 130.667], files)
    
    # Paciente 7  (sz07: 01:08:02 → 01:09:31)
    analizar_paciente(6, 'Paciente 7', [62.033, 66.033, 70.017, 74.017], files)
    

    

if __name__=="__main__":
    main()
