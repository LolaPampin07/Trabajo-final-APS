# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 18:36:22 2026

@author: lolyy
"""
# %% LIBRERIAS


import matplotlib.pyplot as plt
from os import listdir
from scipy.signal import welch, windows, sosfiltfilt, iirnotch, butter, filtfilt, find_peaks
from numpy.fft import fft
from scipy.interpolate import interp1d
import math
import numpy as np



# %% Variables globales

fs=200 #Hz --> establecida en paper 
F_QRS=[5,15] # [Hz] rango de frecuencia complejo QRS

FS_hr = 4.0  # Hz frecuencia para muestreo uniforme de HR para aplicacion de la FFT

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

# %%Filtrado ECG
def ecg_filter(ecg, fs, low=8.0, high=20.0, order=4, q_notch=30):
    #FILTRO BUTTER PASA BANDA Y NOTCH RECHAZA BANDA + FILTRADO BIDERECCIONAL
    ecg = np.asarray(ecg, dtype=np.float64).ravel() #CONVERSION A ARRAY PLANO
    # Pasabanda Butterworth en Hz
    sos = butter(order, [low, high], btype='bandpass', output='sos', fs=fs) #FILTRO BUTTER TIPO PASABANDA DE ORDEN 4 FILTRADO EN LA FRECUENCIA DEL QRS
    y = sosfiltfilt(sos, ecg)

    #filtro rechazanbanda
    b, a = iirnotch(w0=50, Q=q_notch, fs=fs) #FILTRO IIR RECHAZABANDA LOS 50HZ DE ALTERNA
    y = filtfilt(b, a, y)
    
    return y 
# %% Deteccion de latidos

def detect_rpeaks(ecg, fs, band=(8,20), win_ms=120, refractory_ms=200, prominence=None):
    """
    Devuelve índices de picos R en la señal original.
    """
    # 1) Filtrado para resaltar QRS atenuar: bajas frec(deriva de línea base, respiración, onda P) y altas frec (ruido muscular/EMG, zumbido residual).
    ecg_f = ecg_filter(ecg, fs, low=band[0], high=band[1], order=4)

    # 2) Realce tipo Pan-Tompkins: derivada + cuadrado + ventana móvil
    d = np.diff(ecg_f, prepend=ecg_f[0]) #derivda discreta
    y = d**2                            #energia de la pendiente
    win = max(1, int(round(win_ms * fs / 1000.0)))#ventana de muestras
    kernel = np.ones(win) / win
    # Convolución con misma longitud
    y_int = np.convolve(y, kernel, mode='same') #promedio movil

    # 3) Umbral adaptativo simple
    # Arranque con mediana (robusta). Ajustá factor según SNR de tu señal.
    thr = 0.5 * np.median(y_int[y_int > 0])  # evita ceros
    if prominence is None:
        # Prominence mínima relativa
        prom = 0.5 * thr
    else:
        prom = prominence

    # 4) Búsqueda de picos en y_int (no directamente en ecg)
    # refractory = 200 ms (para adulto típico). Ajustable.
    distance = int(round(refractory_ms * fs / 1000.0))
    peaks, props = find_peaks(y_int, height=thr, prominence=prom, distance=distance)

    # 5) Refino posición del pico: mover al máximo local en ecg_f cerca del pico
    # Esto te alinea al R verdadero (máximo en ventana ±80 ms aprox)
    r_locs = []
    radius = max(1, int(round(0.08 * fs)))  # ±80 ms
    n = len(ecg_f)
    for p in peaks:
        i0 = max(0, p - radius)
        i1 = min(n, p + radius + 1)
        if i1 > i0:
            local = np.argmax(ecg_f[i0:i1]) + i0
            r_locs.append(local)
    r_locs = np.array(sorted(set(r_locs)))  # únicos y ordenados

    return r_locs, ecg_f, y_int, thr


# def detect_rpeaks(ecg,t):

#     # Altura mínima relativa y distancia mínima entre picos
#     # Distancia mínima según HR máxima permitida
#     min_dist = int(FS * 60.0 / MAX_HR) #distancia que tendrian los latidos si hr=220 lat/min

#     # Altura mínima: percentil relativo --> umbral adaptativo
#     med = np.median(ecg) #promedio robusto
#     mad = np.median(np.abs(ecg - med)) # MAD = Median Absolute Deviation --> a cada elemento le resto la media y calculo la media del modulo de resta
#     k=4 # define un umbral minimo de altura
    
#     h = med + k * 1.4826 * mad   # 1.4826: factor que convierte la MAD en una desviacion estandar (distribucion gaussiana)
#     #mediana + 4 veces el estimador de desviacion estandar
    
#     peaks, _ = find_peaks(ecg, distance=min_dist, height=h)

#     # Filtrar picos demasiado cercanos según HR fisiológico
#     rr = np.diff(peaks) / FS
#     rr_ok = (rr > 60.0/MAX_HR) & (rr < 60.0/MIN_HR)
#     keep = np.insert(rr_ok, 0, True)
#     peaks = peaks[keep]
    

#     return peaks



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

    
# %% Construccion temporal de 


def const_RR (ecg, r_locs, fs=FS, fs_hr=4.0, hr_bounds=(40, 180), detrend_deg=None):
    """
    A partir de picos R (índices), devuelve:
      - t_hr: tiempos de cada HR instantánea (s)
      - hr: HR instantánea (bpm)
      - t_u, hr_u: HR uniformizada a fs_hr (opcionalmente con detrend si se pide)
      - (trend si detrend_deg no es None)
    """
    r_locs = np.asarray(r_locs).astype(int)
    r_locs = r_locs[(r_locs > 0) & (r_locs < len(ecg))]  # seguridad
    if len(r_locs) < 2:
        raise ValueError("No hay suficientes latidos para calcular RR/HR.")

    # 1) RR en segundos
    rr = np.diff(r_locs) / fs  # s
    
    # 2) Tiempo asociado a cada RR/HR: usar el tiempo del segundo R (o el medio)
    # opción más estable para time-stamp: el punto medio entre R_i y R_i+1
    r_times = r_locs / fs
    t_hr = 0.5 * (r_times[1:] + r_times[:-1])

    # 3) HR instantánea
    hr = 60.0 / rr  # bpm

    # 4) Outliers en HR
    lo, hi = hr_bounds
    mask = (hr > lo) & (hr < hi)
    t_hr = t_hr[mask]
    hr = hr[mask]

    # Si querés devolver sólo HR por latido:
    results = {
        "t_hr": t_hr,
        "hr": hr,
    }

    # 5) Serie HR uniforme (para graficar suave o HRV)
    if len(t_hr) >= 2:
        t0, tf = t_hr[0], t_hr[-1]
        n = int(np.floor((tf - t0) * fs_hr)) + 1
        n = max(n, 2)
        t_u = np.linspace(t0, tf, n)

        f = interp1d(t_hr, hr, kind='linear', fill_value='extrapolate', bounds_error=False)
        hr_u = f(t_u)

        if detrend_deg is not None:
            # Ajuste polinomial sobre puntos válidos
            deg = int(detrend_deg)
            use_t, use_hr = t_u, hr_u
            if len(t_u) >= deg + 1:
                p = np.polyfit(use_t, use_hr, deg=deg)
                trend = np.polyval(p, t_u)
                hr_detr = hr_u - trend
                results.update({"t_u": t_u, "hr_u": hr_u, "trend": trend, "hr_detr": hr_detr})
            else:
                results.update({"t_u": t_u, "hr_u": hr_u})
        else:
            results.update({"t_u": t_u, "hr_u": hr_u})
    return results

# def const_RR (latidos):
    
#     #latidos = np.asarray(latidos, dtype=float)
#     t_r = latidos / FS  # tiempos de ocurrencia de cada latido (s)
#     rr = np.diff(t_r)   # en segundos --> np.diff resta el actual con el anterior == armo los intervalor RR
    

#     good = (rr >= RR_MIN) & (rr <= rr_max) #compruebo eliminar valores absurdos

#     rr_clean = rr[good]
#     t = t_r[1:][good]   # tiempos asociados a cada RR
    
#     hr = 60/rr_clean # frecuencia cardiaca  
    

# #### INTERPOLACION
#     t_u = np.arange(t[0], t[-1], 1.0/FS_hr)   # eje de tiempo uniforme
    
#     # Interpolación lineal (suele ser suficiente para HRV lenta)
#     f = interp1d(t, hr, kind='linear', fill_value='extrapolate', bounds_error=False)
#     hr_u = f(t_u)   # HR(t) uniforme, en bpm
    
#     # Ajuste polinomial de grado 4 (paper) sobre HR_u vs t_u
#     p = np.polyfit(t_u, hr_u, deg=4)
#     trend = np.polyval(p, t_u)
    
#        # señal de HR sin tendencia (bpm)
    
#     t_u_good = []
#     hr_u_good = []
#     trend_good = []
    
#     for ti, hri, trendi in zip(t_u, hr_u, trend):
#         if 50 < hri < 100:
#             t_u_good.append(ti)
#             hr_u_good.append(hri)
#             trend_good.append(trendi)
    
#     t_u_good = np.array(t_u_good)
#     hr_u_good = np.array(hr_u_good)
    
#     plt.figure()
#     plt.plot(t_u_good,hr_u_good)

#     hr_detr = hr_u_good - trend_good
#     N = len(hr_detr)
    
#     # Ventana rectangular (implícita); FFT de una sola cara
#     HR = np.fft.rfft(hr_detr, n=N)
#     freqs = np.fft.rfftfreq(N, d=1.0/FS_hr)
    
#     # PSD no normalizada (proporcional)
#     PSD = (np.abs(HR)**2) / N
    
#     # Recorte de banda 0.01–0.10 Hz
#     f_lo, f_hi = 0.01, 0.10
#     band = (freqs >= f_lo) & (freqs <= f_hi) #array de booleanos
#     freq_band = freqs[band]
#     psd_band  = PSD[band]
    
#     # Pico espectral dentro de 0.01–0.10 Hz
#     if np.any(band):
#         kmax = np.argmax(psd_band)
#         f_peak = freq_band[kmax]
#         p_peak = psd_band[kmax]
#     else:
#         f_peak = np.nan
#         p_peak = np.nan
    
#     print(f"Pico en banda 0.01–0.10 Hz: {f_peak:.4f} Hz (potencia relativa={p_peak:.3g})")
#     return hr_detr, hr_u_good, t_u_good, trend_good

def grafico_hr_det(t_u, hr_u, trend, name):
    
    #hr_detr = hr_u - trend 
    
    
    # HR(t) cruda vs detrendida
    plt.figure(figsize=(12,4))
    plt.plot(t_u, hr_u, label='HR interpolada (bpm)', alpha=0.6)
    #plt.plot(t_u, hr_detr, label='HR sin tendencia (bpm)', color='r')
    plt.plot(t_u, trend, label='Tendencia polinomial grado 4', color='k', lw=2)
    plt.ylim(30, 120)
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
    latidos = detect_rpeaks(ecg)
    prueba_latidos(ecg,t, latidos)
    hr_d, hr_u, t_u, trend = const_RR(ecg,latidos)
    grafico_hr_det(t_u, hr_u, trend, f"HR ENTERO {nombre}")
    #Ff,PSD= transformada_rapida(hr_d, f"HR ENTERO {nombre}")

    ecg_pre,t_pre= leer_archivo(paciente, tiempos[0], tiempos[1])
    #latidos_pre= detect_rpeaks(ecg_pre,t_pre)
    latidos_pre= detect_rpeaks(ecg_pre)
    hr_pre, tr_pre, _ , _= const_RR(ecg_pre,latidos_pre)
    Ff_pre,PSD_pre= transformada_rapida(hr_pre, f"PRE ICTAL {nombre}")
    f_hr_pre,pxx_pre= welch_psd(hr_pre, f"PRE ICTAL {nombre}")
    
    ecg_post, t_post= leer_archivo(paciente, tiempos[2], tiempos[3])
    latidos_post= detect_rpeaks(ecg_post)
    #latidos_post= detect_rpeaks(ecg_post,t_post)
    
    #prueba_latidos(ecg_post,t_post, latidos_post)
    hr_post, tr,  _ , _ = const_RR(ecg_post,latidos_post)
    Ff_post,PSD_post= transformada_rapida(hr_post, f"POST ICTAL {nombre}")
    f_hr_post,pxx_post= welch_psd(hr_post, f"POST ICTAL {nombre}")
    
    
    presentacion_datos(f_hr_pre,pxx_pre,f_hr_post,pxx_post, nombre)
    presentacion_datos(Ff_pre,PSD_pre,Ff_post,PSD_post, nombre)

# %% main - PACIENTES 

def main():

    files = [file for file in listdir('data')]
    # # Paciente 1  (sz01: 00:14:36 → 00:16:12)
    #analizar_paciente(0, 'Paciente 1', [8.6, 12.600, 16.500, 20.500], files)
    
    # # Paciente 2  (sz02_01: 01:02:43 → 01:03:43)
    #analizar_paciente(1, 'Paciente 2 - Episodio 1', [56.717, 60.717, 64.217, 68.217], files)    
    #analizar_paciente(1, 'Paciente 2 – Episodio 2', [169.850, 173.850, 176.767, 180.767], files)

    
    # # Paciente 3  (sz03_01: 01:24:34 → 01:26:22)
    #analizar_paciente(2, 'Paciente 3 - Episodio 1', [78.567, 82.567, 86.867, 90.867], files)
    #analizar_paciente(2, 'Paciente 3 – Episodio 2', [148.450, 152.450, 156.783, 160.783], files)
    
    
    # # Paciente 4  (sz04: 00:20:10 → 00:21:55)
    #analizar_paciente(3, 'Paciente 4', [14.167, 18.167, 22.417, 26.417], files)
    
    # # Paciente 5  (sz05: 00:24:07 → 00:25:30)
    #analizar_paciente(4, 'Paciente 5', [18.117, 22.117, 26.000, 30.000], files)
    
    # # Paciente 6  (sz06_01: 00:51:25 → 00:52:19)
    #analizar_paciente(5, 'Paciente 6 - Episodio 1', [45.417, 49.417, 52.817, 56.817], files)
    #analizar_paciente(5, 'Paciente 6 – Episodio 2', [118.750, 122.750, 126.667, 130.667], files)
    
    # # Paciente 7  (sz07: 01:08:02 → 01:09:31)
    analizar_paciente(6, 'Paciente 7', [62.033, 66.033, 70.017, 74.017], files)
  


    

if __name__=="__main__":
    main()
