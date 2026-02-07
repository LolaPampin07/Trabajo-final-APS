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
    

def graficar_ecg(ecg_f, ecg_mV, t, r_locs, name):
    #Graficos ECG, ECG filtrado, picos R detectados
    plt.figure(figsize=(12,4))
    plt.plot(t, ecg_mV, label='ECG crudo', alpha=0.4)
    plt.plot(t, ecg_f, label='ECG filtrado', linewidth=1)
    plt.scatter(r_locs/fs, ecg_f[r_locs], color='r', s=20, label='R-peaks')
    plt.legend(); plt.xlabel('Tiempo [s]'); plt.ylabel('Voltage [mV]'); plt.grid(True); plt.title(plt.title(f"ECG ({name})"))
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



    
# %% Construccion temporal de HR a partir de latidos R (RR) y su interpolacion uniforme para FFT/PSD

def const_RR (ecg, r_locs, fs, fs_hr, hr_bounds=(30, 190), detrend_deg=4):
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

    
    results = {
        "t_hr": t_hr,
        "hr": hr,
    }

    # 5) Serie HR uniforme 
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

def grafico_hr(t_hr, hr, name,t_u):
    # HR por latido
    plt.figure(figsize=(12,4))
    plt.plot(t_hr, hr, '-o', label='HR por latido (bpm)')
    plt.xlabel('Tiempo [s]'); plt.ylabel('HR[bpm]'); plt.grid(True); plt.legend(); plt.title(f"FC instantánea ({name})")
    plt.show()

    if t_u is not None:
        plt.figure(figsize=(10,3))
        plt.plot(t_u, hr_u, label='HR uniforme (4 Hz)')
        if "trend" in res:
            plt.plot(t_u, res["trend"], label='Tendencia (polinomio)', linestyle='--')
            plt.plot(t_u, res["hr_detr"], label='HR sin tendencia')
        plt.xlabel('Tiempo [s]'); plt.ylabel('Latidos por minuto [#bpm]'); plt.grid(True); plt.legend(); plt.title('Serie de HR uniformizada')
        plt.show()
    return

def transformada_rapida(x, name):
    X = fft(x)    
    PDS = np.abs(X)**2
    
    N= len(x)
    df= df = fs_hr / N #resolucion espectral = [[1/(s*muestras)]
    
    
    Ff=np.arange(N)*df #mi eje x en hz
    
    Ff = Ff[:N//2]
    PDS = (np.abs(X)**2) / (N * fs_hr)
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

    nperseg = int(seg_len_sec * fs_hr)
    nperseg = max(64, min(nperseg, n))
    win = windows.hann(nperseg, sym=False)
    noverlap = int(nperseg * overlap)

    def next_pow2(x):
        return 2 ** math.ceil(math.log2(max(1, int(x))))
    nfft = max(nperseg, next_pow2(nperseg))

 
    detrend_arg = False if detrend_already_done else 'constant'

    f, Pxx = welch(x,
                   fs=fs_hr,
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
    r_locs, ecg_f, y_int, thr = detect_rpeaks(ecg, fs, band=(8,20), win_ms=120, refractory_ms=220)
    res = const_RR(ecg, r_locs)

    t_hr, hr = res["t_hr"], res["hr"]
    t_u, hr_u = res.get("t_u", None), res.get("hr_u", None)

    graficar_ecg(ecg_f,ecg, t, r_locs, nombre) #grafica el ECG
    grafico_hr(t_hr, hr, nombre, t_u) #grafica la HR por latido
    #Ff,PSD= transformada_rapida(hr_d, f"HR ENTERO {nombre}")

    ecg_pre,t_pre= leer_archivo(paciente, tiempos[0], tiempos[1])
    r_locs_pre, ecg_f_pre, y_int_pre, thr_pre = detect_rpeaks(ecg_pre, fs, band=(8,20), win_ms=120, refractory_ms=220)
    res_pre = const_RR(ecg_pre, r_locs_pre, fs, fs_hr, hr_bounds=(MIN_HR, MAX_HR), detrend_deg=4)

    t_hr_pre, hr_pre = res_pre["t_hr"], res_pre["hr"]
    t_u_pre, hr_u_pre = res_pre.get("t_u", None), res_pre.get("hr_u", None)

    
    Ff_pre,PSD_pre= transformada_rapida(hr_u_pre, f"PRE ICTAL {nombre}")
    f_hr_pre,pxx_pre= welch_psd(hr_u_pre, f"PRE ICTAL {nombre}")
    


    ecg_post, t_post= leer_archivo(paciente, tiempos[2], tiempos[3])
    r_locs_post, ecg_f_post, y_int_post, thr_post = detect_rpeaks(ecg_post, fs, band=(8,20), win_ms=120, refractory_ms=220)
    res_post = const_RR(ecg_post, r_locs_post, fs, fs_hr, hr_bounds=(MIN_HR, MAX_HR), detrend_deg=4)
    t_hr_post, hr_post = res_post["t_hr"], res_post["hr"]
    t_u_post, hr_u_post = res_post.get("t_u", None), res_post.get("hr_u", None)    
    Ff_post,PSD_post= transformada_rapida(hr_u_post, f"POST ICTAL {nombre}")
    f_hr_post,pxx_post= welch_psd(hr_u_post, f"POST ICTAL {nombre}")
    
    
    
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
