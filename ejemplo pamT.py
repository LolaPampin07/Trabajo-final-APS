# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 06:52:12 2026

@author: lolyy
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch

fs=200

def bandpass_ecg(ecg, fs, low=8.0, high=20.0, order=4, notch_50=False, q_notch=30):
    ecg = np.asarray(ecg, dtype=np.float64).ravel()
    # Pasabanda Butterworth en Hz (SciPy moderno)
    sos = butter(order, [low, high], btype='bandpass', output='sos', fs=fs)
    y = sosfiltfilt(sos, ecg)

    if notch_50:
        # Notch 50 Hz (f0=50, Q alrededor de 30–35 va bien)
        b, a = iirnotch(w0=50, Q=q_notch, fs=fs)
        # Una sola pasada es suficiente; si tu pasabanda < 40 Hz, probablemente no haga falta
        from scipy.signal import filtfilt
        y = filtfilt(b, a, y)
    return y

from scipy.signal import find_peaks

def detect_rpeaks(ecg, fs, band=(8,20), win_ms=120, refractory_ms=200, prominence=None, notch_50=False):
    """
    Devuelve índices de picos R en la señal original.
    """
    # 1) Filtrado para resaltar QRS
    ecg_f = bandpass_ecg(ecg, fs, low=band[0], high=band[1], order=4, notch_50=notch_50)

    # 2) Realce tipo Pan-Tompkins: derivada + cuadrado + ventana móvil
    d = np.diff(ecg_f, prepend=ecg_f[0])
    y = d**2
    win = max(1, int(round(win_ms * fs / 1000.0)))
    kernel = np.ones(win) / win
    # Convolución con misma longitud
    y_int = np.convolve(y, kernel, mode='same')

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
from scipy.interpolate import interp1d

def build_hr_from_r(ecg, r_locs, fs, fs_hr=4.0, hr_bounds=(40, 180), detrend_deg=None):
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

def leer_archivo(path, start, stop):
    #Lectura de archivos
    start = round(start * 12000)
    stop = round(stop * 12000)
    raw = np.fromfile('data/' + path, dtype=np.int16)
    gain = 25
    baseline = 26

    ecg_mV = (raw - baseline) / gain
    
    t = np.arange(len(ecg_mV)) / fs

    return ecg_mV[start:stop],t[start:stop]
from os import listdir
files = [file for file in listdir('data')]
paciente = files[0]
ecg, t = leer_archivo(paciente, 8.6 , 20.500)

# Supongamos ecg (array) y fs (Hz)
r_locs, ecg_f, y_int, thr = detect_rpeaks(ecg, fs=200, band=(8,20), win_ms=120, refractory_ms=220, notch_50=True)

res = build_hr_from_r(ecg, r_locs, fs=200, fs_hr=4.0, hr_bounds=(40, 180), detrend_deg=4)

t_hr, hr = res["t_hr"], res["hr"]
t_u, hr_u = res.get("t_u", None), res.get("hr_u", None)

# Graficar
import matplotlib.pyplot as plt

# ECG y picos R
t = np.arange(len(ecg)) / fs
plt.figure(figsize=(12,4))
plt.plot(t, ecg, label='ECG crudo', alpha=0.4)
plt.plot(t, ecg_f, label='ECG filtrado', linewidth=1)
plt.scatter(r_locs/200.0, ecg_f[r_locs], color='r', s=20, label='R-peaks')
plt.legend(); plt.xlabel('Tiempo (s)'); plt.ylabel('mV'); plt.grid(True); plt.title('Detección de R')

# HR por latido
plt.figure(figsize=(10,3))
plt.plot(t_hr, hr, '-o', label='HR por latido (bpm)')
plt.xlabel('Tiempo (s)'); plt.ylabel('bpm'); plt.grid(True); plt.legend(); plt.title('FC instantánea')

# HR uniforme + (opcional) detrend
if t_u is not None:
    plt.figure(figsize=(10,3))
    plt.plot(t_u, hr_u, label='HR uniforme (4 Hz)')
    if "trend" in res:
        plt.plot(t_u, res["trend"], label='Tendencia (polinomio)', linestyle='--')
        plt.plot(t_u, res["hr_detr"], label='HR sin tendencia')
    plt.xlabel('Tiempo (s)'); plt.ylabel('bpm'); plt.grid(True); plt.legend(); plt.title('Serie de HR uniformizada')
plt.show()