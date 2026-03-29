# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 17:51:38 2026

@author: lolyy
"""
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

import variables_globales
# %% Construccion temporal de HR a partir de latidos R (RR) y su interpolacion uniforme para FFT/PSD

def const_RR (ecg, r_locs, fs= variables_globales.fs, fs_hr=variables_globales.fs_hr, hr_bounds=(40, 180), detrend_deg=10):
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

def grafico_hr(res, name):
    # HR por latido
    plt.figure(figsize=(12,4))
    plt.plot(res["t_hr"], res["hr"], label='HR por latido (bpm)')
    plt.xlabel('Tiempo [s]'); plt.ylabel('HR[bpm]'); plt.grid(True); plt.legend(); plt.title(f"FC instantánea ({name})")
    plt.show()

    if res["t_u"] is not None:
        plt.figure(figsize=(10,3))
        plt.plot(res['t_u'],res['hr_u'], label='HR uniforme (4 Hz)')
        if "trend" in res:
            plt.plot(res['t_u'], res["trend"], label='Tendencia (polinomio)', linestyle='--')
            plt.plot(res['t_u'], res["hr_detr"], label='HR sin tendencia')
        plt.xlabel('Tiempo [s]'); plt.ylabel('Latidos por minuto [#bpm]'); plt.grid(True); plt.legend(); plt.title('Serie de HR uniformizada')
        plt.show()
    return
