# %% LIBRERIAS

import matplotlib.pyplot as plt
from scipy.signal import find_peaks, lfilter
import numpy as np
import scipy.io as sio


import variables_globales

def det_lat_filt_adp (ecg):
    # ----------------------------
    # 1) Preprocesar el patrón (cero media y normalización)
    # ----------------------------
    mat = sio.loadmat('./ecg.mat')
    patron = mat['qrs_pattern1'].flatten() - np.mean(mat['qrs_pattern1'].flatten()) #para tener area neta nula, util para filtrar
    plt.figure()
    plt.plot(patron) #muestro el patron que utilizo para detectar latidos
    plt.title("PATRON PARA DETECTAR LATIDOS")
    patron_norm = np.linalg.norm(patron)
    patron = patron / patron_norm

    return

# %% Deteccion de latidos - Filtro adaptado
def matched_filter_ecg(
    ecg,                            # senal ECG 1D
    fs=variables_globales.fs,       # Frecuencia de muestreo [Hz]
    threshold_rel=4.0,              # Factor sobre la mediana absoluta de la señal de detección (robusto)
    distance_ms=300,                # Refractario mínimo entre picos (ms)
    prominence_min=None,            # Prominencia mínima absoluta (si None, se fija automáticamente)
    width_ms=(60, 120),             # Rango de ancho de pico esperado (ms) ~60-120 ms para QRS
    ventana_qrs_muestras=60,        # Mitad de ventana para extraer latido (± muestras)
    devolver_senal_filtrada=False   # Para depurar: devuelve ecg filtrado
):
    # ----------------------------
    # 1) Preprocesar el patrón (cero media y normalización)
    # ----------------------------    
    # ----------------------------
    # 1) Preprocesar el patrón (cero media y normalización)
    # ----------------------------
    mat = sio.loadmat('./ecg.mat')
    patron = mat['qrs_pattern1'].flatten() - np.mean(mat['qrs_pattern1'].flatten()) #para tener area neta nula, util para filtrar
    plt.figure()
    plt.plot(patron) #muestro el patron que utilizo para detectar latidos
    plt.title("PATRON PARA DETECTAR LATIDOS")
    patron_norm = np.linalg.norm(patron)
    patron = patron / patron_norm



    # Si la polaridad típica de la correlación es negativa, invierto patrón --> (útil si el QRS dominante es negativo o la referencia está invertida).
    # usar una correlación corta para estimar signo predominante
    test_len = 5*len(patron) # minimo de muestras para testear si esta invertido
    corr_test = lfilter(b=patron[::-1], a=1, x=ecg[:test_len])
    if np.nanmean(corr_test) < 0:
        patron = -patron

    # ----------------------------
    # 2) Filtro adaptado / correlación
    # ----------------------------
    # FIR con h[::-1] equivale a correlación con h
    ecg_detection = lfilter(b=patron[::-1], a=1, x=ecg)
    # Energía absoluta para facilitar el umbral
    det_abs = np.abs(ecg_detection)

    # ----------------------------
    # 3) Umbral robusto + parámetros de find_peaks
    # ----------------------------
    # Mediana y MAD-like (mediana de |x - mediana|)
    med = np.median(det_abs)
    mad = np.median(np.abs(det_abs - med)) + 1e-12
    umbral = med + threshold_rel * mad
    
    umbral=0.8

    # Refractario en muestras
    distance_samp = int(round((distance_ms/1000.0)*fs))
    distance_samp = max(distance_samp, 1)

    # Anchos en muestras (opcional)
    width_samp = None
    if width_ms is not None:
        wmin, wmax = width_ms
        wmin_s = int(round((wmin/1000.0)*fs))
        wmax_s = int(round((wmax/1000.0)*fs))
        # limitar a valores válidos
        wmin_s = max(wmin_s, 1)
        wmax_s = max(wmax_s, wmin_s+1)
        width_samp = (wmin_s, wmax_s)

    # Prominencia mínima (si no se pasa) — relativo a mad
    if prominence_min is None:
        prominence_min = 2.0 * mad  # razonable; ajustable según ruido

    # ----------------------------
    # 4) Detección de picos
    # ----------------------------
    peak_kwargs = dict(
        height=umbral,
        distance=distance_samp,
        prominence=prominence_min
    )
    if width_samp is not None:
        peak_kwargs["width"] = width_samp

    qrs_idx, peak_props = find_peaks(det_abs, **peak_kwargs)

    # ----------------------------
    # 5) Extracción de latidos (ventanas centradas)
    # ----------------------------
    win = int(ventana_qrs_muestras)
    # aseguramos que haya suficiente espacio
    valid_mask = (qrs_idx - win >= 0) & (qrs_idx + win < len(ecg))
    qrs_idx_valid = qrs_idx[valid_mask]

    if qrs_idx_valid.size > 0:
        qrs_mat = np.array([ecg[ii-win:ii+win] for ii in qrs_idx_valid])
        # quitar media por latido (como hacías)
        qrs_mat = qrs_mat - np.mean(qrs_mat, axis=1, keepdims=True)
    else:
        qrs_mat = np.empty((0, 2*win))

    # Extra: prominencias y anchos (en muestras)
    prominencias = peak_props.get("prominences", np.array([]))
    width_samples = peak_props.get("widths", np.array([]))

    resultados = {
        "qrs_indices": qrs_idx_valid.astype(int),
        "ecg_detection": ecg_detection,
        "umbral": float(umbral),
        "qrs_mat": qrs_mat,
        "prominencias": prominencias,
        "ancho_muestras": width_samples,
        "params": {
            "fs": fs,
            "threshold_rel": threshold_rel,
            "distance_ms": distance_ms,
            "prominence_min": prominence_min,
            "width_ms": width_ms,
            "ventana_qrs_muestras": ventana_qrs_muestras
        }
    }

    qrs_idx = qrs_idx[:-1]
    # Visualizar patrón y detección
    plt.figure(figsize=(12,4))
    plt.plot(ecg_detection, label='Detección (|corr|)')
    plt.hlines(umbral, 0, len(ecg_detection), color='r', linestyles='--', label='Umbral')
    plt.plot(qrs_idx, np.abs(ecg_detection[qrs_idx_valid.astype(int)]), 'go', label='QRS')
    plt.title("Señal de detección y picos QRS")
    plt.legend()
    plt.tight_layout()

    return qrs_idx

def filtro_adp(ecg):
    sio.whosmat('./ecg.mat')
    mat_struct = sio.loadmat('./ecg.mat')
    patron = mat_struct['qrs_pattern1'].flatten()
    patron_2 = patron - np.mean(patron) #para tener area neta nula, util para filtrar

    ecg_detection = lfilter(b=patron_2, a=1, x=ecg)

    ecg_detection_abs = np.abs(ecg_detection)/np.std(np.abs(ecg_detection))
    ecg_one_lead_dev = ecg/np.std(ecg)

    mis_qrs, _ = find_peaks(ecg_detection_abs, height=1, distance=300) #300 parametro electrofisiologico

    plt.figure(figsize=(15,5))
    plt.plot(ecg_one_lead_dev, label = 'ECG original')
    plt.plot(ecg_detection_abs[57:], label = 'Salida Matched Filter')
    plt.legend()
    plt.show()

    plt.figure(figsize=(15,5))
    plt.plot(ecg_one_lead_dev[:5000], label = 'ECG original')
    plt.plot(ecg_detection_abs[57:5000], label = 'Salida Matched Filter')
    plt.legend()
    plt.grid(True)

    return mis_qrs

