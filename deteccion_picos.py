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
    patron = patron / np.linalg.norm(patron)

    # ----------------------------
    # 2) Filtro adaptado / correlación
    # ----------------------------
    # FIR con h[::-1] equivale a correlación con h
    ecg_detection = lfilter(b=patron, a=1, x=ecg)
    # Energía absoluta para facilitar el umbral
    det_abs = np.abs(ecg_detection)


    det_abs= np.abs(ecg_detection)/np.std(np.abs(ecg_detection))
    ecg_one_lead_dev = ecg/np.std(ecg)

    mis_qrs, _ = find_peaks(det_abs, height=1, distance=300) #300 parametro electrofisiologico

    plt.figure(figsize=(15,5))
    plt.plot(ecg_one_lead_dev, label = 'ECG original')
    plt.plot(det_abs[57:], label = 'Salida Matched Filter')
    plt.legend()
    plt.show()

    plt.figure(figsize=(15,5))
    plt.plot(ecg_one_lead_dev[:5000], label = 'ECG original')
    plt.plot(det_abs[57:5000], label = 'Salida Matched Filter')
    plt.legend()
    plt.grid(True)



    puntos_spline = np.array(mis_qrs - 110, dtype=int).flatten()

    plt.figure(figsize=(15,5))
    plt.title('Puntos Clave')
    plt.plot(ecg, label='ECG')
    plt.plot(mis_qrs, ecg[mis_qrs.astype(int)], 'ro', label='Pico QRS')
    plt.plot(puntos_spline, ecg[puntos_spline], 'go', label='Puntos Isoeléctricos')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Muestras')
    plt.show()
    
    plt.figure(figsize=(15,5))
    plt.title('Vista en detalle')
    plt.plot(ecg, label='ECG')
    plt.plot(mis_qrs, ecg[mis_qrs.astype(int)], 'ro', label='Pico QRS')
    plt.plot(puntos_spline, ecg[puntos_spline], 'go', label='Puntos Isoeléctricos')
    plt.legend()
    plt.xlim(10000,15000)
    plt.grid(True)
    plt.xlabel('Muestras')
    plt.show()

    return mis_qrs