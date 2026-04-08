# -*- cot_hrding: utf-8 -*-
"""
Created on Thu Feb 19 17:51:38 2026

@author: lolyy
"""
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
# %% Construccion temporal de HR a partir de latidos R (RR) y su interpolacion uniforme para FFT/PSD

import numpy as np
import matplotlib.pyplot as plt

def const_RR(latidos, fs, mostrar=True):
    latidos = np.asarray(latidos) #paso latidos a array
    rr_vect= np.diff(latidos)
    hr = 60/rr_vect 
    t_latidos= latidos/fs #esta en muestras
    t_hr = t_latidos[:-1] + rr_vect/2  #[:-1] --> Toma todos menos el último
    if mostrar:
       plt.figure(figsize=(12, 8))
       # Subplot 1: RR
       plt.subplot(2, 1, 1)
       plt.plot(rr_vect, label='Segmentos RR')
       plt.xlabel('Índice de latido')
       plt.ylabel('RR [ms]')
       plt.grid(True)
       plt.legend()
       plt.title("Construcción del vector RR")
       
       plt.subplot(2, 1, 2)
       plt.plot(t_hr,hr, label='Frecuencia cardiaca')
       plt.xlabel('Tiempo [s]')
       plt.ylabel('HR [bpm]')
       plt.grid(True)
       plt.legend()
       plt.title('Frecuencia cardiaca a partir de RR')
       plt.tight_layout()
       plt.show()
    return rr_vect, hr, t_hr
def chequeo (t_hr, hr, t_pre, hr_pre, t_post, hr_post, seizure_time):
    plt.figure(figsize=(12,4))
    plt.plot(t_hr, hr, label='HR total', alpha=0.4)
    plt.plot(t_pre, hr_pre, 'g', label='Pre-ictal')
    plt.plot(t_post, hr_post, 'r', label='Post-ictal')
    plt.axvline(seizure_time[0][0], color='k', linestyle='--', label='Onset')
    plt.axvline(seizure_time[0][1], color='k', linestyle='--')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('HR [latido/s]')
    plt.legend()
    plt.grid(True)
    plt.show()

    return

