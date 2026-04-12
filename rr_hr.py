# -*- cot_hrding: utf-8 -*-
"""
Created on Thu Feb 19 17:51:38 2026

@author: lolyy
"""
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import filtrado_ecg
# %% Construccion temporal de HR a partir de latidos R (RR) y su interpolacion uniforme para FFT/PSD

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def const_RR(latidos, mostrar=False):
    
    latidos = np.asarray(latidos) #lo convierto en array por las dudas
   
    # RR y HR
    rr = np.diff(latidos)
    hr = 60 / rr
    t_hr = latidos[:-1] + rr / 2  

    # Filtrado de mediana (ventana corta) p/ eliminacion de spikes
    rr_clean = medfilt(rr, kernel_size=5)
    hr_clean = medfilt(hr, kernel_size=5)

    if mostrar:
        fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # ----- RR -----
        axs[0].plot(t_hr, rr, color='gray', alpha=0.6,
                    label='RR original (detección automática)')
        axs[0].plot(t_hr, rr_clean, color='blue', linewidth=2,
                    label='RR filtrado (mediana, W=5)')
        axs[0].set_ylabel('RR [s]')
        axs[0].set_title('Eliminación de artefactos en RR y HR mediante filtro de mediana')
        axs[0].grid(True)
        axs[0].legend()

        # ----- HR -----
        axs[1].plot(t_hr, hr, color='gray', alpha=0.6,
                    label='HR original (detección automática)')
        axs[1].plot(t_hr, hr_clean, color='blue', linewidth=2,
                    label='HR filtrada (mediana, W=5)')
        axs[1].set_xlabel('Tiempo [s]')
        axs[1].set_ylabel('Frecuencia cardíaca [bpm]')
        
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    return rr_clean, hr_clean, t_hr


def chequeo (t_hr, hr, t_pre, hr_pre, t_post, hr_post, seizure_time):
    plt.figure(figsize=(12,4))
    plt.plot(t_hr, hr, label='HR total', alpha=0.4)
    plt.plot(t_pre, hr_pre, 'g', label='Pre-ictal')
    plt.plot(t_post, hr_post, 'r', label='Post-ictal')
    plt.axvline(seizure_time[0][0], color='k', linestyle='--', label='Onset')
    plt.axvline(seizure_time[0][1], color='k', linestyle='--')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('HR [bpm]')
    plt.legend()
    plt.grid(True)
    plt.show()

    return

