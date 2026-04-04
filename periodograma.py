# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 12:12:15 2026

@author: lolyy
"""

from scipy.signal import welch, windows
from numpy.fft import fft
import math
import numpy as np
import matplotlib.pyplot as plt

# %%FFT
def transformada_rapida(x, name, fs_hr=4, mostrar=False):
    X = fft(x)    
    PDS = np.abs(X)**2
    
    N= len(x)
    df= fs_hr / N #resolucion espectral = [[1/(s*muestras)]
    
    
    Ff=np.arange(N)*df #mi eje x en hz
    
    Ff = Ff[:N//2]
    PDS = (np.abs(X)**2) / (N * fs_hr)
    PDS = PDS[:N//2]

#    Gráfico
    if mostrar:
        plt.figure(figsize=(20, 10))
        plt.plot(Ff, 10 * np.log10(PDS + 1e-20), 'x-', label='FFT')
        plt.xlim([0, fs_hr/2])
        plt.title("PDS [dB] ")
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('PDS [dB]')
        plt.title(f'PSD (FFT) - {name}')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return Ff, PDS

# %% PSD POR WELCH
def welch_psd(x, name="PSD WELCH",
              seg_len_sec=60,
              overlap=0.5,
              detrend_already_done=True):

    n = len(x)
    if n < 4:
        raise ValueError("La señal es demasiado corta.")

    nperseg = int(seg_len_sec * variables_globales.fs_hr)
    nperseg = max(64, min(nperseg, n))
    win = windows.hann(nperseg, sym=False)
    noverlap = int(nperseg * overlap)

    def next_pow2(x):
        return 2 ** math.ceil(math.log2(max(1, int(x))))
    nfft = max(nperseg, next_pow2(nperseg))

 
    detrend_arg = False if detrend_already_done else 'constant'

    f, Pxx = welch(x,
                   fs=variables_globales.fs_hr,
                   window=win,
                   nperseg=nperseg,
                   noverlap=noverlap,
                   nfft=nfft,
                   detrend=detrend_arg,   # <- clave
                   return_onesided=True,
                   average='median',
                   scaling='density')
    return f, Pxx

# %% Resultados finales

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