# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 12:12:15 2026

@author: lolyy
"""

from scipy.signal import welch, windows
from numpy.fft import fft
import numpy as np
import matplotlib.pyplot as plt

# %% resultados como en el paper
import numpy as np
import matplotlib.pyplot as plt

def fft_pre_post(
    x_pre,
    x_post,
    fs=4.0,
    banda=(0.01, 0.1),
    xlim=(0, 0.4),
    db=False,
    mostrar=False,
    name="HR FFT"
):

    def _fft(x):
        x = np.asarray(x)
        x = x - np.mean(x)
        N = len(x)

        X = np.fft.fft(x)
        f = np.fft.fftfreq(N, d=1/fs)

        mask = f >= 0
        return f[mask], np.abs(X[mask])**2 /(N * fs)

    # FFT
    f_pre, P_pre = _fft(x_pre)
    f_post, P_post = _fft(x_post)

    if mostrar:
        eps = 1e-12
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        f_lo, f_hi = banda

        # ===== Cálculo de ylim común =====
        if db:
            P_pre_plot = 10 * np.log10(P_pre + eps)
            P_post_plot = 10 * np.log10(P_post + eps)

            ymin = min(P_pre_plot.min(), P_post_plot.min())
            ymax = max(P_pre_plot.max(), P_post_plot.max())
        else:
            ymin = 0
            ymax = max(P_pre.max(), P_post.max())

        ylim = (ymin, ymax)

        # ===== PRE ICTAL =====
        if db:
            axs[0].plot(f_pre, P_pre_plot, linewidth=2, label="PRE ICTAL")
            axs[0].set_ylabel("Amplitud espectral [dB]")
        else:
            axs[0].plot(f_pre, P_pre, linewidth=2, label="PRE ICTAL")
            axs[0].set_ylabel("Amplitud espectral")

        axs[0].axvspan(f_lo, f_hi, color="green", alpha=0.15)
        axs[0].axvline(f_lo, color="green", linestyle="--", alpha=0.8)
        axs[0].axvline(f_hi, color="green", linestyle="--", alpha=0.8)
        axs[0].set_ylim(ylim)
        axs[0].set_title("PRE ICTAL")
        axs[0].grid(True)
        axs[0].legend()

        # ===== POST ICTAL =====
        if db:
            axs[1].plot(f_post, P_post_plot, linewidth=2, label="POST ICTAL")
            axs[1].set_ylabel("Amplitud espectral [dB]")
        else:
            axs[1].plot(f_post, P_post, linewidth=2, label="POST ICTAL")
            axs[1].set_ylabel("Amplitud espectral")

        axs[1].axvspan(f_lo, f_hi, color="green", alpha=0.15,
                       label=f"Banda {f_lo}–{f_hi} Hz")
        axs[1].axvline(f_lo, color="green", linestyle="--", alpha=0.8)
        axs[1].axvline(f_hi, color="green", linestyle="--", alpha=0.8)
        #axs[1].set_ylim(ylim)
        axs[1].set_title("POST ICTAL")
        axs[1].grid(True)
        axs[1].legend()

        # ===== Ajustes comunes =====
        axs[1].set_xlim(xlim)
        axs[1].set_xlabel("Frecuencia [Hz]")
        fig.suptitle(f"FFT ritmo cardíaco – {name}", fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    return f_pre, P_pre, f_post, P_post


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
        plt.plot(Ff, PDS, 'x-', label='FFT')
        plt.xlim([0, 0.5])
        plt.title("PDS [dB] ")
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('PDS [dB]')
        plt.title(f'PSD (FFT) - {name}')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return Ff, PDS

# %% PSD POR WELCH
import numpy as np
from scipy.signal import welch, detrend

def welch_psd(x, fs=4):

    x = np.asarray(x)
    N = len(x)

    # === Parámetros Welch conservadores ===
    cant_segmentos = 4                 # CLAVE: pocos segmentos para no matar las oscilaciones
    nperseg = N // cant_segmentos
    noverlap = nperseg // 2

    # Zero padding moderado (mejora visual, no resolución real)
    nfft = 2 * nperseg

    # Ventana suave, pero sin excesiva pérdida de amplitud
    window = np.hanning(nperseg)

    f, Pxx = welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        scaling='density',
        average='mean',
        return_onesided=True
    )

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


    # Estilo base
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)

    # Curvas PSD en dB
    ax.plot(f_pre, pxx_pre, 
            label="PRE ICTAL", 
            color="#1f77b4", linewidth=2.2)
    ax.plot(f_post, pxx_post, 
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
    ax.set_ylabel("PSD ", fontsize=12)
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