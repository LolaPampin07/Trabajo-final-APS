# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 18:36:22 2026

@author: lolyy
"""
# %% LIBRERIAS

import matplotlib.pyplot as plt
from os import listdir
import numpy as np
import leer_archivo 
import filtrado_ecg
import periodograma
import rr_hr
import presentacion_datos
import wfdb

# %% Invocacion de funciones presentacion de los datos

def analizar_paciente(record_name):

    # %% lectura de archivo
    ecg, t, fs, seizure_times, r_peaks = leer_archivo.leer_archivo(record_name)

    # %% construccion intervalos RR y frecuencia cardiaca 
    rr, hr, t_hr=rr_hr.const_RR(r_peaks, fs)

    # %% definicion ventana preictal y postictal
    ventana_pre = [seizure_times[0][0] - 240, seizure_times[0][0]-100]
    mask_pre = (t_hr >= ventana_pre[0]) & (t_hr <= ventana_pre[1])
    hr_pre = hr[mask_pre]
    t_pre = t_hr[mask_pre]

    ventana_post = [seizure_times[0][1]+140, seizure_times[0][1] + 470]
    mask_post = (t_hr >= ventana_post[0]) & (t_hr <= ventana_post [1])
    hr_post = hr[mask_post]
    t_post = t_hr[mask_post]

    rr_hr.chequeo(t_hr, hr, t_pre, hr_pre, t_post, hr_post, seizure_times)

    # %% detrend polinomio
    # TOTAL
    tu, hr_u, hr_dt, trend = filtrado_ecg.interp_y_detrend(t_hr, hr, fs=4.0, deg=4)
    # PRE
    tu_pre, hr_pre_u, hr_pre_dt, trend_pre = filtrado_ecg.interp_y_detrend(t_pre, hr_pre, fs=4.0, deg=4)
    # POST
    tu_post, hr_post_u, hr_post_dt, trend_post = filtrado_ecg.interp_y_detrend(t_post, hr_post, fs=4.0, deg=4)

    # %% Espectro
    # por FFT
    ff_pre_fft, psd_pre_fft= periodograma.transformada_rapida(hr_pre_dt, "FFT HR det pre")
    ff_post_fft, psd_post_fft= periodograma.transformada_rapida(hr_post_dt, "FFT HR det post")

    # Welch
    ff_pre_welch, psd_pre_welch= periodograma.transformada_rapida(hr_pre_dt, "Welch HR det pre")
    ff_post_welch, psd_post_welch= periodograma.transformada_rapida(hr_post_dt, "Welch HR det post")

    # Resultados finales
    #periodograma.presentacion_datos(ff_pre_fft, psd_pre_fft, ff_post_fft, psd_post_fft, name="PSD HR FFT")
    #periodograma.presentacion_datos(ff_pre_welch, psd_pre_welch, ff_post_welch, psd_post_welch, name="PSD HR Welch")

    # HR ya interpolada y detrendeada
    f_pre, P_pre, f_post, P_post = periodograma.fft_pre_post(hr_pre_dt, hr_post_dt)
    
    return f_pre, P_pre, f_post, P_post, hr_dt, seizure_times, tu
   

# %% main - PACIENTES 

def main():

    f_pre1, P_pre1, f_post1, P_post1, hr1, sz_t1, tu1 = analizar_paciente("sz01") 
    f_pre2, P_pre2, f_post2, P_post2, hr2, sz_t2, tu2 = analizar_paciente("sz02")
    f_pre3, P_pre3, f_post3, P_post3, hr3, sz_t3, tu3 = analizar_paciente("sz03")
    f_pre4, P_pre4, f_post4, P_post4, hr4, sz_t4, tu4 = analizar_paciente("sz04")
    f_pre5, P_pre5, f_post5, P_post5, hr5, sz_t5, tu5 = analizar_paciente("sz05")
    f_pre6, P_pre6, f_post6, P_post6, hr6, sz_t6, tu6 = analizar_paciente("sz06")
    f_pre7, P_pre7, f_post7, P_post7, hr7, sz_t7, tu7 = analizar_paciente("sz07")

# %% Presentacion de resultados
    eps = 1e-12
    banda=(0.01, 0.1)
    f_lo, f_hi = banda
    db= False
    xlim = (0, 0.5)

    # Agrupamos los datos en listas para iterar cómodo
    f_pre  = [f_pre1, f_pre2, f_pre3, f_pre4, f_pre5, f_pre6, f_pre7]
    P_pre  = [P_pre1, P_pre2, P_pre3, P_pre4, P_pre5, P_pre6, P_pre7]
    f_post = [f_post1, f_post2, f_post3, f_post4, f_post5, f_post6, f_post7]
    P_post = [P_post1, P_post2, P_post3, P_post4, P_post5, P_post6, P_post7]

    names = ["sz01", "sz02", "sz03", "sz04", "sz05", "sz06", "sz07"]    

    # ===== FIGURA =====
    fig, axs = plt.subplots(7, 2, figsize=(14, 22), sharex=True, sharey='row'    )

    for i in range(7):

        # ===== PRE ICTAL =====
        if db:
            axs[i, 0].plot(f_pre[i], 10*np.log10(P_pre[i] + eps), linewidth=1.8)
            axs[i, 0].set_ylabel("Amplitud [dB]")
        else:
            axs[i, 0].plot(f_pre[i], P_pre[i], linewidth=1.8)
            axs[i, 0].set_ylabel("Amplitud")

        axs[i, 0].axvspan(f_lo, f_hi, color="green", alpha=0.15)
        axs[i, 0].axvline(f_lo, ls="--", color="green", alpha=0.8)
        axs[i, 0].axvline(f_hi, ls="--", color="green", alpha=0.8)
        axs[i, 0].grid(True)

        # Etiqueta del caso
        axs[i, 0].text(
            0.02, 0.9,
            names[i],
            transform=axs[i, 0].transAxes,
            fontsize=10,
            fontweight="bold"
        )

        # ===== POST ICTAL =====
        if db:
            axs[i, 1].plot(
                f_post[i], 10*np.log10(P_post[i] + eps),
                linewidth=1.8
            )
        else:
            axs[i, 1].plot(
                f_post[i], P_post[i],
                linewidth=1.8
            )

        axs[i, 1].axvspan(f_lo, f_hi, color="green", alpha=0.15)
        axs[i, 1].axvline(f_lo, ls="--", color="green", alpha=0.8)
        axs[i, 1].axvline(f_hi, ls="--", color="green", alpha=0.8)
        axs[i, 1].grid(True)
    for ax in axs.flatten():
            ax.set_xlim(xlim)
    # ===== TÍTULOS DE COLUMNAS =====
    axs[0, 0].set_title("PRE ICTAL", fontsize=12)
    axs[0, 1].set_title("POST ICTAL", fontsize=12)

    fig.suptitle(
        "FFT HR – Comparación PRE vs POST (7 casos)",
        fontsize=16
    )

    fig.supxlabel("Frecuencia [Hz]", fontsize=12, y=0.02)

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.show()




if __name__=="__main__":
    main()




