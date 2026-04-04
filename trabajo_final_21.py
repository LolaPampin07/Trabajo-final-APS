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
import wfdb

# %% Invocacion de funciones presentacion de los datos

def analizar_paciente(record_name):

    # %% lectura de archivo
    ecg, t, fs, seizure_times, r_peaks = leer_archivo.leer_archivo(record_name, False)

    # %% construccion intervalos RR y frecuencia cardiaca 
    rr, hr, t_hr=rr_hr.const_RR(r_peaks, fs, False)

    # %% definicion ventana preictal y postictal
    ventana_pre = [seizure_times[0][0] - 180, seizure_times[0][0]]#+3mins
    mask_pre = (t_hr >= ventana_pre[0]) & (t_hr <= ventana_pre[1])
    hr_pre = hr[mask_pre]
    t_pre = t_hr[mask_pre]

    ventana_post = [seizure_times[0][1], seizure_times[0][1] + 300]
    mask_post = (t_hr >= ventana_post[0]) & (t_hr <= ventana_post [1])
    hr_post = hr[mask_post]
    t_post = t_hr[mask_post]

    rr_hr.chequeo(t_hr, hr, t_pre, hr_pre, t_post, hr_post, seizure_times)

    # %% detrend polinomio
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
    periodograma.presentacion_datos(ff_pre_fft, psd_pre_fft, ff_post_fft, psd_post_fft, name="PSD HR FFT")
    periodograma.presentacion_datos(ff_pre_welch, psd_pre_welch, ff_post_welch, psd_post_welch, name="PSD HR Welch")

    # HR ya interpolada y detrendeada
    f_pre, P_pre, f_post, P_post = periodograma.fft_pre_post(hr_pre_dt, hr_post_dt)
   

# %% main - PACIENTES 

def main():

    #analizar_paciente("sz01") 
    #analizar_paciente("sz02")
    analizar_paciente("sz03")
    #analizar_paciente("sz04")
    #analizar_paciente("sz05")
    #analizar_paciente("sz06")
    #analizar_paciente("sz07")


if __name__=="__main__":
    main()




