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
import deteccion_picos
import periodograma
import variables_globales
import rr_hr
import wfdb

# %% Invocacion de funciones presentacion de los datos

def analizar_paciente(record_name):

    ecg, t, fs, seizure_times, r_peaks = leer_archivo.leer_archivo(record_name, False)
    rr, hr, t_hr=rr_hr.const_RR(r_peaks, fs, False)

    ### defino ventana preictal y postictal
    ventana_pre = [seizure_times[0][0] - 180, seizure_times[0][0]]#+3mins
    mask_pre = (t_hr >= ventana_pre[0]) & (t_hr <= ventana_pre[1])
    hr_pre = hr[mask_pre]
    t_pre = t_hr[mask_pre]

    ventana_post = [seizure_times[0][1], seizure_times[0][1] + 300]
    mask_post = (t_hr >= ventana_post[0]) & (t_hr <= ventana_post [1])
    hr_post = hr[mask_post]
    t_post = t_hr[mask_post]

    rr_hr.chequeo(t_hr, hr, t_pre, hr_pre, t_post, hr_post, seizure_times)

    




# %% main - PACIENTES 

def main():

    analizar_paciente("sz01") 

if __name__=="__main__":
    main()




