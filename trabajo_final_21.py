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

# %% Invocacion de funciones presentacion de los datos

def analizar_paciente(indice, nombre, tiempos, archivos):
    
    paciente = archivos[indice]
    ecg, t = leer_archivo.abrir_archivo(paciente, tiempos[0] , tiempos[-1])
    filtro_lineal= filtrado_ecg.ecg_filter(ecg,t)
    
    #filtromediana= filtrado_ecg.filt_mediana(ecg)
    filtromediana= filtrado_ecg.filt_mediana(filtro_lineal)
    latidos= filtrado_ecg.splines_cubicos(filtro_lineal)


    # ecg_pre,t_pre= leer_archivo.abrir_archivo(paciente, tiempos[0], tiempos[1])
    # r_locs_pre, ecg_f_pre, y_int_pre, thr_pre = deteccion_picos.detect_rpeaks(ecg_pre, variables_globales.fs, band=(8,20), win_ms=120, refractory_ms=220)
    # r_locs_pre=deteccion_picos.matched_filter_ecg(ecg_f_pre)
    # res_pre = rr_hr.const_RR(ecg_pre, r_locs_pre, variables_globales.fs, variables_globales.fs_hr, hr_bounds=(variables_globales.MIN_HR, variables_globales.MAX_HR), detrend_deg=4)
    # Ff_pre,PSD_pre= periodograma.transformada_rapida(res_pre['hr_u'], f"PRE ICTAL {nombre}")
    # f_hr_pre,pxx_pre= periodograma.welch_psd(res_pre['hr_u'], f"PRE ICTAL {nombre}")
    


    # ecg_post, t_post= leer_archivo.abrir_archivo(paciente, tiempos[2], tiempos[3])
    # r_locs_post, ecg_f_post, y_int_post, thr_post = deteccion_picos.detect_rpeaks(ecg_post, variables_globales.fs, band=(8,20), win_ms=120, refractory_ms=220)
    # r_locs_post=deteccion_picos.matched_filter_ecg(ecg_f_post)
    # res_post = rr_hr.const_RR(ecg_post, r_locs_post, variables_globales.fs, variables_globales.fs_hr, hr_bounds=(variables_globales.MIN_HR, variables_globales.MAX_HR), detrend_deg=4)
    # Ff_post,PSD_post= periodograma.transformada_rapida(res_post['hr_u'], f"POST ICTAL {nombre}")
    # f_hr_post,pxx_post= periodograma.welch_psd(res_post['hr_u'], f"POST ICTAL {nombre}")
    
    
    # periodograma.presentacion_datos(f_hr_pre,pxx_pre,f_hr_post,pxx_post, nombre)
    # periodograma.presentacion_datos(Ff_pre,PSD_pre,Ff_post,PSD_post, nombre)

# %% main - PACIENTES 

def main():

    files = [file for file in listdir('data')]
    # # Paciente 1  (sz01: 00:14:36 → 00:16:12)
    analizar_paciente(0, 'Paciente 1', [10.6, 13.500, 16.5, 22], files)
    
    # # Paciente 2  (sz02_01: 01:02:43 → 01:03:43)
    #analizar_paciente(1, 'Paciente 2 - Episodio 1', [56.717, 60.717, 64.217, 68.217], files)    
    #analizar_paciente(1, 'Paciente 2 – Episodio 2', [169.850, 173.850, 176.767, 180.767], files)

    
    # # Paciente 3  (sz03_01: 01:24:34 → 01:26:22)
    #analizar_paciente(2, 'Paciente 3 - Episodio 1', [78.567, 82.567, 86.867, 90.867], files)
    #analizar_paciente(2, 'Paciente 3 – Episodio 2', [148.450, 152.450, 156.783, 160.783], files)
    
    
    # # Paciente 4  (sz04: 00:20:10 → 00:21:55)
    #analizar_paciente(3, 'Paciente 4', [14.167, 18.167, 22.417, 26.417], files)
    
    # # Paciente 5  (sz05: 00:24:07 → 00:25:30)
    #analizar_paciente(4, 'Paciente 5', [18.117, 22.117, 26.000, 30.000], files)
    
    # # Paciente 6  (sz06_01: 00:51:25 → 00:52:19)
    #analizar_paciente(5, 'Paciente 6 - Episodio 1', [45.417, 49.417, 52.817, 56.817], files)
    #analizar_paciente(5, 'Paciente 6 – Episodio 2', [118.750, 122.750, 126.667, 130.667], files)
    
    # # Paciente 7  (sz07: 01:08:02 → 01:09:31)
    #analizar_paciente(6, 'Paciente 7', [62.033, 66.033, 70.017, 74.017], files)   

if __name__=="__main__":
    main()
