# %% librerias + imports + definicion de variables
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy import signal as sig
from scipy.signal import periodogram , get_window
import pandas as pd
from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio
import wave
from numpy.fft import fft

fs = 200  # Hz


"""
Tiempo de sz

paciente1 00:14:36 00:16:12 --> tomo del min 14 hasta el 15
paciente2 00:24:07 00:25:30 --> tomo del 18 al 33
min8 = 480s = 480*200=96000
min24 = 288000
"""

# %% Lectura  + Graficos ECG sin filtrar

def leer_archivo(path, start, stop):
    #Lectura de archivos
    start *= 12000
    stop *= 12000
    raw = np.fromfile('data/' + path, dtype=np.int16)
    gain = 25
    baseline = 26

    ecg_mV = (raw - baseline) / gain
    
    t = np.arange(len(ecg_mV)) / fs
    
    return ecg_mV[start:stop],t[start:stop]
    

def graficar_archivo(ecg_mV, t, name):
    #Graficos ECG
    
    plt.figure(figsize=(12, 4))
    plt.plot(t, ecg_mV, linewidth=1)
    #plt.xlim([0, 10])
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title(f"ECG ({name})")
    plt.grid(True)
    plt.show()
    
    return
 


# %% ### Analisis espectral ###

# Calculo FFT

#calculo la transformada, su modulo y su argumento
def transformada_rapida(x, name):
    X = fft(x)    
    PDS = np.abs(X)**2
    
    N= len(x)
    df= df = fs / N #resolucion espectral = [[1/(s*muestras)]
    
    
    Ff=np.arange(N)*df #mi eje x en hz
    
    Ff = Ff[:N//2]
    PDS = (np.abs(X)**2) / (N * fs)
    PDS = PDS[:N//2]

    # Gráfico
    plt.figure(figsize=(20, 10))
    plt.plot(Ff, 10 * np.log10(PDS + 1e-20), 'x', label=f'FFT - {name}')
    plt.xlim([0, fs/2])
    plt.title(f"PDS [dB] ({name})")
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('PDS [dB]')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return Ff, PDS

# %% Estimacion espectral - Welch

def welch(ecg, name):
    #-----------------PRUEBA COMPARATIVA----------------------
    """
        # --- Welch ---
        nperseg = 1024
        f_welch, Pxx_welch = sig.welch(ecg, fs=fs, window='boxcar', nperseg=nperseg)

        # --- FFT ---
        X = fft(ecg)
        Pxx_fft = (np.abs(X)**2) / (n * fs)  # normalización para PSD
        Ff = np.arange(n) * (fs / n)
        # Solo mitad positiva
        Ff = Ff[:n//2]
        Pxx_fft = Pxx_fft[:n//2]

        # --- Gráfico comparativo ---
        plt.figure(figsize=(20, 10))
        plt.plot(f_welch, 10*np.log10(Pxx_welch + 1e-20), 'o-', label='Welch')
        plt.plot(Ff, 10*np.log10(Pxx_fft + 1e-20), '-', alpha=0.5, label='FFT directa')
        plt.title(f'Comparación PSD - {name}')
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('PSD [dB/Hz]')
        plt.xlim([0, fs/2])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    """
#sig.welch:
#    - Divide la señal en segmentos (con posible solapamiento, si se especifica).
#   - Aplica la ventana a cada segmento.
#    - Calcula la FFT de cada segmento.
#   - Promedia los espectros de potencia de todos los segmentos.

#PARAMETROS WELCH
    cant_seg = 10000000 #cantidad de segmentos ==> mientras mas chico mas varianza
    win_ecg = "boxcar" #ventana rectangular porque es la utilizada en el paper
    # tamaño de segmento tentativo
    n = len(ecg)
    nperseg_ecg = max(8, n // max(1, cant_seg))  # al menos 8 muestras por segmento
    nperseg_ecg = min(nperseg_ecg, n)            # no puede exceder la longitud de la señal
    
    # Si quedó muy chico, reducimos el número de segmentos para asegurar un tamaño razonable
    if nperseg_ecg < 64 and n >= 64:
        nperseg_ecg = 64
    if nperseg_ecg > n:
        nperseg_ecg = n

    # nfft al menos nperseg; redondeo a potencia de 2 para eficiencia
    def _next_pow2(x):
        return 1 << (int(np.ceil(np.log2(max(1, int(x))))))
    nfft_ecg = max(nperseg_ecg, _next_pow2(nperseg_ecg))
    
    # --- Welch ---
    f_ecg, Pxx_ecg = sig.welch(ecg, fs=fs, window=win_ecg, nperseg=nperseg_ecg, nfft=nfft_ecg, return_onesided=True, 
                               average="median", noverlap=nperseg_ecg//2)

    # --- Gráfico ---
    eps = 1e-20  # para evitar log10(0)
    plt.figure(figsize=(20, 10))
    plt.plot(f_ecg, 10 * np.log10(Pxx_ecg + eps), 'x', label=f'PSD (Welch) - {name}')
    plt.title(f'PSD (Welch) - {name}')
    plt.ylabel('Densidad Espectral de Potencia [dB]')
    plt.xlabel('Frecuencia [Hz]')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    return f_ecg, Pxx_ecg
 
'''  
#%% ###### Filtrado lineal ########
# 1) PLANTILLA DE DISEÑO - PASABANDA DIGITAL

wp = (0.8, 35)
ws = (0.1, 40)

# Atenuaciones 
alpha_p = 0.5     # dB
alpha_s = 20      # dB

# %% 1) DISEÑO IIR (BUTTER - CHEBY1 - CHEBY2 - CAUER)

mi_sos_butt  = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s, ftype='butter', fs=fs, output='sos')

mi_sos_cauer = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s, ftype='ellip', fs=fs, output='sos')

mi_sos_cheb1 = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s, ftype='cheby1', fs=fs, output='sos')

mi_sos_cheb2 = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s, ftype='cheby2', fs=fs, output='sos')


# %% Ejemplo grafico de plantilla de 2 filtros iir

mi_sos1 = mi_sos_cauer
tipo1 = "Cauer"

mi_sos2 = mi_sos_cheb1
tipo2 = "Cheby 1"


### Respuesta en frecuencia ###
"""
# freqz_sos = Compute the frequency response of a digital filter in SOS format.
#        Returns:w = ndarray The frequencies at which h was computed, in the same units as fs. By default, w is normalized to the range [0, pi) (radians/sample).
#                h = ndarray, The frequency response, as complex numbers.
"""
w1, h1 = sig.freqz_sos(mi_sos1, worN=2048, fs=fs)
w2, h2 = sig.freqz_sos(mi_sos2, worN=2048, fs=fs)

# Fase
fase1 = np.unwrap(np.angle(h1))
w_rad1 = w1 / (fs/2) * np.pi
gd1 = -np.diff(fase1) / np.diff(w_rad1)
fase2 = np.unwrap(np.angle(h2))
w_rad2 = w2 / (fs/2) * np.pi
gd2 = -np.diff(fase2) / np.diff(w_rad2)

# GRAFICOS de ejemplo de los filtros
# CAUER
plt.figure(figsize=(12, 10))
fig = plt.gcf()

# ===== TÍTULO GENERAL =====
fig.suptitle(f'IIR {tipo1}', fontsize=14, fontweight='bold')

# ===== Subplot 1: Magnitud =====
ax1 = plt.subplot(3, 1, 1)
plt.plot(w1, 20*np.log10(np.abs(h1)), label='|H(ω)|')
# Aseguramos que plot_plantilla dibuje en este subplot activándolo
plt.sca(ax1)
plot_plantilla('bandpass', wp, alpha_p*2, ws, alpha_s*2, fs)
plt.title('Magnitud')
plt.ylabel('|H(ω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

# ===== Subplot 2: Fase =====
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(w1, fase1, label='Fase')
plt.title('Fase')
plt.ylabel('Fase [rad]')
plt.grid(True, which='both', ls=':')
plt.legend()

# ===== Subplot 3: Retardo de grupo =====
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(w1[1:], gd1, label='τg')
plt.title('Retardo de grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Ajuste posicion
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# CHEBY 1
plt.figure(figsize=(12, 10))
fig = plt.gcf()

# ===== TÍTULO GENERAL =====
fig.suptitle(f'IIR {tipo2}', fontsize=14, fontweight='bold')

# ===== Subplot 1: Magnitud =====
ax1 = plt.subplot(3, 1, 1)
plt.plot(w2, 20*np.log10(np.abs(h2)), label='|H(ω)|')
# Aseguramos que plot_plantilla dibuje en este subplot activándolo
plt.sca(ax1)
plot_plantilla('bandpass', wp, alpha_p*2, ws, alpha_s*2, fs)
plt.title('Magnitud')
plt.ylabel('|H(ω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

# ===== Subplot 2: Fase =====
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(w2, fase2, label='Fase')
plt.title('Fase')
plt.ylabel('Fase [rad]')
plt.grid(True, which='both', ls=':')
plt.legend()

# ===== Subplot 3: Retardo de grupo =====
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(w2[1:], gd2, label='τg')
plt.title('Retardo de grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Ajuste posicion
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %% 2) DISEÑO FIR (FIRWIN2 y FIRLS)

frecuencias = np.array([0, 0.1, 0.8, 35, 35.7, fs/2])
deseado = np.array([0, 0, 1, 1, 0, 0])

# FIR con ventana rectangular
numtaps = 1001
fir_win = sig.firwin2(numtaps, frecuencias, deseado, fs=fs, window='boxcar')
retardo = (numtaps - 1)//2

# FIR Least-Squares
numtaps_ls = 2001  # impar
fir_ls = sig.firls(numtaps_ls, frecuencias, deseado, fs=fs)
retardo_ls = (numtaps_ls - 1)//2

# Respuesta en frecuencia
w_fir, h_fir = sig.freqz(fir_win, worN=2048, fs=fs)
fase_fir = np.unwrap(np.angle(h_fir))
gd_fir = -np.diff(fase_fir) / np.diff(w_fir/fs*np.pi)

w_ls, h_ls = sig.freqz(fir_ls, worN=2048, fs=fs)
fase_ls = np.unwrap(np.angle(h_ls))
gd_ls = -np.diff(fase_ls) / np.diff(w_ls/fs*np.pi)

# %% GRAFICOS filtros fir 
# VENTANA RECTANGULAR
plt.figure(figsize=(12, 10))
fig = plt.gcf()

# ===== TÍTULO GENERAL =====
fig.suptitle('FIR ventana rectangular', fontsize=14, fontweight='bold')

# ===== Subplot 1: Magnitud =====
ax1 = plt.subplot(3, 1, 1)
plt.plot(w_fir, 20*np.log10(np.abs(h_fir)), label='|H(ω)|')
plt.sca(ax1)  # asegurar que plot_plantilla dibuje en este subplot
plot_plantilla('bandpass', wp, alpha_p*2, ws, alpha_s*2, fs)
plt.title('Magnitud')
plt.ylabel('|H(ω)| [dB]')
plt.xlim(0, 500)
plt.grid(True, which='both', ls=':')
plt.legend()

# ===== Subplot 2: Fase =====
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(w_fir, fase_fir, label='Fase')
plt.title('Fase')
plt.ylabel('Fase [rad]')
plt.xlim(0, 500)
plt.grid(True, which='both', ls=':')
plt.legend()

# ===== Subplot 3: Retardo de grupo =====
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(w_fir[1:], gd_fir, label='τg')
plt.title('Retardo de grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.xlim(0, 65)
plt.grid(True, which='both', ls=':')
plt.legend()

# Margen para suptitle
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# CUADRADOS MINIMOS 
plt.figure(figsize=(12, 10))
fig = plt.gcf()

# ===== TÍTULO GENERAL =====
fig.suptitle('FIR cuadrados mínimos', fontsize=14, fontweight='bold')

# ===== Subplot 1: Magnitud =====
ax1 = plt.subplot(3, 1, 1)
plt.plot(w_ls, 20*np.log10(np.abs(h_ls)), label='|H(ω)|')
plt.sca(ax1)  # asegurar que plot_plantilla dibuje en este subplot
plot_plantilla('bandpass', wp, alpha_p*2, ws, alpha_s*2, fs)
plt.title('Magnitud')
plt.ylabel('|H(ω)| [dB]')
plt.xlim(0, 500)
plt.grid(True, which='both', ls=':')
plt.legend()

# ===== Subplot 2: Fase =====
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(w_ls, fase_ls, label='Fase')
plt.title('Fase')
plt.ylabel('Fase [rad]')
plt.xlim(0, 500)
plt.grid(True, which='both', ls=':')
plt.legend()

# ===== Subplot 3: Retardo de grupo =====
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(w_ls[1:], gd_ls, label='τg')
plt.title('Retardo de grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.xlim(0, 65)
plt.grid(True, which='both', ls=':')
plt.legend()

# Margen para suptitle
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %% ### Filtrado no lineal ###
'''
# %% main    

def main():
    files = [file for file in listdir('data')]
    paciente1 = files[0]
    ecg, t = leer_archivo(paciente1, 14, 17)
    graficar_archivo(ecg, t, "Paciente 1")
    transformada_rapida(ecg, "Paciente 1")
    welch(ecg,"Paciente 1")
    
    
    files = [file for file in listdir('data')]
    paciente7 = files[2]
    ecg, t = leer_archivo(paciente7, 67, 70)
    graficar_archivo(ecg, t, "Paciente7")
    Ff, PSD=transformada_rapida(ecg, "Paciente 7")
    f_ecg, pxx_ecg= welch(ecg,"Paciente 7")

if __name__=="__main__":
    main()
    



