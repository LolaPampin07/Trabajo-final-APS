# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 17:17:39 2026

@author: lolyy
"""

# %% Librerias + imports

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert
import os
import pandas as pd

# %% Lectura de EEG (time domain)

# Ruta del archivo EDF
pac_0_0 = "C:\PN00-1.edf" #paciente 0 crisis 0

# Leer el archivo EDF
raw = mne.io.read_raw_edf(pac_0_0, preload=True)

# Mostrar información básica
print(raw.info)

# Graficar las señales
raw.plot(duration=100, n_channels=29)  # duración en segundos y número de canales

# %%



# %% Frecuency domain

# Filtrado: notch 50 Hz + bandpass 0.5–45 Hz
raw_f = raw.copy().load_data()
raw_f.notch_filter(freqs=[50], picks='eeg')
raw_f.filter(l_freq=0.5, h_freq=45, picks='eeg')

# PSD comparativa en 0.5–45 Hz para algunos canales
picks_names = ['EEG Fp1', 'EEG F3', 'EEG C3', 'EEG P3']  # ajustá nombres exactos
picks = [raw.ch_names.index(n) for n in picks_names if n in raw.ch_names]

psd0 = raw.compute_psd(method='welch', fmin=0.5, fmax=45, picks=picks)
psd1 = raw_f.compute_psd(method='welch', fmin=0.5, fmax=45, picks=picks)

freqs = psd0.freqs
P0 = psd0.get_data()
P1 = psd1.get_data()

plt.figure(figsize=(10,6))
for i, n in enumerate([raw.ch_names[p] for p in picks]):
    plt.semilogy(freqs, P0[i], label=f'{n} antes', alpha=0.6)
    plt.semilogy(freqs, P1[i], label=f'{n} después', linewidth=2)
plt.title('PSD (Welch) 0.5–45 Hz — antes vs. después de filtrado PACIENTE 0 EPISODIO 0')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia (U²/Hz)')
plt.legend(ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Señal filtrada
raw_f.plot(duration=10, n_channels=len(picks))

# %% correlacion



# Correlación por banda usando Hilbert (envolvente) para TODOS los canales EEG

# ========= CONFIG =========

# Banda para el análisis
band_name = "alfa"
band_limits = (8.0, 12.0)      # (l_freq, h_freq)
# band_name, band_limits = "theta", (4.0, 8.0)
# band_name, band_limits = "beta", (12.0, 30.0)

# Filtros preliminares
apply_notch = True
notch_freqs = [50]             # 50 Hz en Argentina
bandpass_limits = (0.5, 45.0)

# Salidas
save_outputs = True
out_dir = os.path.join(os.path.dirname(pac_0_0), "salidas_hilbert")

# ========= CARGA =========
raw = mne.io.read_raw_edf(pac_0_0, preload=True, verbose=True)
print("Canales encontrados:", raw.ch_names)

# ========= ASEGURAR TIPOS EEG =========
possible_eeg_tags = ("EEG", "Fp", "F", "C", "P", "O", "T")
map_types = {}
for name in raw.ch_names:
    if any(tag in name for tag in possible_eeg_tags):
        map_types[name] = 'eeg'
if map_types:
    raw.set_channel_types(map_types)

# ========= LIMPIEZA: NOTCH + BANDPASS =========
raw_r = raw.copy().load_data()

try:
    if apply_notch and len(notch_freqs) > 0:
        raw_r.notch_filter(freqs=notch_freqs, picks='eeg')
except ValueError:
    raw_r.notch_filter(freqs=notch_freqs, picks=None)

try:
    raw_r.filter(l_freq=bandpass_limits[0], h_freq=bandpass_limits[1], picks='eeg')
except ValueError:
    raw_r.filter(l_freq=bandpass_limits[0], h_freq=bandpass_limits[1], picks=None)

# Re-referencia al promedio (mejora correlaciones)
raw_r.set_eeg_reference('average')

# ========= SELECCIÓN DE EEG Y FILTRO DE BANDA =========
picks = mne.pick_types(raw_r.info, eeg=True)
ch_names = [raw_r.ch_names[i] for i in picks]
sfreq = raw_r.info['sfreq']
print(f"Fs = {sfreq} Hz | N canales EEG = {len(picks)}")

if len(picks) == 0:
    raise RuntimeError("No hay canales EEG detectados. Revisá nombres o set_channel_types.")

# Filtramos a la banda elegida
l_band, h_band = band_limits
raw_band = raw_r.copy().filter(l_freq=l_band, h_freq=h_band, picks=picks)

# ========= HILBERT (ENVOLVENTE) =========
X_band = raw_band.get_data(picks=picks)      # (n_channels, n_samples)
env = np.abs(hilbert(X_band, axis=1))        # envolvente por canal

# ========= MATRIZ DE CORRELACIÓN =========
env_z = (env - env.mean(axis=1, keepdims=True)) / env.std(axis=1, keepdims=True)
C_env = np.corrcoef(env_z)                   # (n_channels, n_channels)

# ========= PLOT =========
plt.figure(figsize=(9, 8))
im = plt.imshow(C_env, vmin=0, vmax=1, cmap='magma')
plt.colorbar(im, label='r (correlación de envolvente)')
plt.xticks(range(len(ch_names)), ch_names, rotation=90)
plt.yticks(range(len(ch_names)), ch_names)
plt.title(f'Correlación de envolvente — Banda {band_name} ({l_band}-{h_band} Hz)')
plt.tight_layout()
if save_outputs:
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"correlacion_envolvente_{band_name}.png")
    plt.savefig(fig_path, dpi=200)
plt.show()

# ========= GUARDADO (sin pandas) =========
if save_outputs:
    # CSV de la matriz
    csv_path = os.path.join(out_dir, f"correlacion_envolvente_{band_name}.csv")
    np.savetxt(csv_path, C_env, delimiter=",")
    # TXT con nombres de canales (para interpretar filas/columnas)
    names_path = os.path.join(out_dir, f"canales_{band_name}.txt")
    with open(names_path, "w", encoding="utf-8") as f:
        for n in ch_names:
            f.write(n + "\n")
    print(f"Guardados:\n  {fig_path}\n  {csv_path}\n  {names_path}")

