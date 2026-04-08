import matplotlib.pyplot as plt

def graficos_hr(hrs, tus, sz_times):
    """
    HR vs tiempo con crisis epilépticas sombreadas.
    Función hardcodeada para 7 registros.
    Cada sz_times[i] es una LISTA de (t_onset, t_offset).
    """

    fig, axes = plt.subplots(7, 1, figsize=(12, 10), sharex=False)

    for i in range(7):
        hr = hrs[i]
        tu = tus[i]
        sz_list = sz_times[i]

        # HR
        axes[i].plot(tu, hr, color="black", linewidth=1)

        # Sombrear todas las crisis del registro
        for (t_on, t_off) in sz_list:
            axes[i].axvspan(
                t_on,
                t_off,
                color="lightgray",
                alpha=0.6
            )

        axes[i].set_ylabel("HR\n[lat/s]")
        axes[i].set_title(f"Registro {i+1}")
        axes[i].grid(True, alpha=0.4)

    axes[-1].set_xlabel("Tiempo [s]")

    plt.tight_layout()
    plt.show()