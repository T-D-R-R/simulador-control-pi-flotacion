"""
simular_local.py
================
Script de validación y ejecución directa en consola / IDE local (VS Code, Spyder, PyCharm)
para que el docente pueda verificar los modelos matemáticos, la sintonización SIMC y
la comparativa entre Control PI (Feedback) y Control PI + Feedforward sin necesidad
de abrir el navegador web.

Uso desde terminal:
    python simular_local.py          (imprime tablas y abre gráficos de Matplotlib)
    python simular_local.py --no-gui (solo imprime validación matemática y exporta CSV)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from parametros import (
    CELLS_DEFAULT_PARAMS,
    PAPER_REPORTED_IAE,
    calcular_sintonizacion_simc,
)
from modelo import simular_banco_flotacion, simular_respuesta_escalon_lazo_abierto
from metricas import comparar_estrategias


def imprimir_verificacion_simc() -> None:
    print("=" * 80)
    print("1. VERIFICACIÓN DE MODELOS MATEMÁTICOS Y SINTONIZACIÓN SIMC (Tabla 2 del Paper)")
    print("=" * 80)
    header = (
        f"{'Celda':<8} | {'k\' [cm/s/lpm]':<15} | {'theta [s]':<10} | "
        f"{'Kc SIMC':<10} | {'Ti SIMC':<10} | {'Kc Final':<10} | {'Ti Final':<10} | {'K_ff':<6}"
    )
    print(header)
    print("-" * len(header))

    for cid in [1, 2, 3]:
        p = CELLS_DEFAULT_PARAMS[cid]
        simc = calcular_sintonizacion_simc(p.k_prime, p.theta, p.tau_c)
        print(
            f"Celda {cid:<2} | {p.k_prime:<15.6f} | {p.theta:<10.1f} | "
            f"{simc['Kc']:<10.1f} | {p.ti_initial:<10.1f} | "
            f"{p.kc_final:<10.1f} | {p.ti_final:<10.1f} | {p.k_ff:<6.1f}"
        )
    print()


def ejecutar_comparativa_local(mostrar_graficos: bool = True) -> None:
    imprimir_verificacion_simc()

    print("=" * 80)
    print("2. EJECUTANDO SIMULACIÓN DINÁMICA (600 s, dt = 1 s) - PI vs. PI + FEEDFORWARD")
    print("=" * 80)

    res_fb = simular_banco_flotacion(usar_feedforward=False, escenario="paper_fig14", semilla=42)
    res_ff = simular_banco_flotacion(usar_feedforward=True, escenario="paper_fig14", semilla=42)

    comp = comparar_estrategias(res_fb["metricas"], res_ff["metricas"])

    header = (
        f"{'Celda':<8} | {'IAE PI (FB)':<14} | {'IAE PI+FF':<14} | "
        f"{'Mejora IAE (%)':<16} | {'SSE PI (FB)':<14} | {'SSE PI+FF':<14} | {'Paper IAE (FB->FF)'}"
    )
    print(header)
    print("-" * len(header))

    for cid in [1, 2, 3]:
        c = comp[cid]
        p_fb = PAPER_REPORTED_IAE["feedback_only"][cid]
        p_ff = PAPER_REPORTED_IAE["feedback_feedforward"][cid]
        print(
            f"Celda {cid:<2} | {c['IAE_Feedback']:<14.2f} | {c['IAE_Feedforward']:<14.2f} | "
            f"{c['Mejora_IAE_pct']:<15.1f}% | {c['SSE_Feedback']:<14.2f} | {c['SSE_Feedforward']:<14.2f} | "
            f"{p_fb} -> {p_ff} cm*s"
        )

    # Exportar resultados a CSV local para revisión del docente
    df_fb = res_fb["dataframe"].copy()
    df_ff = res_ff["dataframe"].copy()
    df_fb.to_csv("resultados_simulacion_PI_feedback.csv", index=False)
    df_ff.to_csv("resultados_simulacion_PI_feedforward.csv", index=False)
    print("\n[OK] Archivos CSV exportados exitosamente:")
    print("     - resultados_simulacion_PI_feedback.csv")
    print("     - resultados_simulacion_PI_feedforward.csv")

    if mostrar_graficos:
        graficar_figura_14(df_fb, df_ff, res_fb["metricas"], res_ff["metricas"])


def graficar_figura_14(df_fb: pd.DataFrame, df_ff: pd.DataFrame, met_fb: dict, met_ff: dict) -> None:
    """
    Genera en Matplotlib la reproducción exacta de la Figura 14 del artículo:
    Columna izquierda: Feedback only (PI)
    Columna derecha: Feedback + Feedforward (PI + FF)
    """
    fig, axes = plt.subplots(6, 2, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "Simulación de Control de Nivel en Banco de Flotación de 3 Celdas (Quintanilla et al., 2023 - Fig. 14)",
        fontsize=13,
        fontweight="bold",
    )

    for idx, cid in enumerate([1, 2, 3]):
        row_lvl = idx * 2
        row_flow = idx * 2 + 1

        # --- Columna Izquierda: Feedback Only ---
        ax_l_lvl = axes[row_lvl, 0]
        ax_l_lvl.plot(df_fb["time_s"], df_fb[f"h_filtered_{cid}_cm"], color="blue", lw=1.5, label="Nivel medido (h_p)")
        ax_l_lvl.plot(df_fb["time_s"], df_fb[f"sp_{cid}_cm"], color="red", lw=1.5, ls="--", label="Setpoint")
        ax_l_lvl.set_ylabel(f"h_p C{cid} [cm]")
        ax_l_lvl.set_title(f"Feedback only - Cell {cid} (IAE = {met_fb[cid]['IAE_cm_s']:.1f} cm·s)", fontsize=10)
        ax_l_lvl.grid(True, alpha=0.3)
        if idx == 0:
            ax_l_lvl.legend(loc="upper right", fontsize=8)

        ax_l_flw = axes[row_flow, 0]
        ax_l_flw.plot(df_fb["time_s"], df_fb[f"u_tailings_{cid}_lpm"], color="black", lw=1.2, label="Manipulada (Q_tailings)")
        ax_l_flw.plot(df_fb["time_s"], df_fb[f"q_in_{cid}_lpm"], color="deepskyblue", lw=1.2, alpha=0.85, label="Perturbación (Q_in)")
        ax_l_flw.set_ylabel(f"Q C{cid} [lpm]")
        ax_l_flw.grid(True, alpha=0.3)
        if idx == 0:
            ax_l_flw.legend(loc="upper right", fontsize=8)

        # --- Columna Derecha: Feedback + Feedforward ---
        ax_r_lvl = axes[row_lvl, 1]
        ax_r_lvl.plot(df_ff["time_s"], df_ff[f"h_filtered_{cid}_cm"], color="blue", lw=1.5, label="Nivel medido (h_p)")
        ax_r_lvl.plot(df_ff["time_s"], df_ff[f"sp_{cid}_cm"], color="red", lw=1.5, ls="--", label="Setpoint")
        ax_r_lvl.set_ylabel(f"h_p C{cid} [cm]")
        ax_r_lvl.set_title(f"Feedback + Feedforward - Cell {cid} (IAE = {met_ff[cid]['IAE_cm_s']:.1f} cm·s)", fontsize=10)
        ax_r_lvl.grid(True, alpha=0.3)

        ax_r_flw = axes[row_flow, 1]
        ax_r_flw.plot(df_ff["time_s"], df_ff[f"u_tailings_{cid}_lpm"], color="black", lw=1.2, label="Manipulada (Q_tailings)")
        ax_r_flw.plot(df_ff["time_s"], df_ff[f"q_in_{cid}_lpm"], color="deepskyblue", lw=1.2, alpha=0.85, label="Perturbación (Q_in)")
        ax_r_flw.set_ylabel(f"Q C{cid} [lpm]")
        ax_r_flw.grid(True, alpha=0.3)

    axes[5, 0].set_xlabel("Tiempo [s]")
    axes[5, 1].set_xlabel("Tiempo [s]")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    no_gui = "--no-gui" in sys.argv
    ejecutar_comparativa_local(mostrar_graficos=not no_gui)
