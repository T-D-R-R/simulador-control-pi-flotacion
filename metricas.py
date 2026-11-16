"""
metricas.py
===========
Cálculo de indicadores cuantitativos de desempeño de control:
1. IAE (Integral Absolute Error) - Ecuación (9) del artículo (pág. 13).
2. SSE (Sum Squared Error) - Mencionado en la pág. 2 del artículo.
3. Error máximo absoluto (|e|_max).
4. Error estacionario final (|e|_final).
5. Variación Total de Control (TV - Total Variation) para evaluar el esfuerzo
   y variabilidad en las bombas peristálticas discutido en la pág. 14.
"""

import numpy as np
from typing import Dict, Any


def calcular_iae(errores: np.ndarray, dt: float = 1.0) -> float:
    """
    Calcula la Integral del Valor Absoluto del Error (IAE) según la Ecuación (9):
        IAE_i = int_0^t |e_i(tau)| d_tau
    Unidades: [cm * s]
    """
    errores_arr = np.asarray(errores, dtype=float)
    return float(np.sum(np.abs(errores_arr)) * dt)


def calcular_sse(errores: np.ndarray, dt: float = 1.0) -> float:
    """
    Calcula la Suma/Integral de Errores al Cuadrado (SSE / ISE) mencionada en la pág. 2:
        SSE_i = sum(e_i^2 * dt)
    Unidades: [cm^2 * s]
    """
    errores_arr = np.asarray(errores, dtype=float)
    return float(np.sum(np.square(errores_arr)) * dt)


def calcular_error_maximo(errores: np.ndarray) -> float:
    """
    Calcula la desviación máxima absoluta respecto al nivel de referencia (Setpoint).
    Unidades: [cm]
    """
    errores_arr = np.asarray(errores, dtype=float)
    if errores_arr.size == 0:
        return 0.0
    return float(np.max(np.abs(errores_arr)))


def calcular_error_final(errores: np.ndarray, ventana_final: int = 10) -> float:
    """
    Calcula el promedio del error absoluto en los últimos segundos de la corrida
    para verificar el estado estacionario.
    Unidades: [cm]
    """
    errores_arr = np.asarray(errores, dtype=float)
    if errores_arr.size == 0:
        return 0.0
    n = min(ventana_final, errores_arr.size)
    return float(np.mean(np.abs(errores_arr[-n:])))


def calcular_variacion_control(u_signal: np.ndarray) -> float:
    """
    Calcula la Variación Total (Total Variation - TV) de la variable manipulada u(t):
        TV = sum |u(k) - u(k-1)|
    Permite cuantificar lo discutido en la pág. 14 del artículo: el control Feedforward
    presenta mayor actividad en la bomba peristáltica para rechazar perturbaciones
    rápidamente antes de que desvíen el nivel.
    Unidades: [lpm]
    """
    u_arr = np.asarray(u_signal, dtype=float)
    if u_arr.size < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(u_arr))))


def resumen_metricas_celda(
    errores: np.ndarray,
    u_signal: np.ndarray,
    dt: float = 1.0,
) -> Dict[str, float]:
    """
    Genera el diccionario completo de indicadores de desempeño para una celda.
    """
    return {
        "IAE_cm_s": round(calcular_iae(errores, dt), 2),
        "SSE_cm2_s": round(calcular_sse(errores, dt), 2),
        "Error_Max_cm": round(calcular_error_maximo(errores), 3),
        "Error_Final_cm": round(calcular_error_final(errores), 4),
        "Variacion_Control_lpm": round(calcular_variacion_control(u_signal), 2),
        "Caudal_Promedio_lpm": round(float(np.mean(u_signal)), 2),
        "Caudal_Max_lpm": round(float(np.max(u_signal)), 2),
    }


def comparar_estrategias(
    metricas_fb: Dict[int, Dict[str, float]],
    metricas_ff: Dict[int, Dict[str, float]],
) -> Dict[int, Dict[str, Any]]:
    """
    Compara cuantitativamente el desempeño entre PI (Feedback solo) y
    PI + Feedforward para las 3 celdas, calculando la reducción de IAE y SSE.
    """
    comparacion = {}
    for cell_id in [1, 2, 3]:
        iae_fb = metricas_fb[cell_id]["IAE_cm_s"]
        iae_ff = metricas_ff[cell_id]["IAE_cm_s"]
        mejora_abs_iae = iae_fb - iae_ff
        mejora_pct_iae = (mejora_abs_iae / iae_fb * 100.0) if iae_fb > 1e-6 else 0.0

        sse_fb = metricas_fb[cell_id]["SSE_cm2_s"]
        sse_ff = metricas_ff[cell_id]["SSE_cm2_s"]
        mejora_pct_sse = ((sse_fb - sse_ff) / sse_fb * 100.0) if sse_fb > 1e-6 else 0.0

        comparacion[cell_id] = {
            "IAE_Feedback": iae_fb,
            "IAE_Feedforward": iae_ff,
            "Reduccion_IAE_cm_s": round(mejora_abs_iae, 2),
            "Mejora_IAE_pct": round(mejora_pct_iae, 1),
            "SSE_Feedback": sse_fb,
            "SSE_Feedforward": sse_ff,
            "Mejora_SSE_pct": round(mejora_pct_sse, 1),
            "Error_Max_FB": metricas_fb[cell_id]["Error_Max_cm"],
            "Error_Max_FF": metricas_ff[cell_id]["Error_Max_cm"],
        }
    return comparacion
