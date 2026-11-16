"""
parametros.py
=============
Parámetros físicos, experimentales, curvas de calibración (PLC 16-bit) y
constantes de sintonización SIMC reportados en el artículo:

Quintanilla, P., Navia, D., Moreno, F., Neethling, S. J., & Brito-Parada, P. R. (2023).
"A methodology to implement a closed-loop feedback-feedforward level control in a
laboratory-scale flotation bank using peristaltic pumps". MethodsX, 10, 102081.
https://doi.org/10.1016/j.mex.2023.102081
"""

from dataclasses import dataclass
from typing import Dict, Any

# =============================================================================
# 1. DIMENSIONES FÍSICAS DEL BANCO DE FLOTACIÓN (Sección "Experimental rig", Fig. 2)
# =============================================================================
CELL_LENGTH_CM: float = 30.0          # Largo de la celda [cm] (300 mm)
CELL_WIDTH_CM: float = 30.0           # Ancho de la celda [cm] (300 mm)
CELL_HEIGHT_CM: float = 45.0          # Altura total de la celda [cm] (450 mm)
CELL_AREA_CM2: float = CELL_LENGTH_CM * CELL_WIDTH_CM  # Área transversal geométrica = 900 cm^2
FEED_TANK_CAPACITY_L: float = 130.0   # Capacidad del tanque de alimentación T-1 [litros]

# Condiciones de aireación en estado estacionario (Sección "Level sensors" y "Method validation")
AIR_FLOWRATE_LPM: float = 47.0        # Caudal volumétrico de aire Q_air [lpm]
SUPERFICIAL_GAS_VELOCITY_CMS: float = 0.87  # Velocidad superficial de gas j_g = Q_air / A_cell [cm/s]


# =============================================================================
# 2. CURVAS DE CALIBRACIÓN DE INSTRUMENTOS (PLC 16-bit: 0 a 32,767 bits)
# =============================================================================
PLC_MAX_BITS: int = 32767
PLC_MIN_BITS: int = 0

# Fig. 8: Controladores de flujo másico de aire (FIC-1/2/3) -> Q_air [lpm] = m * bits
AIR_CALIBRATION = {
    "sensor_slope_lpm_per_bit": 0.0032,      # y = 0.0032 * x (Fig. 8 izquierda)
    "controller_slope_lpm_per_bit": 0.0031,  # y = 0.0031 * x (Fig. 8 derecha)
}

# Fig. 9: Bombas peristálticas Masterflex L/S (0-600 rpm) -> Q [lpm] = slope * bits
PUMP_CALIBRATION = {
    "feed_pump_P4": {
        "tag": "P-4 (Alimentación)",
        "slope_lpm_per_bit": 0.0035,         # Q_feed = 0.0035 * x
        "max_tested_bits": 28000,
        "max_flow_lpm": 0.0035 * 28000,      # ~98.0 lpm (a 32767 bits -> 114.7 lpm)
    },
    "tailings_pump_P1": {
        "tag": "P-1 (Descarga Celda 1)",
        "slope_lpm_per_bit": 0.0029,         # Q_tailings1 = 0.0029 * x
        "max_tested_bits": 28000,
        "max_flow_lpm": 0.0029 * 32767,      # ~95.0 lpm a escala completa de 16-bit
    },
    "tailings_pump_P2": {
        "tag": "P-2 (Descarga Celda 2)",
        "slope_lpm_per_bit": 0.0033,         # Q_tailings2 = 0.0033 * x
        "max_tested_bits": 28000,
        "max_flow_lpm": 0.0033 * 32767,      # ~108.1 lpm a escala completa de 16-bit
    },
    "tailings_pump_P3": {
        "tag": "P-3 (Descarga Celda 3)",
        "slope_lpm_per_bit": 0.0033,         # Q_tailings3 = 0.0033 * x
        "max_tested_bits": 28000,
        "max_flow_lpm": 0.0033 * 32767,      # ~108.1 lpm a escala completa de 16-bit
    },
}

# Fig. 11: Sensores ópticos de nivel (LI-1/2/3) -> h_p [cm] = slope * bits + intercept
LEVEL_SENSOR_CALIBRATION = {
    1: {"tag": "LI-1 (Celda 1)", "slope_cm_per_bit": -0.0018, "intercept_cm": 40.1363},
    2: {"tag": "LI-2 (Celda 2)", "slope_cm_per_bit": -0.0019, "intercept_cm": 43.8823},
    3: {"tag": "LI-3 (Celda 3)", "slope_cm_per_bit": -0.0020, "intercept_cm": 41.2930},
}


# =============================================================================
# 3. DATOS DE PRUEBA DE ESCALÓN EN LAZO ABIERTO (Figura 12 y Ecuación 5)
# =============================================================================
# k' = delta_y / (delta_t * delta_u)
STEP_TEST_DATA = {
    1: {
        "t_start_s": 30.0,
        "t_end_s": 100.0,
        "h_start_cm": 35.8,
        "h_end_cm": 32.7,
        "delta_u_lpm": 43.5,
        "theta_s": 7.0,
    },
    2: {
        "t_start_s": 30.0,
        "t_end_s": 100.0,
        "h_start_cm": 36.0,
        "h_end_cm": 32.9,
        "delta_u_lpm": 49.5,
        "theta_s": 6.0,
    },
    3: {
        "t_start_s": 30.0,
        "t_end_s": 100.0,
        "h_start_cm": 37.1,
        "h_end_cm": 34.1,
        "delta_u_lpm": 49.5,
        "theta_s": 3.0,
    },
}


# =============================================================================
# 4. PARÁMETROS DEL MODELO Y CONTROLADORES (Tabla 2 y Ecuaciones 7-8)
# =============================================================================
@dataclass
class CellParameters:
    """Parámetros dinámicos y de control para una celda de flotación i."""
    cell_id: int
    k_prime: float        # Ganancia integradora k' [cm * s^-1 * lpm^-1]
    theta: float          # Retardo de tiempo muerto (delay) [s]
    tau_c: float          # Parámetro de sintonización SIMC tau_c [s]
    kc_initial: float     # Ganancia proporcional inicial SIMC K_c^(1) [lpm * cm^-1]
    ti_initial: float     # Tiempo integral inicial SIMC tau_i^(1) [s]
    kc_final: float       # Ganancia proporcional final ajustada K_c [lpm * cm^-1]
    ti_final: float       # Tiempo integral final ajustado tau_i [s]
    k_ff: float           # Constante proporcional Feedforward K_ff [-] (Eq. 7 y 8)
    u_min_lpm: float      # Caudal mínimo de la bomba peristáltica [lpm]
    u_max_lpm: float      # Caudal máximo de la bomba peristáltica [lpm]
    default_sp_cm: float  # Nivel de referencia inicial típico (Fig. 14) [cm]


# Valores exactos reportados en la Tabla 2 y página 12 del artículo
CELLS_DEFAULT_PARAMS: Dict[int, CellParameters] = {
    1: CellParameters(
        cell_id=1,
        k_prime=-0.001018,
        theta=7.0,
        tau_c=7.0,
        kc_initial=-70.2,
        ti_initial=56.0,
        kc_final=-62.0,
        ti_final=56.0,
        k_ff=0.8,
        u_min_lpm=0.0,
        u_max_lpm=90.0,       # Basado en Fig. 14 y rango de bomba P-1
        default_sp_cm=30.0,   # Basado en Fig. 14 (Celda 1 inicia en 30 cm -> 29 cm)
    ),
    2: CellParameters(
        cell_id=2,
        k_prime=-0.001018,
        theta=6.0,
        tau_c=6.0,
        kc_initial=-81.8,
        ti_initial=48.0,
        kc_final=-76.0,
        ti_final=30.0,
        k_ff=0.8,
        u_min_lpm=0.0,
        u_max_lpm=95.0,       # Basado en Fig. 14 y rango de bomba P-2
        default_sp_cm=34.0,   # Basado en Fig. 14 (Celda 2 inicia en 34 cm -> 33 cm)
    ),
    3: CellParameters(
        cell_id=3,
        k_prime=-0.001018,
        theta=3.0,
        tau_c=8.0,
        kc_initial=-78.1,
        ti_initial=52.0,
        kc_final=-85.0,
        ti_final=44.0,
        k_ff=1.0,
        u_min_lpm=0.0,
        u_max_lpm=95.0,       # Basado en Fig. 14 y rango de bomba P-3
        default_sp_cm=33.0,   # Basado en Fig. 14 (Celda 3 inicia en 33 cm -> 32 cm)
    ),
}

# Valores de IAE experimentales reportados en la Figura 14 del artículo (para validación)
PAPER_REPORTED_IAE = {
    "feedback_only": {1: 198, 2: 325, 3: 260},
    "feedback_feedforward": {1: 164, 2: 187, 3: 221},
}

# Parámetros globales de simulación por defecto (Sección "Method validation", pág. 13)
DEFAULT_SIMULATION_CONFIG: Dict[str, Any] = {
    "duration_s": 600,             # 600 s mostrados en las gráficas de la Figura 14
    "dt_s": 1.0,                   # Tiempo de muestreo = 1 segundo (pág. 13)
    "q_feed_nominal_lpm": 45.0,    # Q_feed constante en 45 lpm (pág. 14)
    "median_filter_window_s": 10,  # Ventana de filtro medfilt1 = 10 s (pág. 13)
    "sensor_noise_std_cm": 0.08,   # Ruido óptico por burbujeo de aire (47 lpm)
}


def calcular_sintonizacion_simc(k_prime: float, theta: float, tau_c: float = None) -> Dict[str, float]:
    """
    Calcula los parámetros iniciales del controlador PI usando las reglas SIMC
    para procesos integradores con retardo (Skogestad, 2004; Ecuaciones 2, 3 y 6).

    Ecuación (2): Kc = (1 / k') * (1 / (tau_c + theta))
    Ecuación (3) y (6): tau_i = 4 * (tau_c + theta)  [o 8 * theta cuando tau_c = theta]

    Parámetros:
        k_prime: Pendiente de la respuesta integradora [cm * s^-1 * lpm^-1]
        theta: Tiempo muerto / retardo del proceso [s]
        tau_c: Constante de tiempo deseada en lazo cerrado [s]. Si es None, se usa tau_c = theta.

    Retorna:
        Diccionario con Kc, tau_i y tau_c calculados.
    """
    if tau_c is None:
        tau_c = theta

    kc = (1.0 / k_prime) * (1.0 / (tau_c + theta))
    tau_i = 4.0 * (tau_c + theta)

    return {
        "Kc": round(kc, 2),
        "tau_i": round(tau_i, 2),
        "tau_c": round(tau_c, 2),
    }


def lpm_a_bits_plc(caudal_lpm: float, pump_key: str) -> int:
    """
    Convierte un caudal en [lpm] al registro entero de 16 bits (0 - 32,767)
    enviado al módulo de salida analógica ALG392 del PLC (Fig. 9).
    """
    slope = PUMP_CALIBRATION[pump_key]["slope_lpm_per_bit"]
    bits = int(round(caudal_lpm / slope))
    return max(PLC_MIN_BITS, min(PLC_MAX_BITS, bits))


def cm_a_bits_sensor(nivel_cm: float, cell_id: int) -> int:
    """
    Convierte un nivel de pulpa en [cm] al registro entero de 16 bits (0 - 32,767)
    leído por el módulo de entrada analógica ALG223 del PLC (Fig. 11).
    """
    cal = LEVEL_SENSOR_CALIBRATION[cell_id]
    bits = int(round((nivel_cm - cal["intercept_cm"]) / cal["slope_cm_per_bit"]))
    return max(PLC_MIN_BITS, min(PLC_MAX_BITS, bits))


def bits_sensor_a_cm(bits: int, cell_id: int) -> float:
    """
    Convierte el registro entero de 16 bits del PLC a nivel de pulpa en [cm] (Fig. 11).
    """
    cal = LEVEL_SENSOR_CALIBRATION[cell_id]
    return cal["slope_cm_per_bit"] * bits + cal["intercept_cm"]
