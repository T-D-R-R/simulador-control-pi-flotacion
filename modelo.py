"""
modelo.py
=========
Modelo matemático dinámico del banco de flotación de tres celdas conectadas en serie
(C-1, C-2, C-3) reguladas por bombas peristálticas (P-1, P-2, P-3) y alimentadas
por la bomba P-4 desde el tanque de recirculación T-1.

Ecuaciones fundamentales implementadas (Quintanilla et al., 2023):
1. Proceso integrador con retardo (Tabla 2 y Ecuación 4):
       G_p,i(s) = Y_i(s) / U_i(s) = (k'_i * e^{-theta_i * s}) / s
   donde k'_i = -0.001018 [cm * s^-1 * lpm^-1] y theta = [7, 6, 3] s.

2. Balance dinámico de caudales en el dominio del tiempo (sección 3.1 del plan):
       dh_i(t) / dt = (Q_in,i(t) - Q_out,i(t - theta_i)) / A_efectiva,i
                    = |k'_i| * (Q_in,i(t) - Q_out,i(t - theta_i))
   donde |k'_i| = 1 / A_efectiva,i relaciona directamente el desbalance de caudales
   en [lpm] con la tasa de cambio del nivel de pulpa h_i(t) en [cm/s].

3. Acoplamiento en serie / cascada de perturbaciones (Figura 15):
       Q_in,1(t) = Q_feed(t)
       Q_in,2(t) = Q_tailings,1(t) = u_1(t)
       Q_in,3(t) = Q_tailings,2(t) = u_2(t)

4. Filtro de mediana móvil en tiempo real (medfilt1 de ventana 10 s, pág. 13)
   para atenuar el ruido de turbulencia generado por la inyección de aire (47 lpm).
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Any, Optional

from parametros import (
    CELLS_DEFAULT_PARAMS,
    DEFAULT_SIMULATION_CONFIG,
    STEP_TEST_DATA,
    lpm_a_bits_plc,
    cm_a_bits_sensor,
)
from controlador import ControladorPIFeedforward
from metricas import resumen_metricas_celda


class CeldaFlotacionIntegradora:
    """
    Representa una celda física del banco de flotación a escala de laboratorio
    con comportamiento integrador, tiempo muerto (retardo theta) y filtro de mediana.
    """

    def __init__(
        self,
        cell_id: int,
        nivel_inicial_cm: float,
        k_prime: float = -0.001018,
        theta_s: float = 7.0,
        dt_s: float = 1.0,
        q_inicial_lpm: float = 45.0,
        ventana_mediana: int = 10,
        ruido_std_cm: float = 0.0,
        h_min_cm: float = 5.0,
        h_max_cm: float = 45.0,
    ):
        self.cell_id = cell_id
        self.h_real_cm = float(nivel_inicial_cm)
        self.k_prime = float(k_prime)          # Negativo (-0.001018 cm*s^-1*lpm^-1)
        self.abs_k_prime = abs(float(k_prime)) # 1 / A_efectiva
        self.theta_s = max(0.0, float(theta_s))
        self.dt_s = float(dt_s)
        self.ventana_mediana = max(1, int(round(ventana_mediana / self.dt_s)))
        self.ruido_std_cm = float(ruido_std_cm)
        self.h_min_cm = float(h_min_cm)
        self.h_max_cm = float(h_max_cm)

        # Buffer circular para simular el retardo de transporte e^(-theta * s)
        self.pasos_retardo = max(1, int(round(self.theta_s / self.dt_s)))
        self.buffer_u = deque(
            [float(q_inicial_lpm)] * (self.pasos_retardo + 1),
            maxlen=self.pasos_retardo + 1,
        )

        # Buffer circular para el filtro de mediana móvil (medfilt1 de 10 s en MATLAB)
        self.buffer_sensor = deque(
            [float(nivel_inicial_cm)] * self.ventana_mediana,
            maxlen=self.ventana_mediana,
        )

    def leer_sensor_nivel(self, rng: Optional[np.random.Generator] = None, usar_filtro: bool = True) -> Dict[str, float]:
        """
        Simula la lectura del sensor óptico LI-1/2/3 con turbulencia de aire y
        aplica el filtro de mediana móvil de ventana 10 s (sección 'Method validation').
        """
        ruido = rng.normal(0.0, self.ruido_std_cm) if (rng is not None and self.ruido_std_cm > 0) else 0.0
        h_crudo = np.clip(self.h_real_cm + ruido, self.h_min_cm, self.h_max_cm)
        self.buffer_sensor.append(h_crudo)

        if usar_filtro and len(self.buffer_sensor) > 1:
            h_filtrado = float(np.median(self.buffer_sensor))
        else:
            h_filtrado = float(h_crudo)

        return {
            "h_real_cm": self.h_real_cm,
            "h_crudo_cm": float(h_crudo),
            "h_filtrado_cm": h_filtrado,
        }

    def avanzar_paso(self, q_in_lpm: float, u_out_lpm: float) -> float:
        """
        Actualiza el nivel de pulpa h_i(t) integrando la ecuación diferencial de balance
        con el método de Euler explícito y aplicando el retardo theta_i sobre la acción u_i:

            dh_i / dt = |k'_i| * ( Q_in,i(t) - Q_out,i(t - theta_i) )
                      = |k'_i| * Q_in,i(t) + k'_i * u_i(t - theta_i)
        """
        self.buffer_u.append(float(u_out_lpm))
        u_retardada = self.buffer_u[0] if self.theta_s > 0 else float(u_out_lpm)

        # Balance de caudales: si Q_in > u_retardada el nivel sube; si u_retardada > Q_in el nivel baja
        dh_dt = self.abs_k_prime * (float(q_in_lpm) - u_retardada)
        self.h_real_cm = float(np.clip(self.h_real_cm + dh_dt * self.dt_s, self.h_min_cm, self.h_max_cm))
        return self.h_real_cm


def generar_perfiles_escenario(
    escenario: str,
    n_steps: int,
    dt_s: float = 1.0,
    q_feed_base: float = 45.0,
    sp_iniciales: Optional[Dict[int, float]] = None,
    sp_cambio_t: float = 200.0,
    sp_deltas: Optional[Dict[int, float]] = None,
    perturbacion_t: float = 150.0,
    perturbacion_delta_lpm: float = 12.0,
    perturbacion_duracion_s: float = 200.0,
) -> Dict[str, Any]:
    """
    Genera los vectores temporales de Setpoint h_sp,i(t) y Caudal de Alimentación Q_feed(t)
    según el escenario seleccionado por el docente/usuario.

    Escenarios soportados:
    1. 'paper_fig14': Reproduce el experimento de validación de la Figura 14 del artículo
       (cambios de setpoint en t=180s y t=420s + fluctuaciones reales de caudal).
    2. 'perturbacion_caudal': Escalón/pulso de perturbación en Q_feed(t) con niveles de referencia constantes.
    3. 'cambio_setpoint': Cambio escalón en los niveles deseados h_sp,i(t) con Q_feed constante.
    4. 'operacion_estable': Operación estacionaria constante para verificar equilibrio.
    """
    t = np.arange(n_steps) * dt_s
    if sp_iniciales is None:
        sp_iniciales = {
            1: CELLS_DEFAULT_PARAMS[1].default_sp_cm,
            2: CELLS_DEFAULT_PARAMS[2].default_sp_cm,
            3: CELLS_DEFAULT_PARAMS[3].default_sp_cm,
        }
    if sp_deltas is None:
        sp_deltas = {1: -1.5, 2: -1.5, 3: -1.5}

    sp = {
        1: np.full(n_steps, float(sp_iniciales[1])),
        2: np.full(n_steps, float(sp_iniciales[2])),
        3: np.full(n_steps, float(sp_iniciales[3])),
    }
    q_feed = np.full(n_steps, float(q_feed_base))

    if escenario == "paper_fig14":
        # Reproducción fiel de los perfiles de Setpoint de la Figura 14 (600 s):
        # - Celda 1: SP = 30 cm (0-180s), baja a 29 cm (180-600s)
        # - Celda 2: SP = 34 cm (0-180s), baja a 33 cm (180-420s), baja a 32.5 cm (420-600s)
        # - Celda 3: SP = 33 cm (0-180s), baja a 32 cm (180-420s), baja a 31.5 cm (420-600s)
        sp[1][t >= 180.0] = sp_iniciales[1] - 1.0
        sp[2][(t >= 180.0) & (t < 420.0)] = sp_iniciales[2] - 1.0
        sp[2][t >= 420.0] = sp_iniciales[2] - 1.5
        sp[3][(t >= 180.0) & (t < 420.0)] = sp_iniciales[3] - 1.0
        sp[3][t >= 420.0] = sp_iniciales[3] - 1.5

        # Perturbaciones de caudal de entrada (Fig. 14 y 15):
        # Variaciones en el bombeo desde el tanque de recirculación T-1 y cambios de carga
        onda_recirculacion = (
            12.0 * np.sin(2.0 * np.pi * t / 180.0)
            + 8.0 * np.sin(2.0 * np.pi * t / 90.0)
        )
        pulsos = np.zeros(n_steps)
        pulsos[(t >= 80.0) & (t <= 150.0)] += 18.0
        pulsos[(t >= 260.0) & (t <= 340.0)] -= 16.0
        pulsos[(t >= 430.0) & (t <= 510.0)] += 15.0
        q_feed = np.clip(q_feed_base + onda_recirculacion + pulsos, 5.0, 88.0)

    elif escenario == "perturbacion_caudal":
        mask = (t >= perturbacion_t) & (t <= (perturbacion_t + perturbacion_duracion_s))
        q_feed[mask] = np.clip(q_feed_base + perturbacion_delta_lpm, 0.0, 95.0)
        # Segunda perturbación en sentido opuesto para evaluar rechazo bidireccional
        t2 = perturbacion_t + perturbacion_duracion_s + 60.0
        mask2 = (t >= t2) & (t <= (t2 + 120.0))
        q_feed[mask2] = np.clip(q_feed_base - 0.7 * perturbacion_delta_lpm, 0.0, 95.0)

    elif escenario == "cambio_setpoint":
        for cid in [1, 2, 3]:
            sp[cid][t >= sp_cambio_t] = sp_iniciales[cid] + sp_deltas[cid]

    elif escenario == "operacion_estable":
        # Mantiene Q_feed y Setpoints constantes durante toda la corrida
        pass

    return {"t": t, "sp": sp, "q_feed": q_feed}


def simular_banco_flotacion(
    usar_feedforward: bool = False,
    duracion_s: int = 600,
    dt_s: float = 1.0,
    escenario: str = "paper_fig14",
    parametros_celdas: Optional[Dict[int, Dict[str, float]]] = None,
    q_feed_base: float = 45.0,
    sp_iniciales: Optional[Dict[int, float]] = None,
    sp_cambio_t: float = 200.0,
    sp_deltas: Optional[Dict[int, float]] = None,
    perturbacion_t: float = 150.0,
    perturbacion_delta_lpm: float = 12.0,
    perturbacion_duracion_s: float = 200.0,
    incluir_ruido: bool = True,
    usar_filtro_mediana: bool = True,
    ruido_std_cm: float = 0.06,
    ventana_mediana_s: int = 10,
    anti_windup: bool = True,
    semilla: int = 42,
) -> Dict[str, Any]:
    """
    Ejecuta la simulación dinámica paso a paso de las 3 celdas de flotación en serie.

    En cada instante k (paso dt_s):
    1. Calcula el caudal de entrada a cada celda (acoplamiento en cascada Fig. 15):
       - Celda 1 recibe Q_feed(t).
       - Celda 2 recibe Q_tailings,1(t) de la Celda 1.
       - Celda 3 recibe Q_tailings,2(t) de la Celda 2.
    2. Lee el nivel de pulpa con el sensor óptico simulado y aplica el filtro medfilt1 (10 s).
    3. Calcula el error e_i(t) = h_sp,i(t) - h_i(t) y la variación de perturbación Delta_Q_d,i(t).
    4. Evalúa la ley de control PI o PI + Feedforward con saturación de bombas y anti-windup.
    5. Convierte caudales y niveles a registros de 16 bits del PLC (0 - 32,767).
    6. Integra el balance diferencial con tiempo muerto theta_i para obtener h_i(t + dt).
    """
    rng = np.random.default_rng(semilla)
    n_steps = int(round(duracion_s / dt_s)) + 1

    perfiles = generar_perfiles_escenario(
        escenario=escenario,
        n_steps=n_steps,
        dt_s=dt_s,
        q_feed_base=q_feed_base,
        sp_iniciales=sp_iniciales,
        sp_cambio_t=sp_cambio_t,
        sp_deltas=sp_deltas,
        perturbacion_t=perturbacion_t,
        perturbacion_delta_lpm=perturbacion_delta_lpm,
        perturbacion_duracion_s=perturbacion_duracion_s,
    )
    t_vec = perfiles["t"]
    sp_vec = perfiles["sp"]
    q_feed_vec = perfiles["q_feed"]

    # Construir configuración por celda (usando Tabla 2 por defecto si no se especifica)
    cfg_celdas = {}
    for cid in [1, 2, 3]:
        base = CELLS_DEFAULT_PARAMS[cid]
        custom = parametros_celdas.get(cid, {}) if parametros_celdas else {}
        cfg_celdas[cid] = {
            "k_prime": custom.get("k_prime", base.k_prime),
            "theta": custom.get("theta", base.theta),
            "kc": custom.get("kc", base.kc_final),
            "tau_i": custom.get("tau_i", base.ti_final),
            "k_ff": custom.get("k_ff", base.k_ff),
            "u_min": custom.get("u_min", base.u_min_lpm),
            "u_max": custom.get("u_max", base.u_max_lpm),
            "h0": custom.get("h0", float(sp_vec[cid][0])),
        }

    # Instanciar las 3 celdas físicas y sus 3 controladores PI/FF
    celdas: Dict[int, CeldaFlotacionIntegradora] = {}
    controladores: Dict[int, ControladorPIFeedforward] = {}

    for cid in [1, 2, 3]:
        c = cfg_celdas[cid]
        celdas[cid] = CeldaFlotacionIntegradora(
            cell_id=cid,
            nivel_inicial_cm=c["h0"],
            k_prime=c["k_prime"],
            theta_s=c["theta"],
            dt_s=dt_s,
            q_inicial_lpm=q_feed_base,
            ventana_mediana=ventana_mediana_s,
            ruido_std_cm=ruido_std_cm if incluir_ruido else 0.0,
        )
        controladores[cid] = ControladorPIFeedforward(
            cell_id=cid,
            kc=c["kc"],
            tau_i=c["tau_i"],
            k_ff=c["k_ff"],
            usar_feedforward=usar_feedforward,
            u_min=c["u_min"],
            u_max=c["u_max"],
            u_bias=q_feed_base,
            anti_windup=anti_windup,
        )

    # Registros históricos para graficación y exportación CSV
    historial: Dict[str, List[float]] = {"time_s": []}
    for cid in [1, 2, 3]:
        for campo in [
            f"sp_{cid}_cm",
            f"h_real_{cid}_cm",
            f"h_sensor_raw_{cid}_cm",
            f"h_filtered_{cid}_cm",
            f"error_{cid}_cm",
            f"q_in_{cid}_lpm",
            f"u_tailings_{cid}_lpm",
            f"u_unsat_{cid}_lpm",
            f"p_term_{cid}",
            f"i_term_{cid}",
            f"ff_term_{cid}",
            f"plc_pump_{cid}_bits",
            f"plc_sensor_{cid}_bits",
        ]:
            historial[campo] = []
    historial["q_feed_lpm"] = []
    historial["plc_feed_pump_bits"] = []

    pump_keys = {1: "tailings_pump_P1", 2: "tailings_pump_P2", 3: "tailings_pump_P3"}

    # Bucle principal de simulación en tiempo discreto (while loop de 1 s como en pág. 13)
    for k in range(n_steps):
        t_actual = float(t_vec[k])
        q_feed_actual = float(q_feed_vec[k])

        historial["time_s"].append(t_actual)
        historial["q_feed_lpm"].append(q_feed_actual)
        historial["plc_feed_pump_bits"].append(lpm_a_bits_plc(q_feed_actual, "feed_pump_P4"))

        # El caudal de entrada a la Celda 1 es Q_feed; para C2 y C3 es la descarga de la celda previa
        q_in_celda = q_feed_actual

        for cid in [1, 2, 3]:
            sp_actual = float(sp_vec[cid][k])

            # 1. Lectura del sensor óptico + filtro de mediana medfilt1
            lectura = celdas[cid].leer_sensor_nivel(rng=rng, usar_filtro=usar_filtro_mediana)
            h_medido = lectura["h_filtrado_cm"]

            # 2. Cálculo de la ley de control PI / PI + Feedforward
            estado_ctrl = controladores[cid].calcular_accion_control(
                h_sp=sp_actual,
                h_medido=h_medido,
                q_in_actual=q_in_celda,
                dt=dt_s,
            )
            u_bomba_lpm = estado_ctrl.u_saturada

            # 3. Conversión a señales enteras de 16 bits del PLC (0 - 32,767)
            bits_bomba = lpm_a_bits_plc(u_bomba_lpm, pump_keys[cid])
            bits_sensor = cm_a_bits_sensor(h_medido, cid)

            # 4. Guardar variables del paso actual
            historial[f"sp_{cid}_cm"].append(sp_actual)
            historial[f"h_real_{cid}_cm"].append(lectura["h_real_cm"])
            historial[f"h_sensor_raw_{cid}_cm"].append(lectura["h_crudo_cm"])
            historial[f"h_filtered_{cid}_cm"].append(h_medido)
            historial[f"error_{cid}_cm"].append(estado_ctrl.error)
            historial[f"q_in_{cid}_lpm"].append(q_in_celda)
            historial[f"u_tailings_{cid}_lpm"].append(u_bomba_lpm)
            historial[f"u_unsat_{cid}_lpm"].append(estado_ctrl.u_sin_saturar)
            historial[f"p_term_{cid}"].append(estado_ctrl.termino_p)
            historial[f"i_term_{cid}"].append(estado_ctrl.termino_i)
            historial[f"ff_term_{cid}"].append(estado_ctrl.termino_ff)
            historial[f"plc_pump_{cid}_bits"].append(bits_bomba)
            historial[f"plc_sensor_{cid}_bits"].append(bits_sensor)

            # 5. Integrar ecuación diferencial de nivel para el siguiente paso k+1
            celdas[cid].avanzar_paso(q_in_lpm=q_in_celda, u_out_lpm=u_bomba_lpm)

            # 6. Conexión en serie exacta (Fig. 15): La descarga de la celda i alimenta a la celda i+1
            #    Q_in,2(t) = Q_tailings,1(t)  y  Q_in,3(t) = Q_tailings,2(t)
            if cid < 3:
                q_in_celda = float(u_bomba_lpm)

    df = pd.DataFrame(historial)

    # Calcular métricas de desempeño (IAE, SSE, Error máx, Error final, TV) por celda
    metricas_celdas = {}
    for cid in [1, 2, 3]:
        metricas_celdas[cid] = resumen_metricas_celda(
            errores=df[f"error_{cid}_cm"].to_numpy(),
            u_signal=df[f"u_tailings_{cid}_lpm"].to_numpy(),
            dt=dt_s,
        )

    return {
        "dataframe": df,
        "metricas": metricas_celdas,
        "config_celdas": cfg_celdas,
        "usar_feedforward": usar_feedforward,
        "escenario": escenario,
        "dt_s": dt_s,
        "duracion_s": duracion_s,
    }


def simular_respuesta_escalon_lazo_abierto(duracion_s: int = 120, dt_s: float = 1.0) -> pd.DataFrame:
    """
    Reproduce las pruebas de escalón en lazo abierto de la Figura 12 del artículo
    para identificar experimentalmente la ganancia integradora k' y el retardo theta
    en cada una de las 3 celdas.
    """
    t_vec = np.arange(0, duracion_s + dt_s, dt_s)
    data = {"time_s": t_vec}

    for cid in [1, 2, 3]:
        info = STEP_TEST_DATA[cid]
        k_prime = CELLS_DEFAULT_PARAMS[cid].k_prime
        theta = info["theta_s"]
        t_step = info["t_start_s"]
        h0 = info["h_start_cm"]
        delta_u = info["delta_u_lpm"]

        u_vec = np.where(t_vec >= t_step, delta_u, 0.0)
        h_vec = np.full_like(t_vec, h0, dtype=float)

        for idx, t_val in enumerate(t_vec):
            t_efectivo = t_val - (t_step + theta)
            if t_efectivo > 0:
                h_vec[idx] = h0 + k_prime * delta_u * t_efectivo

        data[f"u_step_{cid}_lpm"] = u_vec
        data[f"h_resp_{cid}_cm"] = h_vec

    return pd.DataFrame(data)
