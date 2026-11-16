"""
controlador.py
==============
Implementación matemática de los controladores:
1. Controlador Proporcional-Integral (PI) con retroalimentación (Feedback)
   según la Ecuación (1) del artículo (con tau_d = 0).
2. Estrategia combinada Feedback + Feedforward (Prealimentación)
   según las Ecuaciones (7) y (8) del artículo.
3. Restricciones físicas de las bombas peristálticas (saturación min/max)
   y protección Anti-Windup del término integral (clamping condicional).
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class EstadoControlador:
    """Almacena el estado instantáneo del controlador para auditoría y graficación."""
    error: float
    termino_p: float
    termino_i: float
    termino_ff: float
    delta_q_in: float
    u_sin_saturar: float
    u_saturada: float
    saturado: bool


class ControladorPIFeedforward:
    """
    Controlador PI + Feedforward para el control regulatorio de nivel en una celda
    de flotación mediante bomba peristáltica.

    Ecuación matemática implementada (Ecuaciones 7 y 8 de Quintanilla et al., 2023):
        e_i(t) = h_sp,i(t) - h_i(t)
        Delta_Q_d,i(t) = Q_d,i(t) - Q_d,i(t - 1)
        u_i(t) = u_bias,i + K_c,i * ( e_i(t) + (1 / tau_i,i) * int_0^t e_i(tau) d_tau )
                 + K_ff,i * Delta_Q_d,i(t)

    Nota sobre signos físicos (Tabla 2 del artículo):
        - En este proceso integrador, la variable manipulada u_i(t) es el caudal de
          descarga (colas/tailings) de la bomba peristáltica.
        - Al aumentar u_i(t), el nivel de pulpa h_i(t) disminuye (ganancia k' < 0).
        - Por ello, la ganancia del controlador K_c es negativa (ej. -62, -76, -85 lpm/cm).
        - Si el nivel real supera la referencia (h_i > h_sp -> e_i < 0), el producto
          K_c * e_i resulta positivo, incrementando el caudal de bombeo para bajar el nivel.
    """

    def __init__(
        self,
        cell_id: int,
        kc: float,
        tau_i: float,
        k_ff: float = 0.0,
        usar_feedforward: bool = False,
        u_min: float = 0.0,
        u_max: float = 95.0,
        u_bias: float = 45.0,
        anti_windup: bool = True,
    ):
        self.cell_id = cell_id
        self.kc = float(kc)
        self.tau_i = float(tau_i)
        self.k_ff = float(k_ff)
        self.usar_feedforward = bool(usar_feedforward)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.u_bias = float(u_bias)
        self.anti_windup = bool(anti_windup)

        # Estado interno del integrador y memoria de perturbación Q_d(t-1)
        self.integral_error: float = 0.0
        self.q_in_prev: float = float(u_bias)
        self.inicializado: bool = False

    def reiniciar(self, u_bias: float = None, q_in_inicial: float = None) -> None:
        """Reinicia el acumulador integral y el historial de perturbación."""
        if u_bias is not None:
            self.u_bias = float(u_bias)
        self.integral_error = 0.0
        self.q_in_prev = float(q_in_inicial) if q_in_inicial is not None else self.u_bias
        self.inicializado = False

    def calcular_accion_control(
        self,
        h_sp: float,
        h_medido: float,
        q_in_actual: float,
        dt: float = 1.0,
    ) -> EstadoControlador:
        """
        Calcula la señal de control u_i(t) [lpm] para un paso de muestreo dt [s].

        Parámetros:
            h_sp: Nivel deseado (setpoint) y^sp [cm]
            h_medido: Nivel de pulpa actual medido (y filtrado) y(t) [cm]
            q_in_actual: Caudal de entrada medido (perturbación Q_d,i(t)) [lpm]
            dt: Paso de tiempo de muestreo [s]

        Retorna:
            Objeto EstadoControlador con el desglose completo de términos y señal saturada.
        """
        if not self.inicializado:
            self.q_in_prev = q_in_actual
            self.inicializado = True

        # 1. Cálculo del error de nivel: e_i(t) = y_sp(t) - y(t)
        error = h_sp - h_medido

        # 2. Término Proporcional: P = K_c * e(t)
        termino_p = self.kc * error

        # 3. Candidato de actualización del término Integral: int e(tau) d_tau
        nuevo_integral_error = self.integral_error + (error * dt)
        if self.tau_i > 1e-6:
            termino_i_candidato = (self.kc / self.tau_i) * nuevo_integral_error
        else:
            termino_i_candidato = 0.0

        # 4. Término Feedforward (Ecuaciones 7 y 8 del artículo):
        #    Delta_Q_d,i(t) = Q_d,i(t) - Q_d,i(t-1)
        #    Acumula la variación de la perturbación medida para anticipar el cambio
        #    en el caudal de entrada antes de que se desvíe el nivel de pulpa.
        delta_q_in = q_in_actual - self.q_in_prev
        self.q_in_prev = q_in_actual

        if self.usar_feedforward:
            # Acumulación de Delta_Q_d(t) respecto al punto de operación nominal (u_bias)
            # y acción anticipativa directa según Eq. (7): + K_ff,i * Delta_Q_d,i
            termino_ff = self.k_ff * ((q_in_actual - self.u_bias) + delta_q_in)
        else:
            termino_ff = 0.0

        # 5. Señal total antes de límites de la bomba peristáltica
        u_sin_saturar = self.u_bias + termino_p + termino_i_candidato + termino_ff

        # 6. Aplicación de restricciones físicas de la bomba (Saturación + Anti-Windup)
        if u_sin_saturar > self.u_max:
            u_saturada = self.u_max
            saturado = True
            # Clamping Anti-Windup: solo acumula integral si el error ayuda a salir de la saturación
            if not self.anti_windup or (self.kc * error < 0):
                self.integral_error = nuevo_integral_error
        elif u_sin_saturar < self.u_min:
            u_saturada = self.u_min
            saturado = True
            if not self.anti_windup or (self.kc * error > 0):
                self.integral_error = nuevo_integral_error
        else:
            u_saturada = u_sin_saturar
            saturado = False
            self.integral_error = nuevo_integral_error

        # Recalcular el término integral efectivo tras anti-windup
        termino_i = (self.kc / self.tau_i) * self.integral_error if self.tau_i > 1e-6 else 0.0

        return EstadoControlador(
            error=error,
            termino_p=termino_p,
            termino_i=termino_i,
            termino_ff=termino_ff,
            delta_q_in=delta_q_in,
            u_sin_saturar=u_sin_saturar,
            u_saturada=u_saturada,
            saturado=saturado,
        )
