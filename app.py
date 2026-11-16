"""
app.py
======
Aplicación Web Interactiva en Streamlit para la Simulación del Control de Nivel
(Feedback PI vs. Feedback-Feedforward PI) en un Banco de Flotación de 3 Celdas en Serie
usando Bombas Peristálticas.

Basado en el artículo:
Quintanilla, P., Navia, D., Moreno, F., Neethling, S. J., & Brito-Parada, P. R. (2023).
"A methodology to implement a closed-loop feedback-feedforward level control in a
laboratory-scale flotation bank using peristaltic pumps". MethodsX, 10, 102081.
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from parametros import (
    CELLS_DEFAULT_PARAMS,
    PUMP_CALIBRATION,
    LEVEL_SENSOR_CALIBRATION,
    STEP_TEST_DATA,
    PAPER_REPORTED_IAE,
    CELL_LENGTH_CM,
    CELL_WIDTH_CM,
    CELL_HEIGHT_CM,
    FEED_TANK_CAPACITY_L,
    AIR_FLOWRATE_LPM,
    SUPERFICIAL_GAS_VELOCITY_CMS,
    calcular_sintonizacion_simc,
)
from modelo import simular_banco_flotacion, simular_respuesta_escalon_lazo_abierto
from metricas import comparar_estrategias

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="SCADA Control PI + Feedforward | Banco de Flotación",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# ESTILOS CSS AVANZADOS (DISEÑO INDUSTRIAL SCADA / DASHBOARD MODERNO)
# =============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Banner Principal Superior */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-left: 6px solid #38bdf8;
        padding: 1.5rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .hero-title {
        color: #f8fafc !important;
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        margin: 0 0 0.4rem 0 !important;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94a3b8 !important;
        font-size: 0.92rem !important;
        margin: 0 !important;
        line-height: 1.45;
    }
    .badge-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.65rem;
    }
    .scada-badge {
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
    }
    .scada-badge-green {
        background: rgba(34, 197, 94, 0.14);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: #4ade80;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
    }

    /* Tarjetas KPI Superiores */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .kpi-delta {
        display: inline-block;
        background: rgba(34, 197, 94, 0.18);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.35);
        font-size: 0.78rem;
        font-weight: 700;
        padding: 0.18rem 0.55rem;
        border-radius: 6px;
    }
    .kpi-sub {
        color: #64748b;
        font-size: 0.76rem;
        margin-top: 0.45rem;
    }

    /* Estilizado de Pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        padding: 0 16px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Banner Superior Estilo SCADA Industrial
st.markdown(
    """
    <div class="hero-banner">
        <div>
            <div class="hero-title">⚙️ Sistema SCADA de Control Regulatorio PI + Feedforward — Banco de Flotación</div>
            <p class="hero-subtitle">
                Simulación matemática en tiempo discreto de 3 celdas de flotación en serie reguladas por bombas peristálticas Masterflex L/S.<br>
                Basado en: <b>Quintanilla, Navia, Moreno, Neethling & Brito-Parada (2023)</b> — <i>MethodsX 10, 102081</i>.
            </p>
            <div class="badge-row">
                <span class="scada-badge">PLC GE Fanuc 16-bit (0–32,767 bits)</span>
                <span class="scada-badge">Proceso Integrador: k' = -0.001018 cm·s⁻¹·lpm⁻¹</span>
                <span class="scada-badge">Retardos θ = [7s, 6s, 3s]</span>
                <span class="scada-badge">Filtro Mediana Móvil: medfilt1 (10 s)</span>
                <span class="scada-badge-green">● Motor Matemático Python Activo</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# A. PANEL LATERAL DE CONFIGURACIÓN (SIDEBAR)
# =============================================================================
with st.sidebar:
    st.markdown("## 🎛️ A. Panel de Control")

    st.markdown("### 1. Escenario de Simulación")
    escenario_opciones = {
        "paper_fig14": "1. Validación Experimental Paper (Fig. 14)",
        "perturbacion_caudal": "2. Perturbación Escalón en Alimentación (Q_feed)",
        "cambio_setpoint": "3. Seguimiento de Referencia (Cambio de Setpoint)",
        "operacion_estable": "4. Operación Estacionaria Constante",
    }
    escenario_sel = st.selectbox(
        "Seleccione el escenario de prueba:",
        options=list(escenario_opciones.keys()),
        format_func=lambda k: escenario_opciones[k],
        index=0,
    )

    modo_visualizacion = st.radio(
        "Modo de Evaluación de Control:",
        options=[
            "Comparación Paralela (PI vs. PI + Feedforward)",
            "Solo Control PI (Feedback clásico)",
            "Solo Control PI + Feedforward",
        ],
        index=0,
    )

    st.divider()

    st.markdown("### 2. Tiempo y Discretización")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        duracion_s = st.number_input("Duración [s]", min_value=120, max_value=1800, value=600, step=60)
    with col_t2:
        dt_s = st.selectbox("Muestreo Δt [s]", options=[0.5, 1.0, 2.0], index=1)

    st.divider()

    st.markdown("### 3. Alimentación y Perturbaciones")
    q_feed_base = st.slider("Caudal Alimentación Nominal Q_feed [lpm]", 20.0, 75.0, 45.0, 1.0)

    if escenario_sel == "perturbacion_caudal":
        perturbacion_delta = st.slider("Cambio de Caudal ΔQ_feed [lpm]", -25.0, 25.0, 14.0, 1.0)
        perturbacion_t = st.slider("Instante de Perturbación [s]", 30.0, float(duracion_s - 60), 120.0, 10.0)
        perturbacion_dur = st.slider("Duración de Perturbación [s]", 30.0, 300.0, 180.0, 10.0)
    else:
        perturbacion_delta = 12.0
        perturbacion_t = 150.0
        perturbacion_dur = 180.0

    st.divider()

    st.markdown("### 4. Niveles Deseados (Setpoints)")
    col_sp1, col_sp2, col_sp3 = st.columns(3)
    with col_sp1:
        sp1_init = st.number_input("SP C1 [cm]", 20.0, 40.0, 30.0, 0.5)
    with col_sp2:
        sp2_init = st.number_input("SP C2 [cm]", 20.0, 40.0, 34.0, 0.5)
    with col_sp3:
        sp3_init = st.number_input("SP C3 [cm]", 20.0, 40.0, 33.0, 0.5)

    if escenario_sel == "cambio_setpoint":
        sp_cambio_t = st.slider("Instante Cambio de SP [s]", 50.0, float(duracion_s - 50), 200.0, 10.0)
        delta_sp = st.slider("Cambio ΔSP en Celdas [cm]", -4.0, 4.0, -2.0, 0.5)
        sp_deltas = {1: delta_sp, 2: delta_sp, 3: delta_sp}
    else:
        sp_cambio_t = 200.0
        sp_deltas = {1: -1.5, 2: -1.5, 3: -1.5}

    st.divider()

    st.markdown("### 5. Parámetros PI, Feedforward y Bombas")
    preset_tuning = st.selectbox(
        "Cargar Sintonización (Tabla 2 del Paper):",
        options=[
            "Final Ajustado Paper (Tabla 2 Final)",
            "SIMC Inicial Analítico (Tabla 2 Initial)",
            "Personalizado (Editar manualmente)",
        ],
        index=0,
    )

    parametros_celdas_custom = {}
    for cid in [1, 2, 3]:
        p_def = CELLS_DEFAULT_PARAMS[cid]
        if preset_tuning == "Final Ajustado Paper (Tabla 2 Final)":
            kc_val, ti_val, kff_val = p_def.kc_final, p_def.ti_final, p_def.k_ff
            theta_val, umax_val = p_def.theta, p_def.u_max_lpm
        elif preset_tuning == "SIMC Inicial Analítico (Tabla 2 Initial)":
            kc_val, ti_val, kff_val = p_def.kc_initial, p_def.ti_initial, p_def.k_ff
            theta_val, umax_val = p_def.theta, p_def.u_max_lpm
        else:
            with st.expander(f"⚙️ Configurar Celda {cid}", expanded=(cid == 1)):
                kc_val = st.number_input(f"Kc Celda {cid} [lpm/cm]", -150.0, -5.0, float(p_def.kc_final), 1.0)
                ti_val = st.number_input(f"τ_i Celda {cid} [s]", 5.0, 150.0, float(p_def.ti_final), 1.0)
                kff_val = st.slider(f"K_ff Celda {cid} [-]", 0.0, 1.5, float(p_def.k_ff), 0.05)
                theta_val = st.number_input(f"Retardo θ_{cid} [s]", 0.0, 25.0, float(p_def.theta), 1.0)
                umax_val = st.number_input(f"Caudal Máx Bomba P-{cid} [lpm]", 50.0, 115.0, float(p_def.u_max_lpm), 5.0)

        parametros_celdas_custom[cid] = {
            "k_prime": p_def.k_prime,
            "theta": theta_val,
            "kc": kc_val,
            "tau_i": ti_val,
            "k_ff": kff_val,
            "u_min": 0.0,
            "u_max": umax_val,
        }

    st.divider()

    st.markdown("### 6. Instrumentación y Filtrado")
    incluir_ruido = st.checkbox("Simular turbulencia de aire (47 lpm)", value=True)
    usar_filtro_mediana = st.checkbox("Filtro mediana móvil (medfilt1 = 10 s)", value=True)
    anti_windup = st.checkbox("Protección Anti-Windup en Bombas", value=True)


# =============================================================================
# EJECUCIÓN DEL MOTOR MATEMÁTICO
# =============================================================================
sp_iniciales_dict = {1: sp1_init, 2: sp2_init, 3: sp3_init}

kwargs_comunes = dict(
    duracion_s=int(duracion_s),
    dt_s=float(dt_s),
    escenario=escenario_sel,
    parametros_celdas=parametros_celdas_custom,
    q_feed_base=q_feed_base,
    sp_iniciales=sp_iniciales_dict,
    sp_cambio_t=sp_cambio_t,
    sp_deltas=sp_deltas,
    perturbacion_t=perturbacion_t,
    perturbacion_delta_lpm=perturbacion_delta,
    perturbacion_duracion_s=perturbacion_dur,
    incluir_ruido=incluir_ruido,
    usar_filtro_mediana=usar_filtro_mediana,
    anti_windup=anti_windup,
    semilla=42,
)

res_fb = simular_banco_flotacion(usar_feedforward=False, **kwargs_comunes)
res_ff = simular_banco_flotacion(usar_feedforward=True, **kwargs_comunes)
df_fb = res_fb["dataframe"]
df_ff = res_ff["dataframe"]
comp_metricas = comparar_estrategias(res_fb["metricas"], res_ff["metricas"])

# Tarjetas KPI Rápidas en la parte superior
c1_m, c2_m, c3_m = comp_metricas[1], comp_metricas[2], comp_metricas[3]
iae_total_fb = c1_m["IAE_Feedback"] + c2_m["IAE_Feedback"] + c3_m["IAE_Feedback"]
iae_total_ff = c1_m["IAE_Feedforward"] + c2_m["IAE_Feedforward"] + c3_m["IAE_Feedforward"]
mejora_global = ((iae_total_fb - iae_total_ff) / iae_total_fb * 100.0) if iae_total_fb > 0 else 0.0

st.markdown(
    f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Celda 1 (C-1) • IAE PI + Feedforward</div>
            <div class="kpi-value">{c1_m['IAE_Feedforward']:.1f} <span style="font-size:0.9rem;font-weight:500;color:#94a3b8;">cm·s</span></div>
            <span class="kpi-delta">▼ {c1_m['Mejora_IAE_pct']:.1f}% vs Solo PI ({c1_m['IAE_Feedback']:.1f} cm·s)</span>
            <div class="kpi-sub">Kc = {parametros_celdas_custom[1]['kc']} lpm/cm | τi = {parametros_celdas_custom[1]['tau_i']} s | Kff = {parametros_celdas_custom[1]['k_ff']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Celda 2 (C-2) • IAE PI + Feedforward</div>
            <div class="kpi-value">{c2_m['IAE_Feedforward']:.1f} <span style="font-size:0.9rem;font-weight:500;color:#94a3b8;">cm·s</span></div>
            <span class="kpi-delta">▼ {c2_m['Mejora_IAE_pct']:.1f}% vs Solo PI ({c2_m['IAE_Feedback']:.1f} cm·s)</span>
            <div class="kpi-sub">Kc = {parametros_celdas_custom[2]['kc']} lpm/cm | τi = {parametros_celdas_custom[2]['tau_i']} s | Kff = {parametros_celdas_custom[2]['k_ff']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Celda 3 (C-3) • IAE PI + Feedforward</div>
            <div class="kpi-value">{c3_m['IAE_Feedforward']:.1f} <span style="font-size:0.9rem;font-weight:500;color:#94a3b8;">cm·s</span></div>
            <span class="kpi-delta">▼ {c3_m['Mejora_IAE_pct']:.1f}% vs Solo PI ({c3_m['IAE_Feedback']:.1f} cm·s)</span>
            <div class="kpi-sub">Kc = {parametros_celdas_custom[3]['kc']} lpm/cm | τi = {parametros_celdas_custom[3]['tau_i']} s | Kff = {parametros_celdas_custom[3]['k_ff']}</div>
        </div>
        <div class="kpi-card" style="border-color:#38bdf8;">
            <div class="kpi-label">Desempeño Global del Banco (3 Celdas)</div>
            <div class="kpi-value">{iae_total_ff:.1f} <span style="font-size:0.9rem;font-weight:500;color:#94a3b8;">cm·s</span></div>
            <span class="kpi-delta">★ Reducción Total de Error: {mejora_global:.1f}%</span>
            <div class="kpi-sub">IAE Total Solo PI: {iae_total_fb:.1f} cm·s</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# COMPONENTE HTML/CSS/SVG DEL DIAGRAMA P&ID INTERACTIVO (SCADA ANIMADO)
# =============================================================================
def renderizar_scada_pid_html(df_actual: pd.DataFrame, paso_idx: int, titulo_modo: str) -> None:
    """
    Renderiza un panel SCADA P&ID en HTML5 + CSS3 + SVG con animaciones de flujo,
    burbujeo de aire, indicadores digitales de PLC (16-bit) y niveles dinámicos.
    """
    row = df_actual.iloc[paso_idx]
    t_val = row["time_s"]
    q_feed = row["q_feed_lpm"]
    b_feed = int(row["plc_feed_pump_bits"])

    h1, h2, h3 = row["h_filtered_1_cm"], row["h_filtered_2_cm"], row["h_filtered_3_cm"]
    sp1, sp2, sp3 = row["sp_1_cm"], row["sp_2_cm"], row["sp_3_cm"]
    e1, e2, e3 = row["error_1_cm"], row["error_2_cm"], row["error_3_cm"]
    u1, u2, u3 = row["u_tailings_1_lpm"], row["u_tailings_2_lpm"], row["u_tailings_3_lpm"]
    b1, b2, b3 = int(row["plc_pump_1_bits"]), int(row["plc_pump_2_bits"]), int(row["plc_pump_3_bits"])
    sb1, sb2, sb3 = int(row["plc_sensor_1_bits"]), int(row["plc_sensor_2_bits"]), int(row["plc_sensor_3_bits"])

    # Escalar alturas visuales dentro de cada celda (altura física = 45 cm -> 135 px)
    def nivel_px(h_cm: float) -> int:
        pct = np.clip(h_cm / 45.0, 0.12, 0.94)
        return int(round(pct * 135))

    p1_h, p2_h, p3_h = nivel_px(h1), nivel_px(h2), nivel_px(h3)
    sp1_y = 85 + 135 - nivel_px(sp1)
    sp2_y = 120 + 135 - nivel_px(sp2)
    sp3_y = 155 + 135 - nivel_px(sp3)

    html_scada = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
        body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: 'Inter', sans-serif;
            color: #f8fafc;
        }}
        .scada-wrapper {{
            background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 14px 18px;
            box-shadow: inset 0 0 30px rgba(0,0,0,0.45);
        }}
        .scada-topbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #334155;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }}
        .scada-title {{
            font-size: 15px;
            font-weight: 700;
            color: #e2e8f0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .live-dot {{
            width: 10px;
            height: 10px;
            background: #22c55e;
            border-radius: 50%;
            box-shadow: 0 0 8px #22c55e;
            animation: pulse 1.5s infinite;
        }}
        @keyframes pulse {{
            0% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.45; transform: scale(1.15); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}
        .flow-pipe {{
            stroke-dasharray: 10 6;
            animation: dash 1.1s linear infinite;
        }}
        @keyframes dash {{
            to {{ stroke-dashoffset: -32; }}
        }}
        .scada-footer {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 8px;
        }}
        .plc-box {{
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 8px 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11.5px;
        }}
        .plc-box-title {{
            color: #38bdf8;
            font-weight: 700;
            margin-bottom: 4px;
            font-size: 12px;
        }}
        .plc-row {{
            display: flex;
            justify-content: space-between;
            color: #cbd5e1;
            margin: 2px 0;
        }}
    </style>
    </head>
    <body>
    <div class="scada-wrapper">
        <div class="scada-topbar">
            <div class="scada-title">
                <div class="live-dot"></div>
                <span>DIAGRAMA P&ID INTERACTIVO (FIG. 4 DEL PAPER) — MODO: <b style="color:#38bdf8;">{titulo_modo.upper()}</b></span>
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:13px; background:#0284c7; padding:4px 12px; border-radius:6px; font-weight:700;">
                ⏱️ Tiempo: t = {t_val:.0f} s  |  Aire: {AIR_FLOWRATE_LPM:.0f} lpm (jg = {SUPERFICIAL_GAS_VELOCITY_CMS} cm/s)
            </div>
        </div>

        <svg viewBox="0 0 1020 330" width="100%" height="315" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="pulpGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.85"/>
                    <stop offset="100%" stop-color="#1d4ed8" stop-opacity="0.95"/>
                </linearGradient>
                <linearGradient id="frothGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#f8fafc" stop-opacity="0.95"/>
                    <stop offset="100%" stop-color="#bae6fd" stop-opacity="0.85"/>
                </linearGradient>
                <linearGradient id="tankGrad" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stop-color="#1e293b"/>
                    <stop offset="50%" stop-color="#334155"/>
                    <stop offset="100%" stop-color="#1e293b"/>
                </linearGradient>
            </defs>

            <!-- Bus PLC Superior -->
            <rect x="15" y="6" width="990" height="28" rx="6" fill="#090d16" stroke="#38bdf8" stroke-width="1"/>
            <text x="510" y="24" text-anchor="middle" fill="#38bdf8" font-family="JetBrains Mono, monospace" font-size="11.5" font-weight="bold">
                BUS PLC GE FANUC (Módulo Sensores IC693ALG223 [AI] ⇄ MATLAB DDE ⇄ Módulo Actuadores IC693ALG392 [AQ 4-20 mA])
            </text>

            <!-- ==================== TANQUE DE ALIMENTACIÓN T-1 ==================== -->
            <rect x="25" y="185" width="115" height="120" rx="6" fill="url(#tankGrad)" stroke="#94a3b8" stroke-width="2"/>
            <rect x="28" y="218" width="109" height="84" rx="4" fill="url(#pulpGrad)"/>
            <text x="82" y="205" text-anchor="middle" fill="#f8fafc" font-size="12" font-weight="bold">TANQUE T-1</text>
            <text x="82" y="255" text-anchor="middle" fill="#ffffff" font-size="12" font-weight="bold">130 Litros</text>
            <text x="82" y="273" text-anchor="middle" fill="#e0f2fe" font-size="10">Agua + Reactivos</text>
            <text x="82" y="288" text-anchor="middle" fill="#bae6fd" font-size="9.5">(MIBC + Goma Xantana)</text>

            <!-- Tubería Alimentación P-4 -> C-1 -->
            <path d="M 140 270 L 188 270 L 188 95 L 245 95" fill="none" stroke="#1e293b" stroke-width="7"/>
            <path d="M 140 270 L 188 270 L 188 95 L 245 95" fill="none" stroke="#38bdf8" stroke-width="3.5" class="flow-pipe"/>
            <!-- Bomba P-4 -->
            <circle cx="188" cy="185" r="20" fill="#0f172a" stroke="#facc15" stroke-width="3"/>
            <polygon points="182,177 197,185 182,193" fill="#facc15"/>
            <text x="188" y="156" text-anchor="middle" fill="#fde047" font-size="11.5" font-weight="bold">P-4 (Feed)</text>
            <text x="188" y="220" text-anchor="middle" fill="#38bdf8" font-family="JetBrains Mono, monospace" font-size="11" font-weight="bold">{q_feed:.1f} lpm</text>

            <!-- ==================== CELDA 1 (C-1) ==================== -->
            <!-- Sensor LI-1 y Controlador LIC-1 -->
            <line x1="312" y1="34" x2="312" y2="85" stroke="#38bdf8" stroke-width="1.5" stroke-dasharray="3,3"/>
            <rect x="267" y="45" width="90" height="24" rx="5" fill="#0f172a" stroke="#38bdf8" stroke-width="1.5"/>
            <text x="312" y="61" text-anchor="middle" fill="#38bdf8" font-family="JetBrains Mono, monospace" font-size="11" font-weight="bold">LIC-1 / LI-1</text>

            <!-- Cuerpo Celda 1 -->
            <rect x="245" y="85" width="135" height="135" rx="4" fill="#0f172a" stroke="#cbd5e1" stroke-width="2.5"/>
            <!-- Capa de espuma (Froth) -->
            <rect x="248" y="{85 + 135 - p1_h - 11}" width="129" height="11" fill="url(#frothGrad)"/>
            <!-- Nivel de Pulpa h1 -->
            <rect x="248" y="{85 + 135 - p1_h}" width="129" height="{p1_h}" fill="url(#pulpGrad)"/>
            <!-- Burbujas de aire frits -->
            <circle cx="275" cy="205" r="3" fill="#e0f2fe" opacity="0.7"/>
            <circle cx="312" cy="195" r="3.5" fill="#e0f2fe" opacity="0.7"/>
            <circle cx="348" cy="202" r="3" fill="#e0f2fe" opacity="0.7"/>
            <!-- Línea de Setpoint SP1 -->
            <line x1="240" y1="{sp1_y}" x2="385" y2="{sp1_y}" stroke="#f43f5e" stroke-width="2.5" stroke-dasharray="6,3"/>
            <!-- Etiquetas Celda 1 -->
            <text x="312" y="138" text-anchor="middle" fill="#ffffff" font-size="13" font-weight="bold">CELDA 1</text>
            <text x="312" y="158" text-anchor="middle" fill="#ffffff" font-family="JetBrains Mono, monospace" font-size="13" font-weight="bold">h1 = {h1:.2f} cm</text>
            <text x="312" y="175" text-anchor="middle" fill="#fecdd3" font-family="JetBrains Mono, monospace" font-size="11">SP = {sp1:.1f} cm</text>
            <text x="312" y="214" text-anchor="middle" fill="#bae6fd" font-size="9.5">Frits 20µm (FIC-1: 47 lpm)</text>

            <!-- Tubería y Bomba P-1 (C-1 -> C-2) -->
            <path d="M 380 205 L 440 205 L 440 130 L 490 130" fill="none" stroke="#1e293b" stroke-width="7"/>
            <path d="M 380 205 L 440 205 L 440 130 L 490 130" fill="none" stroke="#4ade80" stroke-width="3.5" class="flow-pipe"/>
            <circle cx="435" cy="205" r="20" fill="#0f172a" stroke="#4ade80" stroke-width="3"/>
            <polygon points="429,197 444,205 429,213" fill="#4ade80"/>
            <text x="435" y="240" text-anchor="middle" fill="#4ade80" font-size="11.5" font-weight="bold">P-1: {u1:.1f} lpm</text>
            <text x="435" y="254" text-anchor="middle" fill="#94a3b8" font-family="JetBrains Mono, monospace" font-size="10">({b1} bits)</text>

            <!-- ==================== CELDA 2 (C-2) ==================== -->
            <line x1="557" y1="34" x2="557" y2="120" stroke="#38bdf8" stroke-width="1.5" stroke-dasharray="3,3"/>
            <rect x="512" y="80" width="90" height="24" rx="5" fill="#0f172a" stroke="#38bdf8" stroke-width="1.5"/>
            <text x="557" y="96" text-anchor="middle" fill="#38bdf8" font-family="JetBrains Mono, monospace" font-size="11" font-weight="bold">LIC-2 / LI-2</text>

            <rect x="490" y="120" width="135" height="135" rx="4" fill="#0f172a" stroke="#cbd5e1" stroke-width="2.5"/>
            <rect x="493" y="{120 + 135 - p2_h - 11}" width="129" height="11" fill="url(#frothGrad)"/>
            <rect x="493" y="{120 + 135 - p2_h}" width="129" height="{p2_h}" fill="url(#pulpGrad)"/>
            <circle cx="520" cy="240" r="3" fill="#e0f2fe" opacity="0.7"/>
            <circle cx="557" cy="230" r="3.5" fill="#e0f2fe" opacity="0.7"/>
            <circle cx="593" cy="238" r="3" fill="#e0f2fe" opacity="0.7"/>
            <line x1="485" y1="{sp2_y}" x2="630" y2="{sp2_y}" stroke="#f43f5e" stroke-width="2.5" stroke-dasharray="6,3"/>
            <text x="557" y="173" text-anchor="middle" fill="#ffffff" font-size="13" font-weight="bold">CELDA 2</text>
            <text x="557" y="193" text-anchor="middle" fill="#ffffff" font-family="JetBrains Mono, monospace" font-size="13" font-weight="bold">h2 = {h2:.2f} cm</text>
            <text x="557" y="210" text-anchor="middle" fill="#fecdd3" font-family="JetBrains Mono, monospace" font-size="11">SP = {sp2:.1f} cm</text>
            <text x="557" y="249" text-anchor="middle" fill="#bae6fd" font-size="9.5">Frits 20µm (FIC-2: 47 lpm)</text>

            <!-- Tubería y Bomba P-2 (C-2 -> C-3) -->
            <path d="M 625 240 L 685 240 L 685 165 L 735 165" fill="none" stroke="#1e293b" stroke-width="7"/>
            <path d="M 625 240 L 685 240 L 685 165 L 735 165" fill="none" stroke="#4ade80" stroke-width="3.5" class="flow-pipe"/>
            <circle cx="680" cy="240" r="20" fill="#0f172a" stroke="#4ade80" stroke-width="3"/>
            <polygon points="674,232 689,240 674,248" fill="#4ade80"/>
            <text x="680" y="275" text-anchor="middle" fill="#4ade80" font-size="11.5" font-weight="bold">P-2: {u2:.1f} lpm</text>
            <text x="680" y="289" text-anchor="middle" fill="#94a3b8" font-family="JetBrains Mono, monospace" font-size="10">({b2} bits)</text>

            <!-- ==================== CELDA 3 (C-3) ==================== -->
            <line x1="802" y1="34" x2="802" y2="155" stroke="#38bdf8" stroke-width="1.5" stroke-dasharray="3,3"/>
            <rect x="757" y="115" width="90" height="24" rx="5" fill="#0f172a" stroke="#38bdf8" stroke-width="1.5"/>
            <text x="802" y="131" text-anchor="middle" fill="#38bdf8" font-family="JetBrains Mono, monospace" font-size="11" font-weight="bold">LIC-3 / LI-3</text>

            <rect x="735" y="155" width="135" height="135" rx="4" fill="#0f172a" stroke="#cbd5e1" stroke-width="2.5"/>
            <rect x="738" y="{155 + 135 - p3_h - 11}" width="129" height="11" fill="url(#frothGrad)"/>
            <rect x="738" y="{155 + 135 - p3_h}" width="129" height="{p3_h}" fill="url(#pulpGrad)"/>
            <circle cx="765" cy="275" r="3" fill="#e0f2fe" opacity="0.7"/>
            <circle cx="802" cy="265" r="3.5" fill="#e0f2fe" opacity="0.7"/>
            <circle cx="838" cy="273" r="3" fill="#e0f2fe" opacity="0.7"/>
            <line x1="730" y1="{sp3_y}" x2="875" y2="{sp3_y}" stroke="#f43f5e" stroke-width="2.5" stroke-dasharray="6,3"/>
            <text x="802" y="208" text-anchor="middle" fill="#ffffff" font-size="13" font-weight="bold">CELDA 3</text>
            <text x="802" y="228" text-anchor="middle" fill="#ffffff" font-family="JetBrains Mono, monospace" font-size="13" font-weight="bold">h3 = {h3:.2f} cm</text>
            <text x="802" y="245" text-anchor="middle" fill="#fecdd3" font-family="JetBrains Mono, monospace" font-size="11">SP = {sp3:.1f} cm</text>
            <text x="802" y="284" text-anchor="middle" fill="#bae6fd" font-size="9.5">Frits 20µm (FIC-3: 47 lpm)</text>

            <!-- Bomba P-3 y Retorno al Tanque T-1 -->
            <path d="M 870 275 L 965 275 L 965 318 L 82 318 L 82 305" fill="none" stroke="#334155" stroke-width="5"/>
            <path d="M 870 275 L 965 275 L 965 318 L 82 318 L 82 305" fill="none" stroke="#38bdf8" stroke-width="2.5" class="flow-pipe"/>
            <circle cx="925" cy="275" r="20" fill="#0f172a" stroke="#4ade80" stroke-width="3"/>
            <polygon points="919,267 934,275 919,283" fill="#4ade80"/>
            <text x="925" y="244" text-anchor="middle" fill="#4ade80" font-size="11.5" font-weight="bold">P-3: {u3:.1f} lpm</text>
            <text x="925" y="230" text-anchor="middle" fill="#94a3b8" font-family="JetBrains Mono, monospace" font-size="10">({b3} bits)</text>
        </svg>

        <!-- Telemetría Digital PLC en Tiempo Real -->
        <div class="scada-footer">
            <div class="plc-box">
                <div class="plc-box-title">📡 ALIMENTACIÓN (P-4 / T-1)</div>
                <div class="plc-row"><span>Caudal Q_feed:</span><b>{q_feed:.2f} lpm</b></div>
                <div class="plc-row"><span>PLC AQ004:</span><b>{b_feed} bits</b></div>
                <div class="plc-row"><span>Tanque T-1:</span><b>130.0 L</b></div>
            </div>
            <div class="plc-box">
                <div class="plc-box-title">🎛️ TELEMETRÍA CELDA 1</div>
                <div class="plc-row"><span>LI-1 (AI001):</span><b>{h1:.2f} cm ({sb1} b)</b></div>
                <div class="plc-row"><span>Error e1(t):</span><b style="color:{'#4ade80' if abs(e1)<0.5 else '#facc15'}">{e1:+.2f} cm</b></div>
                <div class="plc-row"><span>P-1 (AQ001):</span><b>{u1:.2f} lpm ({b1} b)</b></div>
            </div>
            <div class="plc-box">
                <div class="plc-box-title">🎛️ TELEMETRÍA CELDA 2</div>
                <div class="plc-row"><span>LI-2 (AI002):</span><b>{h2:.2f} cm ({sb2} b)</b></div>
                <div class="plc-row"><span>Error e2(t):</span><b style="color:{'#4ade80' if abs(e2)<0.5 else '#facc15'}">{e2:+.2f} cm</b></div>
                <div class="plc-row"><span>P-2 (AQ002):</span><b>{u2:.2f} lpm ({b2} b)</b></div>
            </div>
            <div class="plc-box">
                <div class="plc-box-title">🎛️ TELEMETRÍA CELDA 3</div>
                <div class="plc-row"><span>LI-3 (AI003):</span><b>{h3:.2f} cm ({sb3} b)</b></div>
                <div class="plc-row"><span>Error e3(t):</span><b style="color:{'#4ade80' if abs(e3)<0.5 else '#facc15'}">{e3:+.2f} cm</b></div>
                <div class="plc-row"><span>P-3 (AQ003):</span><b>{u3:.2f} lpm ({b3} b)</b></div>
            </div>
        </div>
    </div>
    </body>
    </html>
    """
    components.html(html_scada, height=565, scrolling=False)


# =============================================================================
# FUNCIONES DE GRAFICACIÓN INTERACTIVA (PLOTLY)
# =============================================================================
def crear_grafico_comparativo_fig14(df_fb: pd.DataFrame, df_ff: pd.DataFrame, met_fb: dict, met_ff: dict) -> go.Figure:
    """
    Crea la gráfica interactiva en Plotly replicando la Figura 14 del artículo:
    Columna izquierda: Feedback Only (PI)
    Columna derecha: Feedback + Feedforward (PI + FF)
    """
    fig = make_subplots(
        rows=6,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.045,
        horizontal_spacing=0.06,
        subplot_titles=(
            f"<b>Feedback only - Cell 1</b> (IAE = {met_fb[1]['IAE_cm_s']:.1f} cm·s)",
            f"<b>Feedback + Feedforward - Cell 1</b> (IAE = {met_ff[1]['IAE_cm_s']:.1f} cm·s)",
            "Caudales Celda 1 [lpm]",
            "Caudales Celda 1 [lpm]",
            f"<b>Feedback only - Cell 2</b> (IAE = {met_fb[2]['IAE_cm_s']:.1f} cm·s)",
            f"<b>Feedback + Feedforward - Cell 2</b> (IAE = {met_ff[2]['IAE_cm_s']:.1f} cm·s)",
            "Caudales Celda 2 [lpm]",
            "Caudales Celda 2 [lpm]",
            f"<b>Feedback only - Cell 3</b> (IAE = {met_fb[3]['IAE_cm_s']:.1f} cm·s)",
            f"<b>Feedback + Feedforward - Cell 3</b> (IAE = {met_ff[3]['IAE_cm_s']:.1f} cm·s)",
            "Caudales Celda 3 [lpm]",
            "Caudales Celda 3 [lpm]",
        ),
    )

    for idx, cid in enumerate([1, 2, 3]):
        r_lvl = idx * 2 + 1
        r_flw = idx * 2 + 2
        show_leg = idx == 0

        # --- Feedback Only (Columna 1) ---
        fig.add_trace(
            go.Scatter(
                x=df_fb["time_s"], y=df_fb[f"h_filtered_{cid}_cm"],
                name="Nivel medido (h_p)", line=dict(color="#2563eb", width=2.2),
                showlegend=show_leg, legendgroup="lvl"
            ),
            row=r_lvl, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_fb["time_s"], y=df_fb[f"sp_{cid}_cm"],
                name="Setpoint (y^sp)", line=dict(color="#dc2626", width=2, dash="dash"),
                showlegend=show_leg, legendgroup="sp"
            ),
            row=r_lvl, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_fb["time_s"], y=df_fb[f"u_tailings_{cid}_lpm"],
                name="Variable Manipulada (Q_tailings)", line=dict(color="#0f172a", width=1.6),
                showlegend=show_leg, legendgroup="u"
            ),
            row=r_flw, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df_fb["time_s"], y=df_fb[f"q_in_{cid}_lpm"],
                name="Perturbación Entrada (Q_in)", line=dict(color="#06b6d4", width=1.6),
                showlegend=show_leg, legendgroup="d"
            ),
            row=r_flw, col=1,
        )

        # --- Feedback + Feedforward (Columna 2) ---
        fig.add_trace(
            go.Scatter(
                x=df_ff["time_s"], y=df_ff[f"h_filtered_{cid}_cm"],
                name="Nivel medido (h_p)", line=dict(color="#2563eb", width=2.2),
                showlegend=False, legendgroup="lvl"
            ),
            row=r_lvl, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df_ff["time_s"], y=df_ff[f"sp_{cid}_cm"],
                name="Setpoint (y^sp)", line=dict(color="#dc2626", width=2, dash="dash"),
                showlegend=False, legendgroup="sp"
            ),
            row=r_lvl, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df_ff["time_s"], y=df_ff[f"u_tailings_{cid}_lpm"],
                name="Variable Manipulada (Q_tailings)", line=dict(color="#0f172a", width=1.6),
                showlegend=False, legendgroup="u"
            ),
            row=r_flw, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df_ff["time_s"], y=df_ff[f"q_in_{cid}_lpm"],
                name="Perturbación Entrada (Q_in)", line=dict(color="#06b6d4", width=1.6),
                showlegend=False, legendgroup="d"
            ),
            row=r_flw, col=2,
        )

        fig.update_yaxes(title_text=f"h_p C{cid} [cm]", row=r_lvl, col=1)
        fig.update_yaxes(title_text=f"h_p C{cid} [cm]", row=r_lvl, col=2)
        fig.update_yaxes(title_text=f"Q C{cid} [lpm]", row=r_flw, col=1)
        fig.update_yaxes(title_text=f"Q C{cid} [lpm]", row=r_flw, col=2)

    fig.update_xaxes(title_text="Tiempo [s]", row=6, col=1)
    fig.update_xaxes(title_text="Tiempo [s]", row=6, col=2)
    fig.update_layout(
        height=1050,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=30, t=70, b=40),
    )
    return fig


def crear_grafico_individual(df: pd.DataFrame, met: dict, titulo_estrategia: str) -> go.Figure:
    """Crea la vista detallada de niveles, caudales, errores y acción PI/FF para una sola estrategia."""
    fig = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.07,
        subplot_titles=(
            f"Celda 1 — Nivel h_p vs Setpoint (IAE = {met[1]['IAE_cm_s']:.1f} cm·s)",
            "Celda 1 — Caudal Entrada (Q_feed) vs Bomba P-1 (Q_tailings,1)",
            f"Celda 2 — Nivel h_p vs Setpoint (IAE = {met[2]['IAE_cm_s']:.1f} cm·s)",
            "Celda 2 — Caudal Entrada (Q_tailings,1) vs Bomba P-2 (Q_tailings,2)",
            f"Celda 3 — Nivel h_p vs Setpoint (IAE = {met[3]['IAE_cm_s']:.1f} cm·s)",
            "Celda 3 — Caudal Entrada (Q_tailings,2) vs Bomba P-3 (Q_tailings,3)",
        ),
    )

    for cid in [1, 2, 3]:
        show_leg = cid == 1
        fig.add_trace(
            go.Scatter(
                x=df["time_s"], y=df[f"h_filtered_{cid}_cm"],
                name="Nivel Filtrado (medfilt1)", line=dict(color="#2563eb", width=2.2),
                showlegend=show_leg
            ),
            row=cid, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["time_s"], y=df[f"sp_{cid}_cm"],
                name="Setpoint (Referencia)", line=dict(color="#dc2626", width=2, dash="dash"),
                showlegend=show_leg
            ),
            row=cid, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["time_s"], y=df[f"u_tailings_{cid}_lpm"],
                name="Bomba Peristáltica (Q_out)", line=dict(color="#0f172a", width=1.8),
                showlegend=show_leg
            ),
            row=cid, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=df["time_s"], y=df[f"q_in_{cid}_lpm"],
                name="Caudal Entrada (Q_in)", line=dict(color="#06b6d4", width=1.8),
                showlegend=show_leg
            ),
            row=cid, col=2,
        )
        fig.update_yaxes(title_text="Nivel [cm]", row=cid, col=1)
        fig.update_yaxes(title_text="Caudal [lpm]", row=cid, col=2)

    fig.update_xaxes(title_text="Tiempo [s]", row=3, col=1)
    fig.update_xaxes(title_text="Tiempo [s]", row=3, col=2)
    fig.update_layout(
        title=f"Respuesta Dinámica del Sistema — {titulo_estrategia}",
        height=760,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
    )
    return fig


# =============================================================================
# PESTAÑAS PRINCIPALES DE LA APLICACIÓN (Secciones B, C, D, E, F del Plan)
# =============================================================================
tabs = st.tabs([
    "📐 B & C. Diagrama SCADA P&ID y Simulación",
    "📊 D. Indicadores de Desempeño (IAE, SSE y Esfuerzo)",
    "🔬 Calibración PLC (16-bit) y Sintonización SIMC",
    "📖 E. Sección Técnica: Modelos Matemáticos",
    "💾 F. Exportación de Datos (CSV) y Ejecución Local",
])

# -----------------------------------------------------------------------------
# PESTAÑA 1: DIAGRAMA P&ID Y RESULTADOS GRÁFICOS
# -----------------------------------------------------------------------------
with tabs[0]:
    col_slider, col_selector_pid = st.columns([3, 1])
    with col_slider:
        instante_t = st.slider(
            "⏱️ Línea de Tiempo Interactiva — Deslice para inspeccionar niveles, caudales y bits del PLC en el instante t [s]:",
            min_value=0,
            max_value=int(duracion_s),
            value=min(220, int(duracion_s)),
            step=int(max(1, round(dt_s))),
        )
    with col_selector_pid:
        ver_pid_ff = st.selectbox(
            "Estrategia mostrada en el SCADA P&ID:",
            options=["PI + Feedforward", "Solo PI (Feedback)"],
            index=0 if "Solo Control PI (Feedback" not in modo_visualizacion else 1,
        )

    paso_idx = min(int(round(instante_t / dt_s)), len(df_ff) - 1)
    df_pid = df_ff if ver_pid_ff == "PI + Feedforward" else df_fb
    renderizar_scada_pid_html(df_pid, paso_idx, ver_pid_ff)

    st.subheader("📈 C. Gráficos Interactivos de Respuesta Dinámica (Figura 14)")
    if modo_visualizacion == "Comparación Paralela (PI vs. PI + Feedforward)":
        st.info(
            "💡 **Comparación directa (Figura 14 del artículo):** A la izquierda se muestra el control **PI clásico (Feedback)** "
            "y a la derecha el control **PI + Feedforward** bajo exactamente la misma perturbación y condiciones iniciales."
        )
        fig_comp = crear_grafico_comparativo_fig14(df_fb, df_ff, res_fb["metricas"], res_ff["metricas"])
        st.plotly_chart(fig_comp, use_container_width=True)
    elif modo_visualizacion == "Solo Control PI (Feedback clásico)":
        fig_fb = crear_grafico_individual(df_fb, res_fb["metricas"], "Control PI Clásico (Feedback Only)")
        st.plotly_chart(fig_fb, use_container_width=True)
    else:
        fig_ff = crear_grafico_individual(df_ff, res_ff["metricas"], "Control PI + Feedforward")
        st.plotly_chart(fig_ff, use_container_width=True)

    with st.expander("🔍 Ver Evolución del Error de Control e(t) en las 3 Celdas", expanded=False):
        fig_err = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Error Celda 1: e_1(t) [cm]", "Error Celda 2: e_2(t) [cm]", "Error Celda 3: e_3(t) [cm]"),
        )
        for cid in [1, 2, 3]:
            fig_err.add_trace(
                go.Scatter(
                    x=df_fb["time_s"], y=df_fb[f"error_{cid}_cm"],
                    name="Error PI (Feedback)", line=dict(color="#ef4444", width=1.6),
                    showlegend=(cid == 1)
                ),
                row=1, col=cid,
            )
            fig_err.add_trace(
                go.Scatter(
                    x=df_ff["time_s"], y=df_ff[f"error_{cid}_cm"],
                    name="Error PI + Feedforward", line=dict(color="#16a34a", width=1.8),
                    showlegend=(cid == 1)
                ),
                row=1, col=cid,
            )
            fig_err.update_xaxes(title_text="Tiempo [s]", row=1, col=cid)
            fig_err.update_yaxes(title_text="Error e(t) [cm]", row=1, col=cid)
        fig_err.update_layout(height=360, template="plotly_white")
        st.plotly_chart(fig_err, use_container_width=True)


# -----------------------------------------------------------------------------
# PESTAÑA 2: INDICADORES DE DESEMPEÑO (IAE, SSE, Error Máx, TV)
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("D. Evaluación Cuantitativa del Desempeño de Control")
    st.markdown(
        """
        El artículo evalúa la mejora al añadir la estrategia **Feedforward** sobre el lazo **Feedback PI**
        utilizando la **Integral del Error Absoluto (IAE, Ecuación 9)** y la **Suma de Errores al Cuadrado (SSE, pág. 2)**:
        """
    )

    filas_tabla = []
    for cid in [1, 2, 3]:
        m_fb = res_fb["metricas"][cid]
        m_ff = res_ff["metricas"][cid]
        c_d = comp_metricas[cid]
        filas_tabla.append({
            "Celda": f"Celda {cid} (C-{cid})",
            "IAE Solo PI [cm·s]": m_fb["IAE_cm_s"],
            "IAE PI + FF [cm·s]": m_ff["IAE_cm_s"],
            "Mejora IAE [%]": f"{c_d['Mejora_IAE_pct']:.1f}%",
            "SSE Solo PI [cm²·s]": m_fb["SSE_cm2_s"],
            "SSE PI + FF [cm²·s]": m_ff["SSE_cm2_s"],
            "Error Máx PI [cm]": m_fb["Error_Max_cm"],
            "Error Máx PI+FF [cm]": m_ff["Error_Max_cm"],
            "Error Final PI+FF [cm]": m_ff["Error_Final_cm"],
            "Variación Bomba TV (PI) [lpm]": m_fb["Variacion_Control_lpm"],
            "Variación Bomba TV (PI+FF) [lpm]": m_ff["Variacion_Control_lpm"],
        })
    st.dataframe(pd.DataFrame(filas_tabla), use_container_width=True, hide_index=True)

    col_bar1, col_bar2 = st.columns(2)
    with col_bar1:
        fig_bar_iae = go.Figure(
            data=[
                go.Bar(
                    name="Feedback Only (Solo PI)",
                    x=["Celda 1", "Celda 2", "Celda 3"],
                    y=[comp_metricas[1]["IAE_Feedback"], comp_metricas[2]["IAE_Feedback"], comp_metricas[3]["IAE_Feedback"]],
                    marker_color="#ef4444",
                    text=[f"{comp_metricas[i]['IAE_Feedback']:.1f}" for i in [1, 2, 3]],
                    textposition="auto",
                ),
                go.Bar(
                    name="Feedback + Feedforward (PI + FF)",
                    x=["Celda 1", "Celda 2", "Celda 3"],
                    y=[comp_metricas[1]["IAE_Feedforward"], comp_metricas[2]["IAE_Feedforward"], comp_metricas[3]["IAE_Feedforward"]],
                    marker_color="#2563eb",
                    text=[f"{comp_metricas[i]['IAE_Feedforward']:.1f}" for i in [1, 2, 3]],
                    textposition="auto",
                ),
            ]
        )
        fig_bar_iae.update_layout(
            title="Comparación de IAE [cm·s] (Menor es mejor)",
            barmode="group",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig_bar_iae, use_container_width=True)

    with col_bar2:
        fig_bar_sse = go.Figure(
            data=[
                go.Bar(
                    name="SSE Solo PI [cm²·s]",
                    x=["Celda 1", "Celda 2", "Celda 3"],
                    y=[comp_metricas[1]["SSE_Feedback"], comp_metricas[2]["SSE_Feedback"], comp_metricas[3]["SSE_Feedback"]],
                    marker_color="#f59e0b",
                    text=[f"{comp_metricas[i]['SSE_Feedback']:.1f}" for i in [1, 2, 3]],
                    textposition="auto",
                ),
                go.Bar(
                    name="SSE PI + Feedforward [cm²·s]",
                    x=["Celda 1", "Celda 2", "Celda 3"],
                    y=[comp_metricas[1]["SSE_Feedforward"], comp_metricas[2]["SSE_Feedforward"], comp_metricas[3]["SSE_Feedforward"]],
                    marker_color="#10b981",
                    text=[f"{comp_metricas[i]['SSE_Feedforward']:.1f}" for i in [1, 2, 3]],
                    textposition="auto",
                ),
            ]
        )
        fig_bar_sse.update_layout(
            title="Comparación de SSE [cm²·s] (Menor es mejor)",
            barmode="group",
            template="plotly_white",
            height=380,
        )
        st.plotly_chart(fig_bar_sse, use_container_width=True)


# -----------------------------------------------------------------------------
# PESTAÑA 3: CALIBRACIÓN DE INSTRUMENTOS Y PRUEBAS DE ESCALÓN SIMC
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("🔬 Reproducción de Curvas de Calibración PLC (16-bit) e Identificación SIMC")
    col_cal1, col_cal2 = st.columns(2)
    with col_cal1:
        bits_arr = np.linspace(0, 28000, 100)
        fig_pumps = go.Figure()
        for key, info in PUMP_CALIBRATION.items():
            fig_pumps.add_trace(
                go.Scatter(
                    x=bits_arr,
                    y=info["slope_lpm_per_bit"] * bits_arr,
                    name=f"{info['tag']} (y = {info['slope_lpm_per_bit']}x)",
                )
            )
        fig_pumps.update_layout(
            title="Fig. 9 — Curvas de Calibración de las 4 Bombas Peristálticas",
            xaxis_title="Registro PLC 16-bit [bits]",
            yaxis_title="Caudal Q [lpm]",
            template="plotly_white",
            height=370,
        )
        st.plotly_chart(fig_pumps, use_container_width=True)

    with col_cal2:
        bits_sens = np.linspace(0, 9000, 100)
        fig_sens = go.Figure()
        for cid, info in LEVEL_SENSOR_CALIBRATION.items():
            fig_sens.add_trace(
                go.Scatter(
                    x=bits_sens,
                    y=info["slope_cm_per_bit"] * bits_sens + info["intercept_cm"],
                    name=f"{info['tag']}: y = {info['slope_cm_per_bit']}x + {info['intercept_cm']}",
                )
            )
        fig_sens.update_layout(
            title="Fig. 11 — Curvas de Calibración de Sensores Ópticos de Nivel (h_p)",
            xaxis_title="Registro PLC 16-bit [bits]",
            yaxis_title="Nivel de Pulpa h_p [cm]",
            template="plotly_white",
            height=370,
        )
        st.plotly_chart(fig_sens, use_container_width=True)

    st.markdown("#### 📈 Figura 12 — Pruebas de Escalón en Lazo Abierto y Sintonización SIMC (Tabla 2)")
    df_step = simular_respuesta_escalon_lazo_abierto(duracion_s=120, dt_s=1.0)
    fig_step = make_subplots(
        rows=2, cols=3, shared_xaxes=True,
        subplot_titles=(
            "Step changes Cell 1 (h_p)", "Step changes Cell 2 (h_p)", "Step changes Cell 3 (h_p)",
            "Escalón Bomba P-1 (Δu = 43.5 lpm)", "Escalón Bomba P-2 (Δu = 49.5 lpm)", "Escalón Bomba P-3 (Δu = 49.5 lpm)"
        ),
    )
    for cid in [1, 2, 3]:
        s_info = STEP_TEST_DATA[cid]
        fig_step.add_trace(
            go.Scatter(x=df_step["time_s"], y=df_step[f"h_resp_{cid}_cm"], name=f"Nivel C{cid}", line=dict(color="#dc2626", width=2.2)),
            row=1, col=cid,
        )
        fig_step.add_trace(
            go.Scatter(
                x=[s_info["t_start_s"], s_info["t_end_s"]],
                y=[s_info["h_start_cm"], s_info["h_end_cm"]],
                mode="markers+text",
                marker=dict(color="black", size=9, symbol="star"),
                text=[f"{s_info['h_start_cm']} cm", f"{s_info['h_end_cm']} cm"],
                textposition="top right",
                showlegend=False,
            ),
            row=1, col=cid,
        )
        fig_step.add_trace(
            go.Scatter(x=df_step["time_s"], y=df_step[f"u_step_{cid}_lpm"], name=f"Δu C{cid}", line=dict(color="#0f172a", width=2)),
            row=2, col=cid,
        )
        fig_step.update_yaxes(title_text="h_p [cm]", row=1, col=cid)
        fig_step.update_yaxes(title_text="Q_tails [lpm]", row=2, col=cid)
        fig_step.update_xaxes(title_text="Time [s]", row=2, col=cid)

    fig_step.update_layout(height=520, template="plotly_white", showlegend=False)
    st.plotly_chart(fig_step, use_container_width=True)


# -----------------------------------------------------------------------------
# PESTAÑA 4: SECCIÓN TÉCNICA (Ecuaciones, Parámetros, Unidades y Supuestos)
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("📖 E. Documentación Técnica del Modelo Matemático y Controladores")

    col_eq1, col_eq2 = st.columns(2)
    with col_eq1:
        st.markdown(
            """
            ### 1. Modelo Dinámico del Proceso Integrador
            A diferencia de las celdas reguladas por válvulas de gravedad (que son procesos autorregulados), el uso de **bombas peristálticas** en la descarga convierte a cada celda en un **sistema integrador puro con retardo de tiempo muerto ($\\theta_i$)**, ya que el caudal de salida depende exclusivamente de las revoluciones por minuto (rpm) impuestas a la bomba y no de la presión hidrostática del nivel.

            #### Función de Transferencia (Tabla 2 y Ec. 4):
            $$G_{p,i}(s) = \\frac{Y_i(s)}{U_i(s)} = \\frac{k'_i \\, e^{-\\theta_i s}}{s}$$

            #### Balance Dinámico de Caudales en el Tiempo:
            Para cada celda $i \\in \\{1, 2, 3\\}$, el cambio del nivel de pulpa $h_i(t)$ viene dado por el balance volumétrico entre el caudal de entrada $Q_{\\text{in},i}(t)$ y el caudal de salida manipulado por la bomba $u_i(t) = Q_{\\text{out},i}(t)$ afectado por el retardo $\\theta_i$:
            $$\\frac{dh_i(t)}{dt} = \\frac{Q_{\\text{in},i}(t) - Q_{\\text{out},i}(t - \\theta_i)}{A_{\\text{efectiva},i}} = |k'_i| \\cdot \\Big( Q_{\\text{in},i}(t) - u_i(t - \\theta_i) \\Big)$$

            #### Acoplamiento en Serie (Cascada de Perturbaciones - Fig. 15):
            $$Q_{\\text{in},1}(t) = Q_{\\text{feed}}(t)$$
            $$Q_{\\text{in},2}(t) = Q_{\\text{tailings},1}(t) = u_1(t)$$
            $$Q_{\\text{in},3}(t) = Q_{\\text{tailings},2}(t) = u_2(t)$$
            """
        )

    with col_eq2:
        st.markdown(
            """
            ### 2. Sintonización SIMC y Ley de Control PI + Feedforward
            #### Reglas Analíticas SIMC para Procesos Integradores (Ec. 2, 3, 5 y 6):
            A partir de la respuesta al escalón en lazo abierto (Figura 12):
            $$k'_i = \\frac{\\Delta y_i}{\\Delta t \\cdot \\Delta u_i}, \\qquad K_{c,i} = \\frac{1}{k'_i} \\frac{1}{\\tau_{c,i} + \\theta_i}, \\qquad \\tau_{i,i} = 4(\\tau_{c,i} + \\theta_i)$$

            #### Controlador PI con Prealimentación (Feedforward - Ec. 7 y 8):
            $$e_i(t) = h_{\\text{sp},i}(t) - h_i(t)$$
            $$\\Delta Q_{d,i}(t) = Q_{\\text{in},i}(t) - Q_{\\text{in},i}(t - 1)$$
            $$u_i(t) = u_{\\text{bias},i} + K_{c,i}\\left( e_i(t) + \\frac{1}{\\tau_{i,i}} \\int_0^t e_i(\\tau)\\,d\\tau \\right) + K_{\\text{ff},i}\\,\\Delta Q_{d,i}(t)$$

            * **Por qué $K_c$ es negativo ($-62, -76, -85\\text{ lpm/cm}$):** Como la acción de control manipula la bomba de **descarga**, aumentar $u_i(t)$ reduce el nivel ($k' < 0$). Cuando el nivel supera el setpoint ($h_i > h_{\\text{sp}} \\Rightarrow e_i < 0$), el producto $K_{c,i} \\cdot e_i(t)$ se vuelve **positivo**, aumentando el caudal de extracción de la bomba para restablecer el nivel.
            """
        )

    st.divider()
    st.markdown("### 3. Tabla Comparativa: Parámetros Reportados en el Artículo vs. Supuestos de Simulación")
    tabla_doc = pd.DataFrame([
        {
            "Parámetro / Componente": "Dimensiones de cada celda (C-1, C-2, C-3)",
            "Valor en el Simulador": f"{CELL_LENGTH_CM:.0f} × {CELL_WIDTH_CM:.0f} cm (Altura: {CELL_HEIGHT_CM:.0f} cm)",
            "Fuente / Justificación": "Reportado exactamente en Sección 'Experimental rig' y Fig. 2.",
        },
        {
            "Parámetro / Componente": "Ganancia integradora k'",
            "Valor en el Simulador": "-0.001018 cm·s⁻¹·lpm⁻¹",
            "Fuente / Justificación": "Reportado en Tabla 2. Conecta el balance en [lpm] con la derivada dh/dt en [cm/s].",
        },
        {
            "Parámetro / Componente": "Retardos de tiempo muerto (θ₁, θ₂, θ₃)",
            "Valor en el Simulador": "7 s, 6 s, 3 s",
            "Fuente / Justificación": "Reportado en Tabla 2 y página 10 a partir de pruebas de escalón.",
        },
        {
            "Parámetro / Componente": "Parámetros PI Iniciales SIMC (Kc, τi)",
            "Valor en el Simulador": "Kc = [-70.2, -81.8, -78.1] lpm/cm | τi = [56, 48, 52] s",
            "Fuente / Justificación": "Reportado en Tabla 2 (Ecuaciones 2 y 3 de Skogestad SIMC).",
        },
        {
            "Parámetro / Componente": "Parámetros PI Finales Ajustados (Kc, τi)",
            "Valor en el Simulador": "Kc = [-62, -76, -85] lpm/cm | τi = [56, 30, 44] s",
            "Fuente / Justificación": "Reportado en Tabla 2 tras ajuste fino experimental en el banco.",
        },
        {
            "Parámetro / Componente": "Constantes Feedforward (Kff,1, Kff,2, Kff,3)",
            "Valor en el Simulador": "0.8, 0.8, 1.0",
            "Fuente / Justificación": "Reportado en página 12 (debajo de la Ecuación 8).",
        },
        {
            "Parámetro / Componente": "Muestreo (Δt) y Filtro de Mediana (medfilt1)",
            "Valor en el Simulador": "Δt = 1 s | Ventana mediana = 10 s",
            "Fuente / Justificación": "Reportado en página 13 ('Method validation').",
        },
        {
            "Parámetro / Componente": "Recirculación al Tanque T-1 (130 L)",
            "Valor en el Simulador": "Caudal de alimentación configurable (45 lpm nominal)",
            "Fuente / Justificación": "Simplificación documentada: el tanque T-1 actúa como pulmón de 130 L que entrega Q_feed controlado por la bomba P-4.",
        },
    ])
    st.table(tabla_doc)


# -----------------------------------------------------------------------------
# PESTAÑA 5: EXPORTACIÓN CSV Y GUÍA DE EJECUCIÓN LOCAL
# -----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("💾 F. Descarga de Resultados en CSV y Verificación Local para el Docente")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_fb = df_fb.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar CSV — Simulación Solo PI (Feedback)",
            data=csv_fb,
            file_name="resultados_control_PI_feedback.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_dl2:
        csv_ff = df_ff.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar CSV — Simulación PI + Feedforward",
            data=csv_ff,
            file_name="resultados_control_PI_feedforward.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("#### Vista Previa de la Tabla de Datos Generada (`PI + Feedforward`, primeros 15 segundos)")
    st.dataframe(df_ff.head(15), use_container_width=True)
