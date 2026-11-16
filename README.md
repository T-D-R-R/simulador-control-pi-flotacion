# Simulación de Control Regulatorio PI y Feedforward en un Banco de Flotación de 3 Celdas en Serie

🌐 **Aplicación Web Desplegada (URL Pública en Render):**  
👉 **[https://simulador-control-pi-flotacion.onrender.com](https://simulador-control-pi-flotacion.onrender.com)**

📦 **Repositorio de Código Fuente (GitHub):**  
👉 **[https://github.com/T-D-R-R/simulador-control-pi-flotacion](https://github.com/T-D-R-R/simulador-control-pi-flotacion)**

---

Aplicación web interactiva y motor matemático en Python que reproduce la metodología e implementación experimental del artículo:

> **Quintanilla, P., Navia, D., Moreno, F., Neethling, S. J., & Brito-Parada, P. R. (2023).**  
> *"A methodology to implement a closed-loop feedback-feedforward level control in a laboratory-scale flotation bank using peristaltic pumps"*.  
> **MethodsX**, 10, 102081. [https://doi.org/10.1016/j.mex.2023.102081](https://doi.org/10.1016/j.mex.2023.102081)

---

## 1. Propósito del Proyecto

Este proyecto permite al docente y al estudiante:
1. **Acceder desde el navegador (URL pública en Render)** para ejecutar y visualizar la simulación del banco de flotación de tres celdas en serie.
2. **Descargar el código Python y ejecutarlo localmente**, tanto en interfaz web (`Streamlit`) como directamente por consola/IDE (`simular_local.py`).
3. **Inspeccionar los modelos matemáticos puros** separados de la interfaz visual (`modelo.py`, `controlador.py`, `metricas.py`, `parametros.py`).
4. **Comparar cuantitativa y gráficamente** el desempeño del control clásico **PI (Feedback)** frente al control **PI + Feedforward** mediante los indicadores **IAE**, **SSE**, **Error Máximo** y **Variación Total de Control (TV)**.

---

## 2. Arquitectura de Archivos

```text
SO_1/
├── app.py                  # Aplicación web interactiva en Streamlit (Diagrama P&ID, Gráficos, Métricas, CSV)
├── modelo.py               # Modelo dinámico integrador con retardo (theta) de las 3 celdas en serie y filtro medfilt1
├── controlador.py          # Ley de control PI discreta, compensación Feedforward, saturación y Anti-Windup
├── metricas.py             # Cálculo de IAE (Ec. 9), SSE, Error Máximo, Error Final y Variación Total (TV)
├── parametros.py           # Parámetros de Tabla 2 (SIMC inicial y final), curvas PLC 16-bit (Figs. 8, 9, 11)
├── simular_local.py        # Script de validación local por consola y gráficos Matplotlib (sin navegador)
├── render.yaml             # Configuración lista para despliegue automático en Render
├── requirements.txt        # Librerías requeridas de Python
└── README.md               # Documentación técnica y manual de instalación
```

---

## 3. Resumen de Modelos Matemáticos Implementados

### 3.1 Dinámica Integradora con Retardo (`modelo.py`)
Al regular la descarga de colas (*tailings*) mediante bombas peristálticas ($P_1, P_2, P_3$) en lugar de válvulas por gravedad, cada celda $i \in \{1, 2, 3\}$ se comporta como un **proceso integrador puro con retardo de transporte ($\theta_i$)** (Tabla 2 y Ecuación 4 del artículo):

$$G_{p,i}(s) = \frac{Y_i(s)}{U_i(s)} = \frac{k'_i \, e^{-\theta_i s}}{s}$$

En el dominio del tiempo continuo y discreto ($\Delta t = 1\text{ s}$), el balance de caudales en cada celda es:

$$\frac{dh_i(t)}{dt} = \frac{Q_{\text{in},i}(t) - Q_{\text{out},i}(t - \theta_i)}{A_{\text{efectiva},i}} = |k'_i| \cdot \Big( Q_{\text{in},i}(t) - u_i(t - \theta_i) \Big)$$

donde:
* $k'_i = -0.001018 \text{ cm}\cdot\text{s}^{-1}\cdot\text{lpm}^{-1}$ para las 3 celdas ($30\text{ cm} \times 30\text{ cm}$).
* Retardos de transporte: $\theta_1 = 7\text{ s}$, $\theta_2 = 6\text{ s}$, $\theta_3 = 3\text{ s}$.
* **Conexión en cascada (Figura 15):**
  * $Q_{\text{in},1}(t) = Q_{\text{feed}}(t)$ (bomba de alimentación $P_4$, nominal $45\text{ lpm}$).
  * $Q_{\text{in},2}(t) = Q_{\text{tailings},1}(t) = u_1(t)$
  * $Q_{\text{in},3}(t) = Q_{\text{tailings},2}(t) = u_2(t)$
* **Filtrado de señal:** Lectura de nivel con ruido por burbujeo de aire ($47\text{ lpm}$, $j_g = 0.87\text{ cm/s}$) procesada con un **filtro de mediana móvil (`medfilt1`) de ventana de 10 segundos** (pág. 13).

### 3.2 Sintonización SIMC y Controlador PI + Feedforward (`controlador.py`)
Los parámetros iniciales se obtienen con las reglas **SIMC (Simple Internal Model Control)** de Skogestad (Ecuaciones 2 y 3):

$$K_{c,i} = \frac{1}{k'_i} \frac{1}{\tau_{c,i} + \theta_i}, \qquad \tau_{i,i} = 4(\tau_{c,i} + \theta_i)$$

La ley de control combinada **Feedback PI + Feedforward** (Ecuaciones 7 y 8) es:

$$e_i(t) = h_{\text{sp},i}(t) - h_i(t)$$

$$u_i(t) = u_{\text{bias},i} + K_{c,i}\left( e_i(t) + \frac{1}{\tau_{i,i}} \int_0^t e_i(\tau)\,d\tau \right) + K_{\text{ff},i}\,\Delta Q_{d,i}(t)$$

| Parámetro | Unidades | Celda 1 (C-1) | Celda 2 (C-2) | Celda 3 (C-3) |
| :--- | :---: | :---: | :---: | :---: |
| Ganancia integradora $k'$ | $\text{cm}\cdot\text{s}^{-1}\cdot\text{lpm}^{-1}$ | $-0.001018$ | $-0.001018$ | $-0.001018$ |
| Retardo $\theta$ | $\text{s}$ | $7$ | $6$ | $3$ |
| Constante de sintonización $\tau_c$ | $\text{s}$ | $7$ | $6$ | $8$ |
| Ganancia SIMC Inicial $K_c^{(1)}$ | $\text{lpm}\cdot\text{cm}^{-1}$ | $-70.2$ | $-81.8$ | $-78.1$ |
| Tiempo Integral SIMC Inicial $\tau_i^{(1)}$ | $\text{s}$ | $56$ | $48$ | $52$ |
| **Ganancia Final Ajustada $K_c$** | $\text{lpm}\cdot\text{cm}^{-1}$ | **$-62$** | **$-76$** | **$-85$** |
| **Tiempo Integral Final Ajustado $\tau_i$** | $\text{s}$ | **$56$** | **$30$** | **$44$** |
| **Constante Feedforward $K_{\text{ff}}$** | $-$ | **$0.8$** | **$0.8$** | **$1.0$** |

---

## 4. Instrucciones de Instalación y Ejecución Local (Para el Docente)

### Requisitos previos
* **Python 3.9 o superior** instalado en el sistema.

### Paso 1: Abrir terminal en la carpeta del proyecto y crear entorno virtual (opcional pero recomendado)

**En Windows (PowerShell o CMD):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**En Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Paso 2: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 3A: Ejecutar la Aplicación Web Interactiva (Streamlit)
```bash
streamlit run app.py
```
El navegador se abrirá automáticamente en `http://localhost:8501`.

### Paso 3B: Ejecutar la Validación Matemática Directa en Consola / IDE (Sin navegador)
Si desea revisar la ejecución directa de los modelos matemáticos en terminal y abrir las gráficas nativas de `Matplotlib` (Figura 14 del artículo):
```bash
python simular_local.py
```
*(También genera automáticamente los archivos `resultados_simulacion_PI_feedback.csv` y `resultados_simulacion_PI_feedforward.csv` en la carpeta del proyecto).*

---

## 5. Instrucciones de Despliegue en Render (URL Pública)

El repositorio incluye el archivo `render.yaml` preconfigurado:
1. Subir todos los archivos de esta carpeta a un repositorio de **GitHub**.
2. Ingresar a [https://render.com](https://render.com) -> **New +** -> **Web Service** (o **Blueprint**).
3. Conectar el repositorio de GitHub.
4. Si se configura manualmente como **Web Service**:
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
5. Render compilará e iniciará el servicio entregando una URL pública `https://simulador-control-pi-flotacion.onrender.com`.
