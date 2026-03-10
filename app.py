import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

# ==========================================
# 1. CONFIGURACION DE PAGINA
# ==========================================
st.set_page_config(page_title="Estacion Meteorologica IA", layout="wide")
st.title("Panel de Inteligencia Agrometeorologica (XGBoost)")
st.markdown("Monitor de clima local, predicciones de alta precision y planificador de siembras.")

# ==========================================
# 2. MOTOR DE DATOS Y MACHINE LEARNING
# ==========================================
@st.cache_data
def entrenar_y_procesar():
    df = pl.read_csv("historico_clima_finca.csv")
    df = df.with_columns(pl.col("Fecha").str.to_date("%Y-%m-%d")).drop_nulls()
    
    df = df.with_columns([
        pl.col("Fecha").dt.month().alias("Mes"),
        pl.col("Fecha").dt.ordinal_day().alias("Dia_del_Ano") 
    ])
    
    columnas_clima = ["Precipitacion_mm", "Temperatura_C", "Humedad_Relativa_%", 
                      "Humedad_Suelo_Raices_m3_m3", "Radiacion_Solar_MJ_m2",
                      "Velocidad_Viento_m_s", "Humedad_Suelo_Superficie_m3_m3"]
    lags = []
    for col in columnas_clima:
        for dias_atras in [1, 2, 3]:
            lags.append(pl.col(col).shift(dias_atras).alias(f"{col}_ayer_{dias_atras}"))
    
    df = df.with_columns(lags)
    
    df = df.with_columns([
        pl.col("Precipitacion_mm").shift(-1).alias("Target_Lluvia_Manana"),
        pl.col("Temperatura_C").shift(-1).alias("Target_Temp_Manana")
    ]).drop_nulls()
    
    features = [col for col in df.columns if col not in ["Fecha", "Precipitacion_mm", "Temperatura_C", "Target_Lluvia_Manana", "Target_Temp_Manana"]]
    
    X = df.select(features).to_numpy()
    y = df.select(["Target_Lluvia_Manana", "Target_Temp_Manana"]).to_numpy()
    
    modelo = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
    modelo.fit(X, y)
    
    df_pandas = df.to_pandas()
    
    return df_pandas, modelo, X, features, modelo.feature_importances_

with st.spinner('Procesando 15 anos de datos y calibrando XGBoost...'):
    df_pd, modelo_entrenado, X_matriz, nombres_features, importancias = entrenar_y_procesar()

# ==========================================
# 3. PREDICCION INMEDIATA (MANANA)
# ==========================================
condiciones_hoy = X_matriz[-1].reshape(1, -1)
pred_hoy = modelo_entrenado.predict(condiciones_hoy)[0]
pred_lluvia = max(0.0, pred_hoy[0])
pred_temp = pred_hoy[1]
condiciones_actuales = df_pd.iloc[-1]

st.header("Pronostico Tactico para Manana")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Temperatura Promedio Esperada", value=f"{pred_temp:.1f} C", delta=f"{(pred_temp - condiciones_actuales['Temperatura_C']):.1f} C vs Hoy")
with col2:
    st.metric(label="Lluvia Esperada (Prec. Quirurgica)", value=f"{pred_lluvia:.1f} mm")
with col3:
    humedad_actual = condiciones_actuales['Humedad_Relativa_%']
    if pred_temp > 28.0 and humedad_actual > 75:
        st.error("ALTO ESTRES TERMICO: Bochorno extremo. Riesgo de hongos.")
    elif pred_temp > 30.0 and humedad_actual < 50:
        st.warning("ALTO ESTRES HIDRICO: Aire muy seco. Evaporacion masiva.")
    else:
        st.success("CONDICIONES ESTABLES: Clima moderado para las plantas.")

with st.expander("Ver explicacion del Pronostico Tactico"):
    st.write("""
    **Como interpretar esto en el lote:**
    - **Temperatura vs Hoy:** Te indica si la tendencia de la semana es al alza (mas calor) o a la baja (refrescando).
    - **Lluvia (mm):** Menos de 5 mm se considera rocio o lluvia no efectiva. Mas de 15 mm penetrara hasta la zona de raices.
    - **Estres Termico:** Cruza el calor esperado con la humedad de hoy. Si el aire es muy seco y caluroso, las hojas transpiraran mas agua de la que las raices pueden absorber, causando marchitamiento.
    """)

st.divider()

# ==========================================
# 4. PLANIFICADOR DE SIEMBRAS Y CICLO ANUAL
# ==========================================
st.header("Planificador de Siembras y Ciclo Anual")

# Calculamos el promedio historico para TODAS las variables por dia del ano
variables_clima = [
    'Precipitacion_mm', 'Temperatura_C', 'Humedad_Relativa_%', 
    'Humedad_Suelo_Raices_m3_m3', 'Humedad_Suelo_Superficie_m3_m3', 
    'Radiacion_Solar_MJ_m2', 'Velocidad_Viento_m_s'
]

clima_promedio = df_pd.groupby('Dia_del_Ano')[variables_clima].mean().reset_index()

# Suavizamos las curvas (media movil de 7 dias) para ver las tendencias claras
for var in variables_clima:
    clima_promedio[f'{var}_Suavizada'] = clima_promedio[var].rolling(window=7, min_periods=1).mean()

# TRUCO PARA EL EJE X: Convertimos el "Dia del Ano" a una fecha real del 2026
clima_promedio['Fecha_Grafica'] = pd.to_datetime('2026-01-01') + pd.to_timedelta(clima_promedio['Dia_del_Ano'] - 1, unit='D')
clima_promedio.set_index('Fecha_Grafica', inplace=True)

# Calculo de la ventana de siembra
temporada_lluvias = clima_promedio[(clima_promedio['Dia_del_Ano'] > 60) & (clima_promedio['Precipitacion_mm_Suavizada'] > 4.0)]

if not temporada_lluvias.empty:
    dia_inicio = int(temporada_lluvias.iloc[0]['Dia_del_Ano'])
    fecha_ideal = datetime.strptime(f'2026-{dia_inicio}', '%Y-%j').strftime('%d de %B')
else:
    fecha_ideal = "Finales de Abril"

st.success(f"**Ventana Ideal de Trasplante Estimada:** A partir del **{fecha_ideal}**")

with st.expander("Ver explicacion del Periodo Ideal de Cultivo"):
    st.write(f"""
    **Por que el modelo sugiere el {fecha_ideal}?**
    Al promediar los ultimos 15 anos, el algoritmo identifica que en esta fecha especifica la temporada seca se rompe de forma consistente. 
    A partir de este momento las lluvias superan los 4 mm diarios promedio y la humedad del suelo comienza una curva ascendente.
    **Recomendacion:** Manten tus semillas protegidas. Preparate para llevarlas al lote definitivo (cerca al platano y el cafe) en la semana del {fecha_ideal} para aprovechar el inicio natural de las lluvias.
    """)

st.subheader("Perfil Climatico Anual (Promedio Historico de 15 anos)")

col_p1, col_p2 = st.columns(2)
with col_p1:
    st.write("**Evolucion de Lluvias (mm)**")
    st.area_chart(clima_promedio['Precipitacion_mm_Suavizada'], color="#1f77b4")
with col_p2:
    st.write("**Temperatura Promedio (C)**")
    st.line_chart(clima_promedio['Temperatura_C_Suavizada'], color="#ff7f0e")

col_p3, col_p4 = st.columns(2)
with col_p3:
    st.write("**Humedad en Zona de Raices (m3/m3)**")
    st.line_chart(clima_promedio['Humedad_Suelo_Raices_m3_m3_Suavizada'], color="#2ca02c")
with col_p4:
    st.write("**Humedad en Superficie (m3/m3)**")
    st.line_chart(clima_promedio['Humedad_Suelo_Superficie_m3_m3_Suavizada'], color="#98df8a")

col_p5, col_p6 = st.columns(2)
with col_p5:
    st.write("**Humedad Relativa (%)**")
    st.line_chart(clima_promedio['Humedad_Relativa_%_Suavizada'], color="#9467bd")
with col_p6:
    st.write("**Radiacion Solar (MJ/m2)**")
    st.line_chart(clima_promedio['Radiacion_Solar_MJ_m2_Suavizada'], color="#e377c2")

st.write("**Velocidad del Viento (m/s)**")
st.line_chart(clima_promedio['Velocidad_Viento_m_s_Suavizada'], color="#7f7f7f")

st.divider()

# ==========================================
# 5. METRICAS CLIMATICAS Y EXPLICABILIDAD
# ==========================================
st.header("Auditoria del Clima Actual")

tab1, tab2 = st.tabs(["Motores Climaticos", "Ultimos 30 dias"])

with tab1:
    st.write("**Que variables generan la lluvia en tus coordenadas?**")
    importancias_vars = list(zip(nombres_features, importancias))
    importancias_vars.sort(key=lambda x: x[1], reverse=True)
    
    nombres_v = [v[0] for v in importancias_vars[:5]]
    pesos_v = [v[1] * 100 for v in importancias_vars[:5]]
    
    df_imp = pd.DataFrame({"Variable": nombres_v, "Importancia %": pesos_v}).set_index("Variable")
    st.bar_chart(df_imp, color="#ff7f0e")
    
    with st.expander("Como leer esta grafica?"):
        st.write("""
        Esta grafica muestra los 'secretos' de tu microclima local. 
        Si la Humedad Relativa es la barra mas alta, significa que las lluvias en tu lote dependen mas del vapor de agua en el viento chocando con las montanas cercanas que del simple calor.
        """)

with tab2:
    st.write("**Acumulado de lluvia reciente**")
    df_mes = df_pd.tail(30).set_index("Fecha")
    st.bar_chart(df_mes["Precipitacion_mm"])
    
    with st.expander("Para que sirve este registro?"):
        st.write("""
        Las plantas no solo viven de la lluvia del dia, sino del agua retenida. Ver los ultimos 30 dias te permite saber si vienes de un periodo de sequia prolongada. Si las barras estan casi planas, la reserva de agua subterranea esta agotada y cualquier trasplante requerira riego de auxilio.
        """)
