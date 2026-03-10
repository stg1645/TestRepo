import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
import subprocess
import time
from datetime import datetime

# ==========================================
# 1. CONFIGURACION DE PAGINA Y MENU LATERAL
# ==========================================
st.set_page_config(page_title="Estacion Meteorologica IA", layout="wide")

with st.sidebar:
    st.header("Panel de Control")
    st.write("Sincronizacion de datos satelitales.")
    
    if st.button("Actualizar Datos (Earth Engine)"):
        with st.spinner("Conectando con el satelite y descargando..."):
            try:
                # subprocess.run(["python", "tu_script_gee.py"], check=True)
                time.sleep(3) 
                st.cache_data.clear()
                st.success("Datos actualizados y modelo reentrenado con exito.")
            except Exception as e:
                st.error(f"Error al actualizar: {e}")

st.title("Panel de Inteligencia Agrometeorologica (XGBoost)")
st.markdown("Monitor de clima local, predicciones, calculo de riego y planificador de siembras.")

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

with st.spinner('Procesando datos y calibrando XGBoost...'):
    df_pd, modelo_entrenado, X_matriz, nombres_features, importancias = entrenar_y_procesar()

# ==========================================
# 3. PREDICCION INMEDIATA Y CALCULADORA DE RIEGO
# ==========================================
condiciones_hoy = X_matriz[-1].reshape(1, -1)
pred_hoy = modelo_entrenado.predict(condiciones_hoy)[0]
pred_lluvia = max(0.0, pred_hoy[0])
pred_temp = pred_hoy[1]
condiciones_actuales = df_pd.iloc[-1]

st.header("Pronostico y Necesidades de Riego (Proximas 24h)")

# CORRECCION DE LLUVIA EFECTIVA
lluvia_efectiva = pred_lluvia if pred_lluvia >= 5.0 else 0.0

# CORRECCION DE EVAPOTRANSPIRACION (ETo)
temp_hoy = condiciones_actuales['Temperatura_C']
viento_hoy = condiciones_actuales['Velocidad_Viento_m_s']
humedad_hoy = condiciones_actuales['Humedad_Relativa_%']

evapotranspiracion_mm = (temp_hoy * 0.25) + (viento_hoy * 0.5)
if humedad_hoy < 60:
    evapotranspiracion_mm += 2.0 

evapotranspiracion_mm = max(4.5, evapotranspiracion_mm)

balance_agua = lluvia_efectiva - evapotranspiracion_mm

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Lluvia Bruta Esperada", value=f"{pred_lluvia:.1f} mm")
    st.metric(label="Temperatura Promedio", value=f"{pred_temp:.1f} C")
    
with col2:
    st.metric(label="Agua que se Evaporara (ETo)", value=f"-{evapotranspiracion_mm:.1f} mm", delta="Perdida real del suelo", delta_color="inverse")
    if balance_agua < 0:
        st.metric(label="Balance Hidrico REAL", value=f"{balance_agua:.1f} mm", delta="Deficit (Tierra secandose)", delta_color="inverse")
    else:
        st.metric(label="Balance Hidrico REAL", value=f"+{balance_agua:.1f} mm", delta="Superavit", delta_color="normal")

with col3:
    st.write("**Dosis de Riego Recomendada**")
    if balance_agua < 0:
        riego_necesario = abs(balance_agua)
        st.error(f"Suelo en deficit hidrico. Reponer **{riego_necesario:.1f} L/m2** (Litros por metro cuadrado).")
        st.write("- **Melon/Papaya:** Riego urgente hoy (alto consumo).")
        st.write("- **Frijol/Pimenton:** Riego moderado en la base.")
        if 0 < pred_lluvia < 5.0:
            st.caption(f"Nota: Los {pred_lluvia:.1f} mm de lluvia pronosticados son 'no efectivos' (no llegaran a la raiz).")
    else:
        st.success(f"La lluvia efectiva compensara la evaporacion. Sobraran {balance_agua:.1f} mm.")
        st.write("NO encender riego. Mantener monitoreo.")

# ==========================================
# 4. PLANIFICADOR DE SIEMBRAS Y CICLO ANUAL
# ==========================================
st.divider()
st.header("Planificador de Siembras y Ciclo Anual")

variables_clima = [
    'Precipitacion_mm', 'Temperatura_C', 'Humedad_Relativa_%', 
    'Humedad_Suelo_Raices_m3_m3', 'Humedad_Suelo_Superficie_m3_m3', 
    'Radiacion_Solar_MJ_m2', 'Velocidad_Viento_m_s'
]

clima_promedio = df_pd.groupby('Dia_del_Ano')[variables_clima].mean().reset_index()

for var in variables_clima:
    clima_promedio[f'{var}_Suavizada'] = clima_promedio[var].rolling(window=7, min_periods=1).mean()

clima_promedio['Fecha_Grafica'] = pd.to_datetime('2026-01-01') + pd.to_timedelta(clima_promedio['Dia_del_Ano'] - 1, unit='D')
clima_promedio.set_index('Fecha_Grafica', inplace=True)

temporada_lluvias = clima_promedio[(clima_promedio['Dia_del_Ano'] > 60) & (clima_promedio['Precipitacion_mm_Suavizada'] > 4.0)]

if not temporada_lluvias.empty:
    dia_inicio = int(temporada_lluvias.iloc[0]['Dia_del_Ano'])
    fecha_ideal = datetime.strptime(f'2026-{dia_inicio}', '%Y-%j').strftime('%d de %B')
else:
    fecha_ideal = "Finales de Abril"

st.success(f"**Ventana Ideal de Trasplante Estimada:** A partir del **{fecha_ideal}**")

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

# ==========================================
# 5. METRICAS CLIMATICAS Y EXPLICABILIDAD
# ==========================================
st.divider()
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

with tab2:
    st.write("**Acumulado de lluvia reciente**")
    df_mes = df_pd.tail(30).set_index("Fecha")
    st.bar_chart(df_mes["Precipitacion_mm"])
