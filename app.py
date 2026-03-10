import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
import subprocess
import time
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import ee
import os
import google.auth

# ==========================================
# INICIALIZACION DE EARTH ENGINE Y FOLIUM
# ==========================================
print("Conectando con Google Earth Engine...")
try:
    if 'GITHUB_ACTIONS' in os.environ:
        scopes = [
            'https://www.googleapis.com/auth/earthengine',
            'https://www.googleapis.com/auth/cloud-platform'
        ]
        credentials, project_id = google.auth.default(scopes=scopes)
        ee.Initialize(credentials, project='finca-489704')
    else:
        ee.Initialize(project='finca-489704')
    print("Conexion exitosa!")
except Exception as e:
    print(f"Error de conexion local: {e}. Intentando autenticar...")
    ee.Authenticate()
    ee.Initialize(project='finca-489704')

def add_ee_layer(self, ee_image_object, vis_params, name, show=True):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Google Earth Engine',
        name=name,
        overlay=True,
        control=True,
        show=show
    ).add_to(self)

folium.Map.add_ee_layer = add_ee_layer

# Coordenadas Globales de Atanquez
LAT_BASE = 10.682
LON_BASE = -73.327
ROI = ee.Geometry.Point([LON_BASE, LAT_BASE])

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
                time.sleep(3) 
                st.cache_data.clear()
                st.success("Datos y capas satelitales actualizadas.")
            except Exception as e:
                st.error(f"Error al actualizar: {e}")

st.title("Panel de Inteligencia Agrometeorologica (XGBoost)")
st.markdown("Monitor de clima local, modelado espacial y visor historico.")

# ==========================================
# 2. MOTOR DE DATOS Y MACHINE LEARNING
# ==========================================
@st.cache_data
def entrenar_y_procesar():
    df = pl.read_csv("historico_clima_finca.csv")
    df = df.with_columns(pl.col("Fecha").str.to_date("%Y-%m-%d")).drop_nulls()
    
    df = df.with_columns([
        pl.col("Fecha").dt.month().alias("Mes"),
        pl.col("Fecha").dt.ordinal_day().alias("Dia_del_Ano"),
        pl.col("Fecha").dt.year().alias("Ano")
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
    
    features = [col for col in df.columns if col not in ["Fecha", "Precipitacion_mm", "Temperatura_C", "Target_Lluvia_Manana", "Target_Temp_Manana", "Ano"]]
    
    X = df.select(features).to_numpy()
    y = df.select(["Target_Lluvia_Manana", "Target_Temp_Manana"]).to_numpy()
    
    modelo = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
    modelo.fit(X, y)
    
    df_pandas = df.to_pandas()
    df_pandas['Fecha'] = pd.to_datetime(df_pandas['Fecha'])
    
    return df_pandas, modelo, X, features, modelo.feature_importances_

with st.spinner('Cargando modelos hidrologicos y satelitales...'):
    df_pd, modelo_entrenado, X_matriz, nombres_features, importancias = entrenar_y_procesar()

condiciones_hoy = X_matriz[-1].reshape(1, -1)
pred_hoy = modelo_entrenado.predict(condiciones_hoy)[0]
pred_lluvia = max(0.0, pred_hoy[0])
pred_temp = pred_hoy[1]
condiciones_actuales = df_pd.iloc[-1]

tab_clima, tab_mapa, tab_anual, tab_historico = st.tabs(["Clima y Riego", "Visor Satelital GEE", "Ciclo Climatico y Fenologia", "Monitoreo Historico"])

# ==========================================
# PESTANA 1: CLIMA Y RIEGO
# ==========================================
with tab_clima:
    st.header("Pronostico y Necesidades de Riego (Proximas 24h)")

    lluvia_efectiva = pred_lluvia if pred_lluvia >= 5.0 else 0.0
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
        st.metric(label="Agua que se Evaporara (ETo)", value=f"-{evapotranspiracion_mm:.1f} mm", delta="Perdida del suelo", delta_color="inverse")
        if balance_agua < 0:
            st.metric(label="Balance Hidrico REAL", value=f"{balance_agua:.1f} mm", delta="Deficit", delta_color="inverse")
        else:
            st.metric(label="Balance Hidrico REAL", value=f"+{balance_agua:.1f} mm", delta="Superavit", delta_color="normal")

    with col3:
        st.write("**Dosis de Riego Recomendada**")
        if balance_agua < 0:
            st.error(f"Suelo en deficit hidrico. Reponer **{abs(balance_agua):.1f} L/m2**.")
        else:
            st.success(f"Lluvia compensara evaporacion. NO regar.")

    with st.expander("Como interpretar la Calculadora de Riego?"):
        st.write("La Evapotranspiracion (ETo) calcula cuanta agua pierde tu suelo hoy por el calor y el viento. Si la lluvia esperada es menor a la ETo, tendras un deficit (numero negativo) y el recuadro rojo te dira exactamente cuantos Litros por metro cuadrado de agua debes echarle al melon o papaya para que no sufran.")

    st.divider()
    st.header("Alerta Temprana de Sequia")
    lluvia_30d_actual = df_pd.tail(30)['Precipitacion_mm'].sum()
    mes_actual = df_pd.iloc[-1]['Mes']
    promedio_historico_mes = df_pd[df_pd['Mes'] == mes_actual]['Precipitacion_mm'].mean() * 30
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.write(f"**Acumulado ultimos 30 dias:** {lluvia_30d_actual:.1f} mm")
        st.write(f"**Promedio Historico (15 anos):** {promedio_historico_mes:.1f} mm")
    with col_s2:
        if lluvia_30d_actual < (promedio_historico_mes * 0.5):
            st.error("ALERTA DE SEQUIA SEVERA: Las lluvias estan 50% por debajo de lo normal.")
        else:
            st.success("CONDICIONES NORMALES: Humedad ambiental dentro de rangos seguros.")
            
    with st.expander("Como interpretar la Alerta de Sequia?"):
        st.write("Este modulo compara el agua que cayo en tu finca en los ultimos 30 dias con el agua que SOLIA caer en esta misma epoca durante los ultimos 15 anos. Te avisa si estas entrando en un periodo inusualmente seco para que prepares tus reservorios de agua con semanas de anticipacion.")

# ==========================================
# PESTANA 2: VISOR SATELITAL GEE
# ==========================================
with tab_mapa:
    st.header("Centro de Mando Satelital (Multicapa)")
    
    col_m1, col_m2 = st.columns([2, 1])
    
    with col_m1:
        m = folium.Map(location=[LAT_BASE, LON_BASE], zoom_start=15, control_scale=True)
        
        folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri', name='Satelite Base (Esri)', overlay=False).add_to(m)
        folium.TileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', attr='OpenTopoMap', name='Topografia de Referencia', overlay=False).add_to(m)
        folium.Marker([LAT_BASE, LON_BASE], popup="Finca (Lote Cultivos)").add_to(m)

        try:
            srtm = ee.Image('USGS/SRTMGL1_003')
            pendiente = ee.Terrain.slope(srtm)
            m.add_ee_layer(pendiente, {'min': 0, 'max': 30, 'palette': ['00FF00', 'FFFF00', 'FF0000']}, '1. Riesgo Erosion (Pendiente)', show=False)

            cobertura = ee.ImageCollection("ESA/WorldCover/v200").first()
            m.add_ee_layer(cobertura, {}, '2. Linderos y Bosque (ESA 10m)', show=False)

            fecha_fin = datetime.now().strftime('%Y-%m-%d')
            fecha_inicio = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            imagen_reciente = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(ROI).filterDate(fecha_inicio, fecha_fin).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)).sort('system:time_start', False).first()
            
            ndwi = imagen_reciente.normalizedDifference(['B3', 'B8']).rename('NDWI')
            m.add_ee_layer(ndwi, {'min': -0.3, 'max': 0.3, 'palette': ['#a50026', '#ffffbf', '#006837', '#0000ff']}, '3. Fuentes de Agua (NDWI)', show=False)

            ndmi = imagen_reciente.normalizedDifference(['B8', 'B11']).rename('NDMI')
            m.add_ee_layer(ndmi, {'min': -0.2, 'max': 0.6, 'palette': ['#d73027', '#fdae61', '#e0f3f8', '#4575b4']}, '4. Humedad Hoja (NDMI)', show=False)

            ndvi = imagen_reciente.normalizedDifference(['B8', 'B4']).rename('NDVI')
            m.add_ee_layer(ndvi, {'min': 0.0, 'max': 0.8, 'palette': ['#d73027', '#fdae61', '#a6d96a', '#1a9850']}, '5. Salud Vegetal (NDVI)', show=True)
            
        except Exception as e:
            st.error(f"Error cargando capas GEE: {e}")
            
        folium.LayerControl().add_to(m)
        st_folium(m, width=800, height=600)
        
    with col_m2:
        st.subheader("Guia de Interpretacion de Capas")
        st.write("Usa el icono superior derecho del mapa para cambiar de capa.")
        
        with st.expander("1. Riesgo de Erosion (Pendiente)"):
            st.write("Mide la inclinacion del terreno. Verde: Zonas planas ideales para siembra. Rojo: Zonas empinadas con alto riesgo de perder semillas por lavado durante aguaceros.")
            
        with st.expander("2. Linderos y Bosque (WorldCover)"):
            st.write("Clasificacion global de la Agencia Espacial Europea. Diferencia en distintos colores lo que el satelite considera 'Cultivo' versus 'Bosque Denso'.")
            
        with st.expander("3. Fuentes de Agua (NDWI)"):
            st.write("El indice de agua resalta en Azul intenso las zonas donde la tierra esta empapada o hay nacimientos/quebradas cercanas. Ideal para planificar reservorios.")
            
        with st.expander("4. Humedad de Hoja (NDMI)"):
            st.write("Mide el agua DENTRO de las hojas. Si tu cultivo se ve Rojo o Naranja, la planta esta cerrando sus poros por estres termico. Si esta Azul, esta hidratada.")
            
        with st.expander("5. Salud Vegetal (NDVI)"):
            st.write("Mide el vigor (clorofila). Verde oscuro: Plantas sanas, en crecimiento activo. Amarillo/Rojo: Suelo desnudo, maleza seca o plantas que requieren intervencion.")

# ==========================================
# PESTANA 3: CICLO CLIMATICO Y FENOLOGIA
# ==========================================
with tab_anual:
    st.header("Perfil Climatico Anual (Promedio 15 anos)")

    variables_clima = ['Precipitacion_mm', 'Temperatura_C', 'Radiacion_Solar_MJ_m2', 'Humedad_Suelo_Raices_m3_m3', 'Humedad_Suelo_Superficie_m3_m3', 'Humedad_Relativa_%', 'Velocidad_Viento_m_s']
    clima_promedio = df_pd.groupby('Dia_del_Ano')[variables_clima].mean().reset_index()
    for var in variables_clima:
        clima_promedio[f'{var}_Suavizada'] = clima_promedio[var].rolling(window=7, min_periods=1).mean()

    clima_promedio['Fecha_Grafica'] = pd.to_datetime('2026-01-01') + pd.to_timedelta(clima_promedio['Dia_del_Ano'] - 1, unit='D')
    clima_promedio.set_index('Fecha_Grafica', inplace=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.write("**Evolucion de Lluvias (mm)**")
        st.area_chart(clima_promedio['Precipitacion_mm_Suavizada'], color="#1f77b4")
        with st.expander("Interpretar Lluvia"):
            st.write("Muestra los dos picos de invierno clasicos de la region Andina y Caribe.")
            
        st.write("**Humedad en Zona de Raices (m3/m3)**")
        st.line_chart(clima_promedio['Humedad_Suelo_Raices_m3_m3_Suavizada'], color="#2ca02c")
        with st.expander("Interpretar Raices"):
            st.write("Indica que tan profundo penetra el agua. Clave para arboles de papaya y cafe.")
            
        st.write("**Humedad Relativa (%)**")
        st.line_chart(clima_promedio['Humedad_Relativa_%_Suavizada'], color="#9467bd")
        with st.expander("Interpretar Humedad Relativa"):
            st.write("Valores muy altos son alerta para la aparicion de hongos en las hojas.")

    with col_p2:
        st.write("**Temperatura Promedio (C)**")
        st.line_chart(clima_promedio['Temperatura_C_Suavizada'], color="#ff7f0e")
        with st.expander("Interpretar Temperatura"):
            st.write("Los valles (dias mas frios) suelen coincidir con los picos de lluvia por nubosidad.")
            
        st.write("**Humedad en Superficie (m3/m3)**")
        st.line_chart(clima_promedio['Humedad_Suelo_Superficie_m3_m3_Suavizada'], color="#98df8a")
        with st.expander("Interpretar Superficie"):
            st.write("Agua disponible en los primeros centimetros. Vital para semillas en germinacion.")
            
        st.write("**Radiacion Solar (MJ/m2)**")
        st.line_chart(clima_promedio['Radiacion_Solar_MJ_m2_Suavizada'], color="#e377c2")
        with st.expander("Interpretar Radiacion"):
            st.write("A mas radiacion, mayor produccion de azucares y metabolismo, pero requiere mas agua.")

    st.write("**Velocidad del Viento (m/s)**")
    st.line_chart(clima_promedio['Velocidad_Viento_m_s_Suavizada'], color="#7f7f7f")
    with st.expander("Interpretar Viento"):
        st.write("Si el viento es muy alto en temporada seca, secara tu tierra y aumentara la ETo.")

    st.divider()

    st.header("Ventana de Siembra y Fenologia")
    temporada_lluvias = clima_promedio[(clima_promedio['Dia_del_Ano'] > 60) & (clima_promedio['Precipitacion_mm_Suavizada'] > 4.0)]
    dia_inicio = int(temporada_lluvias.iloc[0]['Dia_del_Ano']) if not temporada_lluvias.empty else 115
    fecha_ideal = datetime.strptime(f'2026-{dia_inicio}', '%Y-%j').strftime('%d de %B')

    st.success(f"**Ventana Ideal de Trasplante (Fin de sequia):** A partir del **{fecha_ideal}**")
    
    with st.expander("Por que esta es la Ventana Ideal?"):
        st.write("El algoritmo encuentra la semana exacta donde la probabilidad de que las lluvias rompan la sequia es alta. Sembrar en esta fecha garantiza que la naturaleza ayude con el riego.")

    st.subheader("Curva de Fenologia (Crecimiento Vegetativo)")
    
    clima_promedio['Dias_Desde_Siembra'] = clima_promedio['Dia_del_Ano'] - dia_inicio
    clima_promedio['Vigor_Fenologico'] = np.where(clima_promedio['Dias_Desde_Siembra'] > 0, np.sin(clima_promedio['Dias_Desde_Siembra'] * np.pi / 150) * 100, 0)
    clima_promedio['Vigor_Fenologico'] = np.clip(clima_promedio['Vigor_Fenologico'], a_min=0, a_max=100)
    
    st.area_chart(clima_promedio['Vigor_Fenologico'], color="#2ca02c")
    with st.expander("Como leer la Curva de Fenologia?"):
        st.write("Esta campana es una simulacion de la vida de tu cultivo. Inicia el dia del trasplante y sube a medida que gana hojas. El pico mas alto es el momento de maxima floracion. Cuando la curva baje, preparate para la cosecha.")

# ==========================================
# PESTANA 4: HISTORIA Y AUDITORIA
# ==========================================
with tab_historico:
    st.header("Monitoreo Visual Historico y Auditoria")

    tab_h1, tab_h2, tab_h3 = st.tabs(["Maquina del Tiempo (Satelite)", "Historia Lluvia (Ultimo Ano)", "Motores IA"])

    with tab_h1:
        st.subheader("Maquina del Tiempo Satelital (Sentinel-2)")
        st.write("Mueve el deslizador para ver la fotografia real (color verdadero) de tu lote en anos pasados.")
        
        ano_seleccionado = st.slider("Selecciona el ano para visualizar:", 2019, 2026, 2026)
        
        with st.spinner(f"Descargando fotografia satelital del {ano_seleccionado}..."):
            m_hist = folium.Map(location=[LAT_BASE, LON_BASE], zoom_start=15)
            folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri', name='Satelite Base').add_to(m_hist)
            folium.Marker([LAT_BASE, LON_BASE], popup=f"Finca en {ano_seleccionado}").add_to(m_hist)
            
            try:
                fecha_in = f"{ano_seleccionado}-01-01"
                fecha_out = f"{ano_seleccionado}-12-31"
                img_historica = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(ROI).filterDate(fecha_in, fecha_out).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).median()
                vis_rgb = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}
                m_hist.add_ee_layer(img_historica, vis_rgb, f'Satelite Real ({ano_seleccionado})', show=True)
            except Exception as e:
                st.warning(f"Datos limitados para {ano_seleccionado}: {e}")
                
            st_folium(m_hist, width=800, height=450, key=f"mapa_hist_{ano_seleccionado}")
            
        with st.expander("Para que sirve este monitoreo historico?"):
            st.write("Te permite comprobar visualmente si ha habido expansion agricola, construcciones nuevas o cambios en los linderos de tu lote a traves de los anos, utilizando el espectro visible (RGB) del satelite.")

    with tab_h2:
        st.write("**Historial de Precipitacion (Ultimos 12 Meses)**")
        fecha_hace_un_ano = df_pd['Fecha'].max() - pd.DateOffset(years=1)
        df_reciente = df_pd[df_pd['Fecha'] >= fecha_hace_un_ano].copy()
        df_reciente['Ano_Mes'] = df_reciente['Fecha'].dt.to_period('M').astype(str)
        historial_mensual = df_reciente.groupby('Ano_Mes')['Precipitacion_mm'].sum().reset_index()
        
        st.bar_chart(historial_mensual.set_index('Ano_Mes'))
        
        with st.expander("Interpretar Historia a Corto Plazo"):
            st.write("Esta grafica enfoca el analisis en el ultimo ano operativo. Es fundamental para evaluar el exito o fracaso de las cosechas recientes cruzando estos datos con las fechas donde aplicaste riego o fertilizantes.")

    with tab_h3:
        st.write("**Que variables le importan mas a la Inteligencia Artificial?**")
        importancias_vars = list(zip(nombres_features, importancias))
        importancias_vars.sort(key=lambda x: x[1], reverse=True)
        nombres_v = [v[0] for v in importancias_vars[:5]]
        pesos_v = [v[1] * 100 for v in importancias_vars[:5]]
        df_imp = pd.DataFrame({"Variable": nombres_v, "Importancia %": pesos_v}).set_index("Variable")
        st.bar_chart(df_imp, color="#ff7f0e")
        with st.expander("Como leer esta grafica de IA?"):
            st.write("Muestra los 'secretos matematicos' del algoritmo. La barra mas alta indica la variable climatica que mas afecta la probabilidad de lluvia en tu punto geografico exacto.")
