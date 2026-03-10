import ee
import folium
import os
import google.auth
import csv
import math
from datetime import datetime, timedelta

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
    print("¡Conexión exitosa!")
except Exception as e:
    print(f"Error de conexión: {e}")
    raise

# ==========================================
# CONFIGURACIÓN BASE
# ==========================================
latitud = 10.682
longitud = -73.327
punto_finca = ee.Geometry.Point([longitud, latitud])

# ==========================================
# EXTRACCIÓN DE 15 AÑOS DE DATOS (ERA5-LAND)
# ==========================================
print("Extrayendo 15 años de variables agrometeorológicas (ERA5-Land)...")

def extraer_datos(imagen):
    fecha = imagen.date().format('YYYY-MM-dd')
    valores = imagen.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=punto_finca,
        scale=11132 # Escala nativa de ERA5
    )
    return ee.Feature(None, {'fecha': fecha, 'valores': valores})

# Para no saturar la memoria de Google, descargamos en bloques de 5 años
años_totales = 15
bloque = 5
datos_completos = []

for i in range(0, años_totales, bloque):
    dias_inicio = 365 * (años_totales - i)
    dias_fin = 365 * (años_totales - i - bloque)
    
    fecha_in = (datetime.now() - timedelta(days=dias_inicio)).strftime('%Y-%m-%d')
    fecha_fi = (datetime.now() - timedelta(days=dias_fin)).strftime('%Y-%m-%d')
    if i == años_totales - bloque:
        fecha_fi = datetime.now().strftime('%Y-%m-%d') # Hasta la fecha actual

    print(f"Descargando bloque: {fecha_in} a {fecha_fi}...")
    
    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
        .filterBounds(punto_finca) \
        .filterDate(fecha_in, fecha_fi)
        
    lote_datos = era5.map(extraer_datos).getInfo()['features']
    datos_completos.extend(lote_datos)

# Guardamos los datos procesados en CSV
archivo_csv = 'historico_clima_finca.csv'
print(f"Procesando y guardando {len(datos_completos)} registros diarios...")

with open(archivo_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Encabezados de tu nueva matriz de datos
    writer.writerow(['Fecha', 'Precipitacion_mm', 'Temperatura_C', 'Humedad_Relativa_%', 
                     'Velocidad_Viento_m_s', 'Humedad_Suelo_Superficie_m3_m3', 
                     'Humedad_Suelo_Raices_m3_m3', 'Radiacion_Solar_MJ_m2'])
    
    for dato in datos_completos:
        fecha = dato['properties'].get('fecha')
        props = dato['properties'].get('valores', {})
        if not props: continue

        # Extracción bruta
        precip_m = props.get('total_precipitation_sum')
        t_2m_k = props.get('temperature_2m')
        td_2m_k = props.get('dewpoint_temperature_2m')
        u_wind = props.get('u_component_of_wind_10m')
        v_wind = props.get('v_component_of_wind_10m')
        soil_surf = props.get('volumetric_soil_water_layer_1')
        soil_root = props.get('volumetric_soil_water_layer_2') # Zona de raíces
        rad_j = props.get('surface_solar_radiation_downwards_sum')

        # CONVERSIONES CIENTÍFICAS
        precip_mm = round(precip_m * 1000, 2) if precip_m is not None else 0
        t_c = round(t_2m_k - 273.15, 2) if t_2m_k is not None else None
        td_c = td_2m_k - 273.15 if td_2m_k is not None else None
        
        # Calcular Velocidad del Viento (Pitágoras)
        wind_speed = round(math.sqrt(u_wind**2 + v_wind**2), 2) if (u_wind is not None and v_wind is not None) else None
        
        # Calcular Humedad Relativa (Aproximación August-Roche-Magnus)
        if t_c is not None and td_c is not None:
            rh = round(100 * (math.exp((17.625 * td_c) / (243.04 + td_c)) / math.exp((17.625 * t_c) / (243.04 + t_c))), 2)
        else:
            rh = None
            
        rad_mj = round(rad_j / 1000000, 2) if rad_j is not None else None # Megajulios

        writer.writerow([fecha, precip_mm, t_c, rh, wind_speed, soil_surf, soil_root, rad_mj])

print(f"¡Base de datos guardada en '{archivo_csv}'!")

# ==========================================
# GENERACIÓN DEL MAPA (Mantenemos NDVI y Elevación)
# ==========================================
def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)
folium.Map.add_ee_layer = add_ee_layer

Map = folium.Map(location=[latitud, longitud], zoom_start=14, tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri')

dem = ee.Image('USGS/SRTMGL1_003')
Map.add_ee_layer(dem, {'min': 0, 'max': 4000, 'palette': ['006600', '002200', 'fff700', 'ab7634', 'c4d0ff', 'ffffff']}, 'Elevación del Terreno')

# Capa NDVI actual
sentinel = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(punto_finca) \
    .filterDate((datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d')) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .median()
ndvi = sentinel.normalizedDifference(['B8', 'B4']).rename('NDVI')
Map.add_ee_layer(ndvi.clip(punto_finca.buffer(3000)), {'min': 0.0, 'max': 0.8, 'palette': ['#d73027', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#1a9850']}, 'Índice de Vegetación (NDVI)')

folium.Marker([latitud, longitud], popup='Ubicación Finca', icon=folium.Icon(color='red')).add_to(Map)
folium.LayerControl().add_to(Map)
Map.save("mapa_finca.html")
print("¡Script finalizado con éxito!")
