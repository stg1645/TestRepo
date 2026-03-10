import ee
import folium
import os
import google.auth
import csv
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
# NUEVO: EXTRACCIÓN DE DATOS CLIMÁTICOS (10 AÑOS)
# ==========================================
print("Extrayendo 10 años de datos de precipitación (CHIRPS)...")

# Definimos el rango de tiempo: Desde hoy, 10 años atrás
fecha_fin = datetime.now().strftime('%Y-%m-%d')
fecha_inicio = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')

# Cargamos la colección CHIRPS (Lluvia diaria en mm)
chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
    .filterBounds(punto_finca) \
    .filterDate(fecha_inicio, fecha_fin) \
    .select('precipitation')

# Función para extraer el valor de cada imagen en nuestro punto exacto
def extraer_lluvia(imagen):
    fecha = imagen.date().format('YYYY-MM-dd')
    valor = imagen.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=punto_finca,
        scale=5566 # Escala nativa de CHIRPS
    ).get('precipitation')
    
    return ee.Feature(None, {'fecha': fecha, 'precipitacion_mm': valor})

# Aplicamos la función y traemos los datos de los servidores de Google a Python
datos_precipitacion = chirps.map(extraer_lluvia).getInfo()['features']

# Guardamos los datos en un archivo CSV
archivo_csv = 'historico_lluvias.csv'
with open(archivo_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Fecha', 'Precipitacion_mm']) # Encabezados
    for dato in datos_precipitacion:
        props = dato['properties']
        writer.writerow([props['fecha'], props.get('precipitacion_mm', 0)])

print(f"¡Historial guardado exitosamente en '{archivo_csv}'!")


# ==========================================
# GENERACIÓN DEL MAPA (Mantenemos lo que ya funciona)
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

# Capa DEM
dem = ee.Image('USGS/SRTMGL1_003')
vis_params_dem = {'min': 0, 'max': 4000, 'palette': ['006600', '002200', 'fff700', 'ab7634', 'c4d0ff', 'ffffff']}
Map.add_ee_layer(dem, vis_params_dem, 'Elevación del Terreno')

# Capa NDVI
# Usamos el último año para tener una imagen clara del estado actual
sentinel = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(punto_finca) \
    .filterDate((datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'), fecha_fin) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .median()
ndvi = sentinel.normalizedDifference(['B8', 'B4']).rename('NDVI')
vis_params_ndvi = {'min': 0.0, 'max': 0.8, 'palette': ['#d73027', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#1a9850']}
Map.add_ee_layer(ndvi.clip(punto_finca.buffer(3000)), vis_params_ndvi, 'Índice de Vegetación (NDVI)')

folium.Marker([latitud, longitud], popup='Ubicación Finca', icon=folium.Icon(color='red')).add_to(Map)
folium.LayerControl().add_to(Map)

Map.save("mapa_finca.html")
print("¡Script finalizado con éxito!")
