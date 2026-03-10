import ee
import folium
import os
import google.auth

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

# Configuración de Folium
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

# Coordenadas reales
latitud = 10.682
longitud = -73.327
punto_finca = ee.Geometry.Point([longitud, latitud])

# Creamos el mapa base
Map = folium.Map(location=[latitud, longitud], zoom_start=14, tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri')

# ==========================================
# CAPA 1: Elevación del Terreno (DEM)
# ==========================================
dem = ee.Image('USGS/SRTMGL1_003')
vis_params_dem = {'min': 0, 'max': 4000, 'palette': ['006600', '002200', 'fff700', 'ab7634', 'c4d0ff', 'ffffff']}
Map.add_ee_layer(dem, vis_params_dem, 'Elevación del Terreno')

# ==========================================
# CAPA 2: Salud de Vegetación (NDVI)
# ==========================================
# 1. Traemos imágenes del satélite Sentinel-2 de los últimos meses, filtramos las nubes y sacamos el promedio
sentinel = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(punto_finca) \
    .filterDate('2026-01-01', '2026-03-09') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .median()

# 2. Calculamos el NDVI: (Infrarrojo Cercano - Rojo) / (Infrarrojo Cercano + Rojo)
ndvi = sentinel.normalizedDifference(['B8', 'B4']).rename('NDVI')

# 3. Colores: Rojo/Amarillo (seco/sin plantas) -> Verde (mucha vegetación/humedad)
vis_params_ndvi = {
    'min': 0.0,
    'max': 0.8,
    'palette': ['#d73027', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#1a9850']
}

# 4. Añadimos la capa al mapa (recortada a 3km a la redonda de tu finca para que cargue rápido)
Map.add_ee_layer(ndvi.clip(punto_finca.buffer(3000)), vis_params_ndvi, 'Índice de Vegetación (NDVI)')

# ==========================================
# Controles y guardado
# ==========================================
folium.Marker([latitud, longitud], popup='Ubicación Finca', icon=folium.Icon(color='red')).add_to(Map)
# Este control es el que te permite prender y apagar las capas
folium.LayerControl().add_to(Map)

archivo_salida = "mapa_finca.html"
Map.save(archivo_salida)
print(f"¡Mapa generado con NDVI! Guardado en '{archivo_salida}'.")
