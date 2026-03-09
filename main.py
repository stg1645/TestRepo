import ee
import folium
import os
import google.auth

print("Conectando con Google Earth Engine...")

try:
    # Verificamos si estamos en la nube (GitHub Actions)
    if 'GITHUB_ACTIONS' in os.environ:
        print("Entorno de GitHub Actions detectado. Extrayendo llave del robot...")
        # Tomamos las credenciales inyectadas por GitHub
        credentials, project_id = google.auth.default()
        # Le pasamos las credenciales explícitamente a Earth Engine
        ee.Initialize(credentials, project='finca-489704')
    else:
        print("Entorno local detectado. Usando sesión de navegador...")
        # En tu PC seguimos usando tu inicio de sesión normal
        ee.Initialize(project='finca-489704')
        
    print("¡Conexión exitosa!")
except Exception as e:
    print(f"Error de conexión: {e}")
    raise

# 1. Definimos una función puente para conectar Earth Engine con Folium
def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

# Agregamos la función a la clase de Folium
folium.Map.add_ee_layer = add_ee_layer

# 2. Coordenadas de prueba
latitud = 10.682
longitud = -73.327

# 3. Creamos el mapa base satelital usando Folium directamente
# Esri World Imagery es un mapa base satelital excelente y estable
Map = folium.Map(location=[latitud, longitud], zoom_start=10, tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri')

# 4. Cargamos y configuramos los datos de Earth Engine (El Relieve)
dem = ee.Image('USGS/SRTMGL1_003')
vis_params = {
    'min': 0,
    'max': 4000,
    'palette': ['006600', '002200', 'fff700', 'ab7634', 'c4d0ff', 'ffffff']
}

# 5. Añadimos las capas de Earth Engine y un marcador
Map.add_ee_layer(dem, vis_params, 'Elevación del Terreno')
folium.Marker([latitud, longitud], popup='Ubicación Finca', icon=folium.Icon(color='red')).add_to(Map)

# Añadimos el control para encender y apagar capas
folium.LayerControl().add_to(Map)

# 6. Guardamos el archivo
archivo_salida = "mapa_finca.html"
Map.save(archivo_salida)

print(f"¡Mapa generado con Folium puro! Abre el archivo '{archivo_salida}'.")
