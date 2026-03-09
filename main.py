import ee
import folium

print("Conectando con Google Earth Engine...")
# Asegúrate de usar tu ID de proyecto real aquí
ee.Initialize(project='finca-489704')
print("¡Conexión exitosa!")

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
