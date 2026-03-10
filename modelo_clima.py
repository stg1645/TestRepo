import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

print("Cargando datos con Polars...")

# 1. CARGA DE DATOS
# Polars carga el CSV a la velocidad de la luz
df = pl.read_csv("historico_clima_finca.csv")

# 2. LIMPIEZA Y TRANSFORMACIÓN (Feature Engineering)
# Convertimos la fecha y creamos variables que el modelo sí entiende
df = df.with_columns(
    pl.col("Fecha").str.to_date("%Y-%m-%d")
).drop_nulls()

df = df.with_columns([
    pl.col("Fecha").dt.month().alias("Mes"),
    pl.col("Fecha").dt.ordinal_day().alias("Dia_del_Año") # Capta la estacionalidad
])

print("Creando memoria climática (Lags)...")
# Creamos variables del pasado para predecir el futuro (ej. qué pasó hace 1, 2 y 3 días)
columnas_clima = ["Precipitacion_mm", "Temperatura_C", "Humedad_Relativa_%", 
                  "Humedad_Suelo_Raices_m3_m3", "Radiacion_Solar_MJ_m2"]

lags = []
for col in columnas_clima:
    for dias_atras in [1, 2, 3]:
        lags.append(pl.col(col).shift(dias_atras).alias(f"{col}_ayer_{dias_atras}"))

df = df.with_columns(lags)

# 3. DEFINIR EL OBJETIVO (Target)
# Queremos predecir la lluvia de MAÑANA
df = df.with_columns(
    pl.col("Precipitacion_mm").shift(-1).alias("Target_Lluvia_Mañana")
)

# Eliminamos las filas donde no tenemos historial suficiente (los primeros 3 días) o el target (el último día)
df = df.drop_nulls()

# 4. PREPARAR MATRICES PARA MACHINE LEARNING
# Scikit-learn necesita formato Numpy
features = [col for col in df.columns if col not in ["Fecha", "Precipitacion_mm", "Target_Lluvia_Mañana"]]

X = df.select(features).to_numpy()
y = df.select("Target_Lluvia_Mañana").to_numpy().ravel()

# Separar en Entrenamiento (Pasado) y Prueba (El último año)
# NUNCA uses train_test_split aleatorio en series de tiempo climáticas
corte = int(len(X) * 0.90) # Entrenamos con el 90% (aprox 13.5 años) y probamos con el último 10%
X_train, X_test = X[:corte], X[corte:]
y_train, y_test = y[:corte], y[corte:]

# 5. ENTRENAMIENTO DEL MODELO (Random Forest)
print(f"Entrenando modelo con {len(X_train)} días de historia...")
modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)

# 6. EVALUACIÓN Y RESULTADOS
predicciones = modelo.predict(X_test)

mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))

print("\n=== RENDIMIENTO DEL MODELO ===")
print(f"Error Medio Absoluto (MAE): {mae:.2f} mm de lluvia")
print(f"Error Cuadrático Medio (RMSE): {rmse:.2f} mm")

# ¿Qué variables determinan realmente si llueve en tu finca?
importancias = list(zip(features, modelo.feature_importances_))
importancias.sort(key=lambda x: x[1], reverse=True)

print("\n=== VARIABLES MÁS IMPORTANTES PARA PREDECIR ===")
for nombre, peso in importancias[:5]:
    print(f"- {nombre}: {peso*100:.1f}%")

print("\nModelo entrenado con éxito. ¡Listo para planificar siembras!")
