import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Cargando datos con Polars...")

# 1. CARGA DE DATOS
df = pl.read_csv("historico_clima_finca.csv")

# 2. LIMPIEZA Y TRANSFORMACION
df = df.with_columns(
    pl.col("Fecha").str.to_date("%Y-%m-%d")
).drop_nulls()

df = df.with_columns([
    pl.col("Fecha").dt.month().alias("Mes"),
    pl.col("Fecha").dt.ordinal_day().alias("Dia_del_Ano") 
])

print("Creando memoria climatica (Lags)...")
columnas_clima = ["Precipitacion_mm", "Temperatura_C", "Humedad_Relativa_%", 
                  "Humedad_Suelo_Raices_m3_m3", "Radiacion_Solar_MJ_m2",
                  "Velocidad_Viento_m_s", "Humedad_Suelo_Superficie_m3_m3"]

lags = []
for col in columnas_clima:
    for dias_atras in [1, 2, 3]:
        lags.append(pl.col(col).shift(dias_atras).alias(f"{col}_ayer_{dias_atras}"))

df = df.with_columns(lags)

# 3. DEFINIR LOS OBJETIVOS (Lluvia y Temperatura de MANANA)
df = df.with_columns([
    pl.col("Precipitacion_mm").shift(-1).alias("Target_Lluvia_Manana"),
    pl.col("Temperatura_C").shift(-1).alias("Target_Temp_Manana")
])

df = df.drop_nulls()

# 4. PREPARAR MATRICES PARA MACHINE LEARNING
features = [col for col in df.columns if col not in ["Fecha", "Precipitacion_mm", "Temperatura_C", "Target_Lluvia_Manana", "Target_Temp_Manana"]]

X = df.select(features).to_numpy()
# Ahora 'y' tiene dos columnas a predecir
y = df.select(["Target_Lluvia_Manana", "Target_Temp_Manana"]).to_numpy()

corte = int(len(X) * 0.90) 
X_train, X_test = X[:corte], X[corte:]
y_train, y_test = y[:corte], y[corte:]

# 5. ENTRENAMIENTO DEL MODELO MULTIDIMENSIONAL
print(f"Entrenando modelo con {len(X_train)} dias de historia...")
modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)

# 6. EVALUACION
predicciones = modelo.predict(X_test)
mae_lluvia = mean_absolute_error(y_test[:, 0], predicciones[:, 0])
mae_temp = mean_absolute_error(y_test[:, 1], predicciones[:, 1])

print("\n=== RENDIMIENTO DEL MODELO ===")
print(f"Margen de error Lluvia (MAE): {mae_lluvia:.2f} mm")
print(f"Margen de error Temp (MAE): {mae_temp:.2f} C")

# 7. PRONOSTICO FINAL Y RECOMENDACION AGRICOLA
print("\n=========================================")
print("PRONOSTICO PARA EL DIA DE MANANA")
print("=========================================")

condiciones_hoy = X[-1].reshape(1, -1)
pred_hoy = modelo.predict(condiciones_hoy)[0]
pred_lluvia = pred_hoy[0]
pred_temp = pred_hoy[1]

print(f"Temperatura esperada: {pred_temp:.2f} C")
print(f"Precipitacion esperada: {pred_lluvia:.2f} mm\n")

# Logica agricola ajustada a tu observacion del verano
if pred_lluvia <= 8.0:
    print("-> CONDICION: Dia mayormente SECO o con lluvia no efectiva.")
    if pred_temp > 32.0:
        print("   ALERTA DE CALOR: Cualquier llovizna se evaporara de inmediato.")
        print("   NO es recomendable trasplantar el aji, pimenton o frijol hoy. Alto riesgo de estres hidrico.")
        print("   Manten las plantulas en la zona de germinacion con buen riego manual.")
    else:
        print("   El terreno seguira seco. Ideal para labores de limpieza o aplicar foliares.")
elif pred_lluvia <= 20.0:
    print("-> CONDICION: Lluvia MODERADA.")
    if pred_temp > 30.0:
        print("   Caera agua, pero hara calor (bochorno). Vigila la aparicion de hongos.")
    else:
        print("   Buen momento para preparar el terreno para los futuros trasplantes.")
else:
    print("-> CONDICION: AGUACERO FUERTE.")
    print("   Cuidado con el encharcamiento, especialmente si el terreno tiene mucha pendiente.")

print("=========================================")
