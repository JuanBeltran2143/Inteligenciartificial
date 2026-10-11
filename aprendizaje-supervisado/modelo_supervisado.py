import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def generar_dataset_simulado():
    """Genera y exporta un dataset simulado de tiempos de viaje en transporte masivo."""
    np.random.seed(42)
    n_muestras = 500
    
    distancia = np.round(np.random.uniform(2.0, 25.0, n_muestras), 2)
    transbordos = np.random.choice([0, 1, 2], size=n_muestras, p=[0.5, 0.4, 0.1])
    hora_pico = np.random.choice([0, 1], size=n_muestras, p=[0.6, 0.4])
    espera = np.random.uniform(2.0, 10.0, n_muestras)
    
    # Ecuación de simulación con variables operativas y ruido aleatorio
    tiempo_real = (
        (distancia * 1.8) + 
        (transbordos * 5.5) + 
        (hora_pico * 12.0) + 
        espera + 
        np.random.normal(0, 2.0, n_muestras)
    )
    tiempo_real = np.round(np.maximum(tiempo_real, 5.0), 2)
    
    df = pd.DataFrame({
        'distancia_km': distancia,
        'num_transbordos': transbordos,
        'es_hora_pico': hora_pico,
        'tiempo_espera_promedio_min': np.round(espera, 2),
        'tiempo_viaje_real_min': tiempo_real
    })
    
    archivo_csv = "dataset_transporte.csv"
    df.to_csv(archivo_csv, index=False)
    print(f"[OK] Dataset generado exitosamente como '{archivo_csv}' con {n_muestras} registros.")
    return archivo_csv, df

def entrenar_y_graficar(ruta_csv, df):
    """Entrena los modelos supervisados, evalúa su rendimiento y genera el gráfico."""
    X = df[['distancia_km', 'num_transbordos', 'es_hora_pico', 'tiempo_espera_promedio_min']]
    y = df['tiempo_viaje_real_min']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n" + "="*50)
    print("      ENTRENAMIENTO DE MODELOS SUPERVISADOS     ")
    print("="*50)
    
    # 1. Regresión Lineal
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    
    print("\n[Modelo A: Regresión Lineal]")
    print(f" -> Error Cuadrático Medio (MSE): {mean_squared_error(y_test, y_pred_lr):.2f}")
    print(f" -> Coeficiente de Determinación (R2 Score): {r2_score(y_test, y_pred_lr):.2f}")
    
    # 2. Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    print("\n[Modelo B: Random Forest Regressor]")
    print(f" -> Error Cuadrático Medio (MSE): {mean_squared_error(y_test, y_pred_rf):.2f}")
    print(f" -> Coeficiente de Determinación (R2 Score): {r2_score(y_test, y_pred_rf):.2f}")
    
    # Prueba de inferencia en vivo
    print("\n" + "-"*50)
    print(" PRUEBA DE PREDICCIÓN EN TIEMPO REAL ")
    print("-"*50)
    nuevo_viaje = pd.DataFrame([[14.5, 1, 1, 4.0]], columns=X.columns)
    prediccion = rf_model.predict(nuevo_viaje)[0]
    print(f" Datos de entrada: Distancia=14.5 km | Transbordos=1 | Hora Pico=Sí | Espera=4 min")
    print(f" > Tiempo de viaje estimado por el modelo: {prediccion:.2f} minutos\n")
    
    # Generación automática del Gráfico de Regresión Lineal (Distancia vs Tiempo Real)
    print("[...] Generando gráfico de regresión lineal en ventana emergente...")
    plt.figure(figsize=(8, 5))
    plt.scatter(df['distancia_km'], df['tiempo_viaje_real_min'], color='royalblue', alpha=0.6, label='Datos Históricos')
    
    # Línea de tendencia estimada usando únicamente la distancia para fines visuales
    lr_simple = LinearRegression()
    lr_simple.fit(df[['distancia_km']], df['tiempo_viaje_real_min'])
    x_range = np.linspace(df['distancia_km'].min(), df['distancia_km'].max(), 100).reshape(-1, 1)
    y_range_pred = lr_simple.predict(x_range)
    
    plt.plot(x_range, y_range_pred, color='crimson', linewidth=2.5, label='Línea de Regresión Lineal')
    plt.title('Modelo Supervisado: Relación Distancia vs Tiempo de Viaje', fontsize=12, fontweight='bold')
    plt.xlabel('Distancia del Trayecto (km)', fontsize=10)
    plt.ylabel('Tiempo Real de Viaje (minutos)', fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_path, dataframe = generar_dataset_simulado()
    entrenar_y_graficar(csv_path, dataframe)