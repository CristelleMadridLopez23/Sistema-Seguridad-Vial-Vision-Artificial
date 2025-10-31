import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# --- CONFIGURACIÓN ---
MODEL_FILE = "lstm_trajectory_predictor.h5"
TEST_DATA_FILE = "test_lstm_sequences.npz"

# Parámetros definidos en la fase de secuencias:
T_HIST = 20  # Timesteps de Input (Historia)
T_PRED = 10  # Timesteps de Output (Futuro)


def run_evaluation():
    """Carga el modelo y los datos de prueba para calcular la precisión y hacer una predicción de muestra."""
    
    # 1. Cargar el Modelo Entrenado
    print(f"Cargando modelo: {MODEL_FILE}...")
    try:
        model = tf.keras.models.load_model(MODEL_FILE, compile=False)
        model.compile(optimizer='adam', loss='mse')  # Compilar para evaluación
    except Exception as e:
        print(f"🚨 ERROR al cargar el modelo. Asegúrate que '{MODEL_FILE}' existe.")
        print(f"Detalle: {e}")
        return

    # 2. Cargar los Datos de Prueba
    print(f"Cargando datos de prueba: {TEST_DATA_FILE}...")
    try:
        data_test = np.load(TEST_DATA_FILE)
        X_test = data_test['X']
        Y_test = data_test['Y']
        print(f"Datos de prueba cargados. Total de muestras: {X_test.shape[0]}")
    except FileNotFoundError:
        print(f"🚨 ERROR: No se encontró el archivo de prueba '{TEST_DATA_FILE}'.")
        return

    # ----------------------------------------------------
    # I. PRUEBA CUANTITATIVA (Métrica de Precisión)
    # ----------------------------------------------------
    print("\n--- I. EVALUACIÓN CUANTITATIVA (MSE) ---")
    
    # Calcular el Error Cuadrático Medio (MSE) en todo el conjunto de prueba
    mse = model.evaluate(X_test, Y_test, verbose=1)
    
    # El MSE está en coordenadas NORMALIZADAS (0 a 1).
    # Para convertirlo a píxeles, asumiremos el tamaño de video (1280x720) y tomamos el promedio.
    # Usaremos el error promedio de desplazamiento por fotograma (Root Mean Squared Error o RMSE).
    
    # RMSE Normalizado: Raíz cuadrada del MSE
    rmse_norm = np.sqrt(mse)
    
    # Error promedio en píxeles (estimación rápida, asumiendo un ancho de 1280)
    # El MSE es la suma de los errores cuadrados de X e Y. Multiplicar por el ancho promedio
    # convierte este error normalizado de nuevo a un error aproximado en píxeles.
    ERROR_PIXELS_EST = (rmse_norm * 1280 + rmse_norm * 720) / 2 # Estima el error de la coordenada promedio

    print(f"\nResultados de Evaluación:")
    print(f"  MSE (Error Cuadrático Medio, Normalizado): {mse:.6f}")
    print(f"  RMSE (Raíz del MSE, Normalizado): {rmse_norm:.6f}")
    print(f"  Error de Predicción Estimado (Píxeles promedio): ~{ERROR_PIXELS_EST:.2f} píxeles")
    
    # ----------------------------------------------------
    # II. PRUEBA CUALITATIVA (Visualización de la Predicción)
    # ----------------------------------------------------
    print("\n--- II. PRUEBA CUALITATIVA (Comparación de Trayectoria) ---")
    
    # Seleccionar una muestra de prueba aleatoria
    sample_index = random.randint(0, X_test.shape[0] - 1)
    X_sample = X_test[sample_index:sample_index + 1] # Historia (Input)

    # Predecir el futuro con el modelo
    Y_pred = model.predict(X_sample)[0]
    
    # Obtener la verdad fundamental (Ground Truth) para la muestra
    Y_true = Y_test[sample_index]
    
    # Concatenar historia, predicción REAL y predicción MODELO para la gráfica
    # Tiempos: [T_HIST] Historia | [T_PRED] Futuro Real | [T_PRED] Futuro Predicho
    
    # Trayectoria Real Completa: Historia + Futuro Real
    full_true_trajectory = np.concatenate([X_sample[0], Y_true], axis=0)
    
    # Trayectoria Predicha: Historia + Futuro Predicho
    full_pred_trajectory = np.concatenate([X_sample[0], Y_pred], axis=0)

    # Graficar la trayectoria
    plt.figure(figsize=(10, 8))
    
    # Trazar la historia (puntos en común)
    plt.plot(X_sample[0][:, 0], X_sample[0][:, 1], 'k--', label='Historia (Input)', linewidth=2) 
    
    # Trazar la predicción REAL (Futuro)
    plt.plot(full_true_trajectory[:, 0], full_true_trajectory[:, 1], 'g-', label='Futuro Real (Ground Truth)', linewidth=2)
    plt.scatter(Y_true[:, 0], Y_true[:, 1], c='g', marker='o') # Puntos futuros reales
    
    # Trazar la predicción del MODELO (Futuro)
    plt.plot(full_pred_trajectory[:, 0], full_pred_trajectory[:, 1], 'r-', label='Predicción del Modelo', linewidth=2, alpha=0.7)
    plt.scatter(Y_pred[:, 0], Y_pred[:, 1], c='r', marker='x') # Puntos futuros predichos

    # Marcar el punto de inicio de la predicción (fin de la historia)
    plt.scatter(X_sample[0][-1, 0], X_sample[0][-1, 1], c='k', marker='D', s=100, label='Punto de Predicción (T=0)')

    plt.title(f"Comparación de Trayectorias (Muestra {sample_index})")
    plt.xlabel("Coordenada X Normalizada")
    plt.ylabel("Coordenada Y Normalizada")
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis() # Los píxeles Y aumentan hacia abajo
    plt.show()

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    run_evaluation()