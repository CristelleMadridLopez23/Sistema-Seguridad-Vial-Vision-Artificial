import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape
import numpy as np

# --- CONFIGURACIÓN ---
# Parámetros definidos en la fase de secuencias:
T_HIST = 20  # Timesteps de Input (Historia)
T_PRED = 10  # Timesteps de Output (Futuro)
FEATURES = 2 # Features: (x_norm, y_norm)

# --- CARGA DE DATOS ---
print("Cargando datos de entrenamiento y validación...")

try:
    # Cargar datos de entrenamiento
    data_train = np.load('train_lstm_sequences.npz')
    X_train = data_train['X']  # Input: (213546, 20, 2)
    Y_train = data_train['Y']  # Output: (213546, 10, 2)

    # Cargar datos de validación
    data_val = np.load('val_lstm_sequences.npz')
    X_val = data_val['X']      # Input: (52680, 20, 2)
    Y_val = data_val['Y']      # Output: (52680, 10, 2)

except FileNotFoundError:
    print("🚨 ERROR: Asegúrese de que los archivos .npz estén en el mismo directorio.")
    exit()

# --- CONSTRUCCIÓN DEL MODELO ---
print("Construyendo el Modelo LSTM...")

model = Sequential([
    # 1. Capa LSTM: Procesa la secuencia de 20 timesteps
    # 128 unidades es un tamaño inicial robusto. return_sequences=False por ahora, 
    # ya que vamos a aplanar la salida para el futuro.
    LSTM(128, activation='relu', input_shape=(T_HIST, FEATURES)),
    
    # 2. Capa Densa: Mapea el estado interno del LSTM a la longitud total del vector de salida
    # La salida final debe ser 10 timesteps * 2 features = 20 valores
    Dense(T_PRED * FEATURES, activation='linear'),
    
    # 3. Capa Reshape: Reorganiza el vector de 20 valores en la matriz de salida deseada (10, 2)
    Reshape((T_PRED, FEATURES)) 
])

# --- COMPILACIÓN DEL MODELO ---
# Usamos 'adam' y 'mse' (Error Cuadrático Medio) porque es un problema de regresión (predicción de coordenadas).
model.compile(optimizer='adam', loss='mse')

model.summary()

# --- ENTRENAMIENTO ---
print("\n--- INICIANDO ENTRENAMIENTO ---")
history = model.fit(
    X_train, Y_train,
    epochs=50,             # Se puede ajustar. 50 es un buen punto de partida.
    batch_size=128,        # Se puede ajustar. 128 usa más memoria pero acelera el entrenamiento.
    validation_data=(X_val, Y_val), # Usar los datos de validación
    verbose=1
)

# --- GUARDAR EL MODELO ---
MODEL_FILE = "lstm_trajectory_predictor"  # Sin .h5
model.save(MODEL_FILE, save_format='keras')  # Formato más compatible
print(f"\n🎉 Entrenamiento completado. Modelo guardado como {MODEL_FILE}")