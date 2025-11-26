import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from pykalman import KalmanFilter

# --- CONFIGURACIÓN ---
MODEL_FILE = "best_lstm_model.pth"  # El archivo generado por el script de PyTorch
TEST_DATA_FILE = "data/test_lstm_sequences.npz" # Asegúrate de tener este archivo o usa val_lstm_sequences.npz para probar

# Parámetros (Deben coincidir con los del entrenamiento)
T_HIST = 20  
T_PRED = 10  
FEATURES = 2 
HIDDEN_DIM = 128

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. DEFINICIÓN DE LA CLASE (Necesaria para cargar los pesos) ---
class TrajectoryPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pred_len):
        super(TrajectoryPredictor, self).__init__()
        self.pred_len = pred_len
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, c_n) = self.encoder(x)
        context_vector = h_n[-1]
        decoder_input = context_vector.unsqueeze(1).repeat(1, self.pred_len, 1)
        decoder_out, _ = self.decoder(decoder_input, (h_n, c_n))
        predictions = self.fc(decoder_out)
        return predictions

# --- 2. HELPER: KALMAN FILTER ---
def apply_kalman_smoothing(dataset_X):
    print(f"   -> Aplicando Kalman a {len(dataset_X)} muestras de prueba...")
    smoothed_dataset = np.zeros_like(dataset_X)
    kf = KalmanFilter(initial_state_mean=np.zeros(2), n_dim_obs=2)
    
    # Procesamos una muestra (sample) aleatoria para mostrar progreso si son muchos datos
    for i in range(len(dataset_X)):
        kf.initial_state_mean = dataset_X[i, 0, :]
        (smoothed_means, _) = kf.smooth(dataset_X[i])
        smoothed_dataset[i] = smoothed_means
    return smoothed_dataset

def run_evaluation():
    # 1. Cargar el Modelo
    print(f"Cargando estructura del modelo y pesos de: {MODEL_FILE}...")
    try:
        model = TrajectoryPredictor(FEATURES, HIDDEN_DIM, FEATURES, T_PRED)
        
        # Cargar los pesos (map_location asegura que cargue en CPU si no hay GPU)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        model.to(device)
        model.eval() # IMPORTANTE: Modo evaluación (congela dropout, etc.)
        print(" Modelo cargado exitosamente.")
        
    except FileNotFoundError:
        print(f" ERROR: No se encontró '{MODEL_FILE}'. Ejecuta el entrenamiento primero.")
        return

    # 2. Cargar Datos de Prueba
    print(f"Cargando datos de prueba: {TEST_DATA_FILE}...")
    try:
        data_test = np.load(TEST_DATA_FILE)
        X_test_raw = data_test['X']
        Y_test = data_test['Y']
        
        # APLICAR KALMAN (Crucial: El modelo aprendió con datos suaves, debe recibir datos suaves)
        X_test = apply_kalman_smoothing(X_test_raw)
        
        print(f"Datos cargados. Muestras: {X_test.shape[0]}")
        
    except FileNotFoundError:
        print(f" ERROR: No se encontró '{TEST_DATA_FILE}'. ¿Quizás quieres probar con 'val_lstm_sequences.npz'?")
        return

    # ----------------------------------------------------
    # I. PRUEBA CUANTITATIVA
    # ----------------------------------------------------
    print("\n--- I. EVALUACIÓN CUANTITATIVA (MSE) ---")
    
    # Convertir todo el set a Tensores para evaluación rápida
    X_tensor = torch.from_numpy(X_test).float().to(device)
    Y_tensor = torch.from_numpy(Y_test).float().to(device)
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        predictions = model(X_tensor)
        mse_loss = criterion(predictions, Y_tensor).item()
    
    rmse_norm = np.sqrt(mse_loss)
    # Estimación en píxeles (asumiendo frame 1280x720)
    ERROR_PIXELS_EST = (rmse_norm * 1280 + rmse_norm * 720) / 2 

    print(f"Resultados:")
    print(f"  MSE (Normalizado):  {mse_loss:.6f}")
    print(f"  RMSE (Normalizado): {rmse_norm:.6f}")
    print(f"  Error Estimado:     ~{ERROR_PIXELS_EST:.2f} píxeles")

    # ----------------------------------------------------
    # II. PRUEBA CUALITATIVA (Visualización)
    # ----------------------------------------------------
    print("\n--- II. PRUEBA CUALITATIVA ---")
    
    # Seleccionar muestra aleatoria
    idx = random.randint(0, X_test.shape[0] - 1)
    
    # Preparar inputs individuales
    # Necesitamos dimensión batch: (1, 20, 2)
    X_sample_tensor = X_tensor[idx].unsqueeze(0) 
    
    # Predecir
    with torch.no_grad():
        Y_pred_tensor = model(X_sample_tensor)
    
    # Convertir a Numpy para graficar (traer de GPU a CPU si es necesario)
    # Usamos X_test_raw para graficar la historia original (con ruido) o X_test (suave) según prefieras.
    # Usaré X_test (suave) para ver qué "vió" el modelo.
    history_plot = X_test[idx] 
    y_true_plot = Y_test[idx]
    y_pred_plot = Y_pred_tensor.cpu().numpy()[0] # Quitar dimensión batch

    # Crear trayectorias completas para conectar las líneas
    # Conectamos el último punto de la historia con el primero de la predicción
    last_hist_point = history_plot[-1].reshape(1, 2)
    
    full_true = np.concatenate([last_hist_point, y_true_plot], axis=0)
    full_pred = np.concatenate([last_hist_point, y_pred_plot], axis=0)

    # Gráfica
    plt.figure(figsize=(10, 8))
    
    # 1. Historia
    plt.plot(history_plot[:, 0], history_plot[:, 1], 'k--', label='Historia (Input Kalman)', linewidth=2)
    
    # 2. Futuro Real
    plt.plot(full_true[:, 0], full_true[:, 1], 'g-', label='Futuro Real', linewidth=2)
    plt.scatter(y_true_plot[:, 0], y_true_plot[:, 1], c='g', s=30)
    
    # 3. Predicción
    plt.plot(full_pred[:, 0], full_pred[:, 1], 'r-', label='Predicción Modelo', linewidth=2)
    plt.scatter(y_pred_plot[:, 0], y_pred_plot[:, 1], c='r', marker='x', s=50)

    # Punto actual
    plt.scatter(history_plot[-1, 0], history_plot[-1, 1], c='blue', marker='o', s=100, label='Vehículo Actual')

    plt.title(f"Predicción de Trayectoria (Muestra {idx})")
    plt.xlabel("X Normalizada")
    plt.ylabel("Y Normalizada")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis() # Coordenadas de imagen
    plt.show()

if __name__ == "__main__":
    run_evaluation()