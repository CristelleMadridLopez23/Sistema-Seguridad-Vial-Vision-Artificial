import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pykalman import KalmanFilter
import time
import os

# --- CONFIGURACIÓN ---
T_HIST = 20
T_PRED = 10
FEATURES = 2
BATCH_SIZE = 128
HIDDEN_DIM = 128
EPOCHS = 50
LEARNING_RATE = 0.001

# Configurar dispositivo (GPU si es posible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Usando dispositivo: {device}")

# --- 1. PRE-PROCESAMIENTO (KALMAN) ---
def apply_kalman_smoothing(dataset_X):
    print(f"   -> Aplicando Kalman a {len(dataset_X)} muestras...")
    start_time = time.time()
    smoothed_dataset = np.zeros_like(dataset_X)
    
    # Configuración Kalman
    kf = KalmanFilter(initial_state_mean=np.zeros(2), n_dim_obs=2)

    # Optimización simple para no tardar una eternidad
    for i in range(len(dataset_X)):
        kf.initial_state_mean = dataset_X[i, 0, :]
        (smoothed_means, _) = kf.smooth(dataset_X[i])
        smoothed_dataset[i] = smoothed_means
        
        if i % 20000 == 0 and i > 0:
            print(f"      {i}/{len(dataset_X)} procesados...")
            
    print(f"   -> Kalman finalizado en {time.time() - start_time:.2f}s")
    return smoothed_dataset

# --- 2. DEFINICIÓN DEL MODELO (Encoder-Decoder) ---
class TrajectoryPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pred_len):
        super(TrajectoryPredictor, self).__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        # ENCODER: Procesa la historia
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # DECODER: Genera el futuro
        # La entrada del decoder será el vector repetido del encoder
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # SALIDA: Regresión lineal para coordenadas
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # --- Encoder ---
        # encoder_out: (batch, seq_len, hidden)
        # h_n, c_n: (num_layers, batch, hidden) - Estado final
        _, (h_n, c_n) = self.encoder(x)
        
        # Tomamos el último estado oculto como "Context Vector"
        # h_n[-1] toma la última capa (si hubiera varias)
        context_vector = h_n[-1]  # Shape: (batch, hidden)
        
        # --- Puente (RepeatVector) ---
        # Repetimos el contexto T_PRED veces para crear la entrada del decoder
        # Shape deseada: (batch, T_PRED, hidden)
        decoder_input = context_vector.unsqueeze(1).repeat(1, self.pred_len, 1)
        
        # --- Decoder ---
        # Pasamos el estado interno (h_n, c_n) del encoder para mantener continuidad
        decoder_out, _ = self.decoder(decoder_input, (h_n, c_n))
        
        # --- Salida ---
        # Aplicamos la capa lineal a toda la secuencia
        predictions = self.fc(decoder_out) # Shape: (batch, T_PRED, features)
        
        return predictions

# --- 3. CARGA Y PREPARACIÓN DE DATOS ---
try:
    print(" Cargando datos .npz...")
    data_train = np.load('data/train_lstm_sequences.npz')
    X_train_raw, Y_train = data_train['X'], data_train['Y']
    
    data_val = np.load('data/val_lstm_sequences.npz')
    X_val_raw, Y_val = data_val['X'], data_val['Y']

    # Aplicar Kalman (Solo a Inputs X)
    X_train = apply_kalman_smoothing(X_train_raw)
    X_val = apply_kalman_smoothing(X_val_raw)

    # Convertir a Tensores PyTorch
    # .float() es importante porque los pesos del modelo son float32
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float())

    # DataLoaders (Manejan los batches automáticamente)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

except FileNotFoundError:
    print(" Error: No se encuentran los archivos .npz")
    exit()

# --- 4. INICIALIZACIÓN ---
model = TrajectoryPredictor(input_dim=FEATURES, hidden_dim=HIDDEN_DIM, output_dim=FEATURES, pred_len=T_PRED)
model.to(device) # Mover modelo a GPU

criterion = nn.MSELoss() # Error Cuadrático Medio
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. BUCLE DE ENTRENAMIENTO ---
print("\n Iniciando entrenamiento en PyTorch...")
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # --- Fase de Entrenamiento ---
    model.train()
    train_loss = 0
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device) # Mover datos a GPU
        
        optimizer.zero_grad()           # 1. Limpiar gradientes
        outputs = model(batch_X)        # 2. Predicción (Forward)
        loss = criterion(outputs, batch_Y) # 3. Calcular error
        loss.backward()                 # 4. Backpropagation
        optimizer.step()                # 5. Actualizar pesos
        
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # --- Fase de Validación ---
    model.eval()
    val_loss = 0
    with torch.no_grad(): # No calcular gradientes en validación (ahorra memoria)
        for batch_X, batch_Y in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] \t Train Loss: {avg_train_loss:.6f} \t Val Loss: {avg_val_loss:.6f}")

    # Guardar el mejor modelo
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_lstm_model.pth")
        print("  💾 Modelo guardado (Mejor Val Loss)")

print("\n Entrenamiento finalizado.")