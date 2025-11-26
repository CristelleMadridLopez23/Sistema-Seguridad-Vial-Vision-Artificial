import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from ultralytics import YOLO
from pykalman import KalmanFilter

# --- CONFIGURACIÓN ---
VIDEO_PATH = "data/BDDA_REDUCED/BDDA_REDUCED/test/735.mp4" # <--- ¡PON TU VIDEO AQUÍ!
MODEL_PATH = "best_lstm_model.pth"    # Tu modelo entrenado
T_HIST = 20
T_PRED = 10
FEATURES = 2
HIDDEN_DIM = 128

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. DEFINICIÓN DE LA CLASE DEL MODELO (Idéntica al entrenamiento) ---
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

# --- 2. FUNCIONES HELPER ---
def apply_kalman_on_buffer(history_array):
    """Suaviza los 20 puntos del historial antes de enviarlos al modelo."""
    kf = KalmanFilter(initial_state_mean=history_array[0], n_dim_obs=2)
    (smoothed_means, _) = kf.smooth(history_array)
    return smoothed_means

# --- 3. CARGAR MODELO ---
print(f"Cargando modelo desde {MODEL_PATH}...")
model = TrajectoryPredictor(FEATURES, HIDDEN_DIM, FEATURES, T_PRED)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
except FileNotFoundError:
    print("ERROR: No se encuentra el archivo del modelo.")
    exit()

# --- 4. INICIALIZAR DETECTOR (YOLOv8) ---
# 'yolov8n.pt' es la versión nano (muy rápida). Se descargará sola la primera vez.
print("Cargando detector YOLOv8...")
detector = YOLO('yolov8n.pt') 

# --- 5. BUCLE DE VIDEO ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error al abrir el video: {VIDEO_PATH}")
    # Si quieres usar la webcam, cambia VIDEO_PATH por 0
    exit()

# Obtener dimensiones para normalizar
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Diccionario para guardar la historia de cada coche: {track_id: deque(maxlen=20)}
vehicle_histories = {}

print("Iniciando inferencia... Presiona 'Q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break # Fin del video

    # A. DETECCIÓN Y TRACKING
    # persist=True es importante para que el tracking (IDs) se mantenga entre frames
    results = detector.track(frame, persist=True, verbose=False, classes=[2, 5, 7]) # 2=car, 5=bus, 7=truck
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy() # Coordenadas cajas
        track_ids = results[0].boxes.id.int().cpu().numpy() # IDs únicos
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            
            # Calcular Centroide (Esto es lo que usamos para entrenar)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Gestionar memoria del vehículo
            if track_id not in vehicle_histories:
                vehicle_histories[track_id] = deque(maxlen=T_HIST)
            
            # Agregar posición actual
            vehicle_histories[track_id].append((cx, cy))
            
            # B. PREDICCIÓN (Solo si tenemos 20 puntos de historia)
            if len(vehicle_histories[track_id]) == T_HIST:
                # 1. Preparar datos
                history_arr = np.array(vehicle_histories[track_id])
                
                # 2. Suavizar con Kalman (Igual que en training)
                history_smooth = apply_kalman_on_buffer(history_arr)
                
                # 3. Normalizar (0 a 1)
                norm_history = history_smooth.copy()
                norm_history[:, 0] /= frame_width
                norm_history[:, 1] /= frame_height
                
                # 4. Convertir a Tensor
                input_tensor = torch.from_numpy(norm_history).float().unsqueeze(0).to(device)
                
                # 5. Inferencia LSTM
                with torch.no_grad():
                    prediction_norm = model(input_tensor).cpu().numpy()[0]
                
                # 6. Des-normalizar (Volver a píxeles)
                prediction_pixels = prediction_norm.copy()
                prediction_pixels[:, 0] *= frame_width
                prediction_pixels[:, 1] *= frame_height
                
                # C. DIBUJAR EN PANTALLA
                # Dibujar Historia (Verde)
                pts_hist = history_smooth.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts_hist], False, (0, 255, 0), 2)
                
                # Dibujar Predicción (Rojo)
                pts_pred = prediction_pixels.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts_pred], False, (0, 0, 255), 3)
                
                # Punto de conexión
                cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)

    # Mostrar frame
    cv2.imshow('Prediccion de Trayectoria LSTM', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()