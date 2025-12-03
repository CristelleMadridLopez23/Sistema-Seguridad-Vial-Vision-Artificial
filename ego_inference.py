import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from ultralytics import YOLO
from pykalman import KalmanFilter
import os
from datetime import datetime

# --- CONFIGURACIÓN ---
VIDEO_SOURCE = "00822.mp4"
MODEL_PATH = "best_lstm_model.pth"
T_HIST = 20
T_PRED = 10
FEATURES = 2
HIDDEN_DIM = 128

# Configuración de la zona del capó (ego-vehicle)
EGO_ZONE_HEIGHT_RATIO = 0.25
COLLISION_THRESHOLD = 80

# NUEVO: Configuración para guardar imágenes
SAVE_COLLISION_IMAGES = True
COLLISION_IMAGES_FOLDER = "collision_alerts"  # Carpeta donde se guardarán las imágenes
SAVE_INTERVAL_FRAMES = 15  # Guardar cada X frames durante una colisión (para evitar spam)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# --- CREAR CARPETA PARA IMÁGENES DE COLISIÓN ---
if SAVE_COLLISION_IMAGES:
    if not os.path.exists(COLLISION_IMAGES_FOLDER):
        os.makedirs(COLLISION_IMAGES_FOLDER)
        print(f"✓ Carpeta creada: {COLLISION_IMAGES_FOLDER}/")
    else:
        print(f"✓ Usando carpeta existente: {COLLISION_IMAGES_FOLDER}/")

# --- 1. MODELO ---
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
    try:
        kf = KalmanFilter(initial_state_mean=history_array[0], n_dim_obs=2)
        (smoothed_means, _) = kf.smooth(history_array)
        return smoothed_means
    except Exception as e:
        return history_array

def define_ego_zone(frame_width, frame_height, height_ratio):
    """
    Define la zona del capó (ego-vehicle) en la parte inferior del frame.
    
    Returns:
        ego_zone: dict con 'y_top', 'y_bottom', 'x_left', 'x_right', 'center'
    """
    y_top = int(frame_height * (1 - height_ratio))
    y_bottom = frame_height
    x_left = 0
    x_right = frame_width
    
    center_x = frame_width // 2
    center_y = int(frame_height * (1 - height_ratio / 2))
    
    return {
        'y_top': y_top,
        'y_bottom': y_bottom,
        'x_left': x_left,
        'x_right': x_right,
        'center': (center_x, center_y)
    }

def check_collision_with_ego_zone(predictions_dict, ego_zone, threshold):
    """
    Verifica si alguna trayectoria predicha colisionará con la zona del capó.
    
    Args:
        predictions_dict: dict {track_id: prediction_array (T_PRED, 2)}
        ego_zone: dict con la definición de la zona del capó
        threshold: distancia mínima para considerar colisión
    
    Returns:
        collision_detected: bool
        colliding_ids: set de IDs que colisionarán
        min_distances: dict {track_id: distancia mínima a la zona del capó}
    """
    collision_detected = False
    colliding_ids = set()
    min_distances = {}
    
    ego_center = np.array(ego_zone['center'])
    ego_y_top = ego_zone['y_top']
    
    for track_id, prediction in predictions_dict.items():
        distances = np.linalg.norm(prediction - ego_center, axis=1)
        min_dist = np.min(distances)
        min_distances[track_id] = min_dist
        
        points_in_zone = prediction[:, 1] > ego_y_top
        
        if min_dist < threshold or np.any(points_in_zone):
            collision_detected = True
            colliding_ids.add(track_id)
    
    return collision_detected, colliding_ids, min_distances

def draw_ego_zone(frame, ego_zone, collision_detected=False):
    """Dibuja la zona del capó en el frame."""
    y_top = ego_zone['y_top']
    y_bottom = ego_zone['y_bottom']
    x_left = ego_zone['x_left']
    x_right = ego_zone['x_right']
    
    color = (0, 0, 255) if collision_detected else (255, 200, 0)
    
    cv2.line(frame, (x_left, y_top), (x_right, y_top), color, 3)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_left, y_top), (x_right, y_bottom), color, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    
    label = "ZONA DEL CAPO (EGO VEHICLE)"
    cv2.putText(frame, label, (10, y_top - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.circle(frame, ego_zone['center'], 8, color, -1)

def save_collision_frame(frame, frame_count, colliding_ids, min_distances, folder):
    """
    Guarda el frame actual como imagen con información de la colisión.
    
    Args:
        frame: Frame actual a guardar
        frame_count: Número del frame
        colliding_ids: Set de IDs de vehículos en colisión
        min_distances: Dict con distancias mínimas de cada vehículo
        folder: Carpeta donde guardar la imagen
    """
    # Generar timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crear nombre de archivo descriptivo
    ids_str = "_".join([str(id) for id in sorted(colliding_ids)])
    min_dist = min(min_distances.values()) if min_distances else 0
    
    filename = f"collision_frame{frame_count}_{timestamp}_IDs{ids_str}_dist{min_dist:.0f}px.jpg"
    filepath = os.path.join(folder, filename)
    
    # Guardar imagen
    cv2.imwrite(filepath, frame)
    
    return filepath

# --- 3. CARGAR MODELO ---
print(f"Cargando modelo desde {MODEL_PATH} (T_PRED={T_PRED})...")
model = TrajectoryPredictor(FEATURES, HIDDEN_DIM, FEATURES, T_PRED)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✓ Modelo cargado exitosamente.")
except RuntimeError as e:
    print(f"\n✗ ERROR DE COMPATIBILIDAD DEL MODELO: {e}")
    exit()
except FileNotFoundError:
    print(f"✗ ERROR: No se encuentra el archivo del modelo: {MODEL_PATH}")
    exit()

# --- 4. INICIALIZAR DETECTOR ---
print("Cargando detector YOLOv8...")
detector = YOLO('yolov8n.pt')

# --- 5. BUCLE PRINCIPAL ---
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"✗ Error: No se pudo abrir la fuente de video: {VIDEO_SOURCE}.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

ego_zone = define_ego_zone(frame_width, frame_height, EGO_ZONE_HEIGHT_RATIO)

vehicle_histories = {}
print(f"✓ Fuente de Video Abierta ({frame_width}x{frame_height}).")
print(f"✓ Zona del capó definida: Y superior = {ego_zone['y_top']} px")
print(f"✓ Predicción a {T_PRED} frames (~{T_PRED/12:.1f}s).")
if SAVE_COLLISION_IMAGES:
    print(f"✓ Guardado de imágenes activado en: {COLLISION_IMAGES_FOLDER}/")
print("Presiona 'Q' para salir.\n")

frame_count = 0
last_saved_frame = -SAVE_INTERVAL_FRAMES  # Para controlar el intervalo de guardado
total_images_saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o error al leer frame.")
        break

    frame_count += 1
    current_width = frame.shape[1]
    current_height = frame.shape[0]
    
    if current_width != frame_width or current_height != frame_height:
        ego_zone = define_ego_zone(current_width, current_height, EGO_ZONE_HEIGHT_RATIO)
        frame_width, frame_height = current_width, current_height
    
    current_frame_predictions = {}
    
    # A. DETECCIÓN Y TRACKING
    results = detector.track(frame, persist=True, verbose=False, classes=[2, 5, 7])
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            if cy > ego_zone['y_top']:
                continue
            
            if track_id not in vehicle_histories:
                vehicle_histories[track_id] = deque(maxlen=T_HIST)
            
            vehicle_histories[track_id].append((cx, cy))
            
            x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1_i, y1_i - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # B. PREDICCIÓN
            if len(vehicle_histories[track_id]) == T_HIST:
                history_arr = np.array(vehicle_histories[track_id])
                history_smooth = apply_kalman_on_buffer(history_arr)
                
                norm_history = history_smooth.copy()
                norm_history[:, 0] /= current_width
                norm_history[:, 1] /= current_height
                
                input_tensor = torch.from_numpy(norm_history).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction_norm = model(input_tensor).cpu().numpy()[0]
                
                prediction_pixels = prediction_norm.copy()
                prediction_pixels[:, 0] *= current_width
                prediction_pixels[:, 1] *= current_height
                
                current_frame_predictions[track_id] = prediction_pixels
                
                # C. VISUALIZACIÓN
                pts_pred = prediction_pixels.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts_pred], False, (0, 0, 255), 3)
                
                pts_hist = history_smooth.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts_hist], False, (0, 255, 0), 2)
                
                cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)

    # --- D. VERIFICACIÓN DE COLISIONES ---
    collision_flag = False
    colliding_ids = set()
    min_distances = {}
    
    if len(current_frame_predictions) > 0:
        collision_flag, colliding_ids, min_distances = check_collision_with_ego_zone(
            current_frame_predictions, ego_zone, COLLISION_THRESHOLD
        )

    # --- E. DIBUJAR ZONA DEL CAPÓ ---
    draw_ego_zone(frame, ego_zone, collision_flag)

    # --- F. ADVERTENCIAS VISUALES ---
    if collision_flag:
        WARNING_TEXT = "¡RIESGO DE COLISION CON MI VEHICULO!"
        text_size = cv2.getTextSize(WARNING_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (current_width - text_size[0]) // 2
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (current_width, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, WARNING_TEXT, (text_x, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        
        if frame_count % 10 < 5:
            cv2.rectangle(frame, (0, 0), (current_width-1, current_height-1), (0, 0, 255), 15)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                if track_id in colliding_ids:
                    x1, y1, x2, y2 = box.astype(int)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 6)
                    
                    dist = min_distances.get(track_id, 0)
                    warning_label = f"¡PELIGRO! Dist: {dist:.0f}px"
                    cv2.putText(frame, warning_label, (x1, y2 + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
                    
                    vehicle_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    cv2.line(frame, vehicle_center, ego_zone['center'], (0, 140, 255), 3)
        
        # --- G. GUARDAR IMAGEN DE COLISIÓN ---
        if SAVE_COLLISION_IMAGES and (frame_count - last_saved_frame >= SAVE_INTERVAL_FRAMES):
            filepath = save_collision_frame(frame, frame_count, colliding_ids, min_distances, COLLISION_IMAGES_FOLDER)
            last_saved_frame = frame_count
            total_images_saved += 1
            print(f"[ALERTA] Frame {frame_count}: Colisión detectada. Imagen guardada: {filepath}")
    
    # Información en pantalla
    info_text = f"Frame: {frame_count} | Vehiculos: {len(current_frame_predictions)} | Colision: {'SI' if collision_flag else 'NO'}"
    if SAVE_COLLISION_IMAGES:
        info_text += f" | Imgs guardadas: {total_images_saved}"
    
    cv2.putText(frame, info_text, (10, current_height - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Mostrar frame
    cv2.imshow('Deteccion de Colision - Perspectiva Ego Vehicle', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✓ Procesamiento completado.")
print(f"✓ Total de frames procesados: {frame_count}")
print(f"✓ Total de imágenes de colisión guardadas: {total_images_saved}")
if total_images_saved > 0:
    print(f"✓ Imágenes guardadas en: {COLLISION_IMAGES_FOLDER}/")