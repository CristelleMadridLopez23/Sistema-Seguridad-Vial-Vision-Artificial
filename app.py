import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from ultralytics import YOLO
from pykalman import KalmanFilter
import os
from datetime import datetime
import threading
import time # Para pausar la actualización del stream

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="IA Anti-Colisión (Multithreaded)", page_icon="🚗", layout="wide")

# --- ESTILOS CSS ---
# (Se mantiene el CSS para las tarjetas y alertas)
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E; padding: 15px; border-radius: 10px;
        border: 1px solid #333; text-align: center;
    }
    .danger-alert {
        background-color: #ff4b4b; color: white; padding: 10px;
        border-radius: 5px; text-align: center; font-weight: bold;
        animation: blinker 1s linear infinite;
    }
    @keyframes blinker { 50% { opacity: 0; } }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURACIÓN GLOBAL ---
FEATURES = 2
HIDDEN_DIM = 128
T_HIST = 20
T_PRED = 10
COLLISION_IMAGES_FOLDER = "collision_alerts"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MODELO LSTM Y CARGA (SIN CAMBIOS) ---
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
        return self.fc(decoder_out)

@st.cache_resource
def load_models(model_path):
    detector = YOLO('yolov8n.pt')
    lstm = TrajectoryPredictor(FEATURES, HIDDEN_DIM, FEATURES, T_PRED)
    try:
        lstm.load_state_dict(torch.load(model_path, map_location=device))
        lstm.to(device).eval()
        return detector, lstm, "✓ Modelos listos"
    except Exception as e:
        return None, None, f"Error: {e}"

# --- 3. CLASE DE ESTADO COMPARTIDO (Thread-Safe) ---
class SharedState:
    """Almacena el frame y las métricas de forma segura entre hilos."""
    def __init__(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8) # Frame por defecto
        self.is_collision = False
        self.min_dist = 0
        self.vehicles_count = 0
        self.lock = threading.Lock()
        
    def update(self, frame, collision, dist, count):
        with self.lock:
            # Almacenar el frame YA en RGB para Streamlit
            self.frame = frame
            self.is_collision = collision
            self.min_dist = dist
            self.vehicles_count = count
            
    def get_state(self):
        with self.lock:
            return self.frame.copy(), self.is_collision, self.min_dist, self.vehicles_count

# --- 4. CLASE TRABAJADORA (Worker Thread) ---
class DetectionWorker(threading.Thread):
    def __init__(self, models, configs, shared_state):
        super().__init__()
        self.detector, self.lstm_model = models
        self.configs = configs
        self.state = shared_state
        self.stop_event = threading.Event()
        self.histories = {} # Las historias de los vehículos viven en este hilo

    def run(self):
        cap = cv2.VideoCapture(self.configs['video_source'])
        if not cap.isOpened():
            st.error(f"Error: No se pudo abrir la fuente de video: {self.configs['video_source']}")
            self.stop_event.set()
            return

        f_count, saved_count, last_save = 0, 0, -100
        
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret: break
            f_count += 1
            
            # --- SALTO DE FRAMES Y LÓGICA PESADA ---
            if f_count % self.configs['frame_skip'] != 0:
                continue # Saltar la detección para acelerar

            h, w = frame.shape[:2]
            zone = self._define_ego_zone(w, h, self.configs['ego_ratio'])
            
            # 1. Tracking YOLO
            results = self.detector.track(frame, persist=True, verbose=False, classes=[2, 5, 7])
            preds = {}
            
            # ... (Lógica de Detección, LSTM, Kalman, y Dibujo, similar al código anterior) ...
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().numpy()

                for box, tid in zip(boxes, ids):
                    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                    if cy > zone['y_top']: continue

                    if tid not in self.histories: self.histories[tid] = deque(maxlen=T_HIST)
                    self.histories[tid].append((cx, cy))
                    
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

                    if len(self.histories[tid]) == T_HIST:
                        hist = np.array(self.histories[tid])
                        smooth = self._apply_kalman(hist)

                        # DIBUJAR HISTORIA (LÍNEA VERDE)
                        cv2.polylines(frame, [smooth.astype(np.int32).reshape((-1, 1, 2))], False, (0, 255, 0), 2)
                        
                        # Predicción
                        norm = smooth.copy(); norm[:,0]/=w; norm[:,1]/=h
                        inp = torch.from_numpy(norm).float().unsqueeze(0).to(device)
                        with torch.no_grad(): pred = self.lstm_model(inp).cpu().numpy()[0]
                        
                        pred[:,0]*=w; pred[:,1]*=h
                        preds[tid] = pred
                        
                        # Dibujar Predicción (Línea Roja)
                        cv2.polylines(frame, [pred.astype(np.int32).reshape((-1, 1, 2))], False, (0, 0, 255), 3)

            is_coll, _, dists = self._check_collision(preds, zone, self.configs['coll_thresh'])
            
            # Dibujar Zona
            color = (0, 0, 255) if is_coll else (255, 200, 0)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, zone['y_top']), (w, h), color, -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.line(frame, (0, zone['y_top']), (w, zone['y_top']), color, 3)

            min_d = min(dists.values()) if dists else 0
            
            # 2. ACTUALIZAR ESTADO COMPARTIDO (Thread-Safe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Conversión aquí!
            self.state.update(frame_rgb, is_coll, min_d, len(preds))

            # Guardar Alerta (Opcional, sin cambiar el stream)
            if is_coll and self.configs['save_imgs'] and (f_count - last_save > 15):
                 self._save_alert(frame, f_count, COLLISION_IMAGES_FOLDER)
                 last_save = f_count; saved_count += 1
        
        cap.release()
        
    def stop(self):
        self.stop_event.set()
        
    # Copias de las funciones helper necesarias en el Worker:
    def _apply_kalman(self, history): return apply_kalman(history)
    def _define_ego_zone(self, w, h, ratio): return define_ego_zone(w, h, ratio)
    def _check_collision(self, preds, zone, thresh): return check_collision(preds, zone, thresh)
    def _save_alert(self, frame, count, folder): return save_alert(frame, count, folder)

# --- 5. INTERFAZ Y BUCLE PRINCIPAL (STREAMLIT) ---

# Funciones helper (Mantenerlas fuera del Worker por conveniencia)
def apply_kalman(history):
    try:
        kf = KalmanFilter(initial_state_mean=history[0], n_dim_obs=2)
        (smoothed, _) = kf.smooth(history)
        return smoothed
    except: return history

def define_ego_zone(w, h, ratio):
    y_top = int(h * (1 - ratio))
    return {'y_top': y_top, 'y_bottom': h, 'center': (w // 2, int(h * (1 - ratio / 2)))}

def check_collision(preds, zone, thresh):
    colliding, min_dists, collision = set(), {}, False
    center = np.array(zone['center'])
    for tid, pred in preds.items():
        min_d = np.min(np.linalg.norm(pred - center, axis=1))
        min_dists[tid] = min_d
        if min_d < thresh or np.any(pred[:, 1] > zone['y_top']):
            collision = True; colliding.add(tid)
    return collision, colliding, min_dists

def save_alert(frame, count, folder):
    if not os.path.exists(folder): os.makedirs(folder)
    path = os.path.join(folder, f"alert_{count}_{datetime.now().strftime('%H%M%S')}.jpg")
    cv2.imwrite(path, frame)
    return path

# --- Interfaz ---
video_source, model_path = "00123.mp4", "best_lstm_model.pth"

# Controles de Velocidad y Detección
st.sidebar.subheader("Rendimiento")
frame_skip = st.sidebar.slider("Saltar Frames (Velocidad)", 1, 5, 2)

st.sidebar.subheader("Detección")
ego_ratio = st.sidebar.slider("Altura Zona (%)", 0.1, 0.5, 0.25)
coll_thresh = st.sidebar.slider("Umbral (px)", 30, 200, 80)
save_imgs = st.sidebar.checkbox("Guardar Fotos", value=True)

start_btn = st.sidebar.button("▶ INICIAR", type="primary", key="start_button")
stop_btn = st.sidebar.button("⏹ DETENER", key="stop_button")

detector, lstm_model, msg = load_models(model_path)
if not lstm_model: st.error(msg); st.stop()

# Layout
col1, col2 = st.columns([3, 1])
with col1: video_placeholder = st.empty()
with col2: 
    st.markdown("### 📊 Estado")
    status_ph = st.empty(); metrics_ph = st.empty()


# Inicialización de estado en Streamlit Session
if 'worker' not in st.session_state:
    st.session_state.worker = None
if 'shared_state' not in st.session_state:
    st.session_state.shared_state = SharedState()
    
# --- Lógica de Control del Hilo ---
if start_btn:
    if st.session_state.worker is None or not st.session_state.worker.is_alive():
        
        # Detener hilo anterior si existía
        if st.session_state.worker and st.session_state.worker.is_alive():
            st.session_state.worker.stop()
            st.session_state.worker.join()
        
        # Configuración para el Worker
        configs = {
            'video_source': video_source,
            'frame_skip': frame_skip,
            'ego_ratio': ego_ratio,
            'coll_thresh': coll_thresh,
            'save_imgs': save_imgs
        }
        
        # Crear y arrancar el Hilo
        st.session_state.worker = DetectionWorker(
            models=(detector, lstm_model),
            configs=configs,
            shared_state=st.session_state.shared_state
        )
        st.session_state.worker.start()
        
    st.session_state.running = True

if stop_btn:
    if st.session_state.worker and st.session_state.worker.is_alive():
        st.session_state.worker.stop()
        st.session_state.worker.join()
        st.session_state.worker = None
    st.session_state.running = False


if st.session_state.worker and st.session_state.worker.is_alive():
    
    # Este bucle se ejecuta continuamente para refrescar la interfaz
    while True:
        # Obtener datos del Worker de forma segura
        frame_rgb, is_coll, min_d, count = st.session_state.shared_state.get_state()
        
        # 1. Actualizar Stream
        if frame_rgb is not None:
             video_placeholder.image(frame_rgb, channels="RGB")
        
        # 2. Actualizar Métricas
        if is_coll: status_ph.markdown('<div class="danger-alert">¡PELIGRO!</div>', unsafe_allow_html=True)
        else: status_ph.success("Seguro")
        
        metrics_ph.markdown(f"""
        <div class="metric-card"><h3>Vehículos</h3><h1>{count}</h1></div><br>
        <div class="metric-card"><h3>Dist. Min</h3><h1 style="color:{'red' if is_coll and min_d < coll_thresh and min_d > 0 else 'white'}">{int(min_d)}px</h1></div>
        """, unsafe_allow_html=True)

        # Pausa para evitar consumir demasiados recursos del navegador
        time.sleep(0.033) # Aproximadamente 30 FPS de refresco del Streamlit UI
        