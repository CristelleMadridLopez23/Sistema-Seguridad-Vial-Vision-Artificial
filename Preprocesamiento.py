import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
import os

# 1. Cargar el modelo pre-entrenado de YOLO (necesita estar instalado)
model = YOLO('yolov8n.pt')  # 'n' es la versión nano, rápida para el prototipo

# Define la clase 'car' (Vehículo) en el dataset COCO (clase 2)
VEHICLE_CLASS_ID = 2

def process_videos_for_tracking(video_dir, output_csv_dir):
    """Procesa videos, realiza detección y seguimiento, y guarda trayectorias."""
    os.makedirs(output_csv_dir, exist_ok=True)
    
    # 2. Iterar sobre todos los videos en el split
    for video_name in os.listdir(video_dir):
        if not video_name.endswith('.mp4'):
            continue

        video_path = os.path.join(video_dir, video_name)
        
        # DataFrame para almacenar las trayectorias de un solo video
        trajectories = []

        # 3. Ejecutar Detección y Tracking con YOLOv8
        # El parámetro 'tracker' activa un algoritmo MOT (como StrongSORT)
        # Aquí limitamos las clases solo a vehículos
        results = model.track(
            source=video_path, 
            tracker="bytetrack.yaml", # Un tracker robusto
            classes=[VEHICLE_CLASS_ID], # Solo vehículos
            stream=True, 
            persist=True
        )

        frame_count = 0
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                frame_count += 1
                continue
            
            # Extraer las coordenadas y IDs de seguimiento
            boxes = result.boxes.xywh.cpu().numpy() # x_center, y_center, width, height
            track_ids = result.boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x_center, y_center, w, h = box
                
                trajectories.append({
                    'video_id': video_name,
                    'frame': frame_count,
                    'track_id': track_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': w,
                    'height': h
                })
            
            frame_count += 1

        # 4. Guardar los resultados del video
        df = pd.DataFrame(trajectories)
        output_file = os.path.join(output_csv_dir, f"{video_name.replace('.mp4', '')}_tracks.csv")
        df.to_csv(output_file, index=False)
        print(f"Guardadas {len(df['track_id'].unique())} trayectorias en {output_file}")



# --- EJECUCIÓN DEL PROCESAMIENTO ---
# Corregir las rutas según la estructura real de tus datos
BASE_DATA_PATH = "/Users/cristellemadrid/Desktop/Sistema-vial-inteligente/BDDA_REDUCED"
OUTPUT_TRACKS_PATH = "/Users/cristellemadrid/Desktop/Sistema-vial-inteligente/processed_tracks"

# Verificar que las rutas existen antes de procesar
train_video_path = os.path.join(BASE_DATA_PATH, "train")
val_video_path = os.path.join(BASE_DATA_PATH, "val")
test_video_path = os.path.join(BASE_DATA_PATH, "test")

if not os.path.exists(train_video_path):
    print(f"Error: La ruta {train_video_path} no existe.")
    exit(1)

print("Iniciando procesamiento de TRAINING...")
process_videos_for_tracking(train_video_path, os.path.join(OUTPUT_TRACKS_PATH, "train"))

print("Iniciando procesamiento de VALIDATION...")
process_videos_for_tracking(val_video_path, os.path.join(OUTPUT_TRACKS_PATH, "val"))

print("Iniciando procesamiento de TEST...")
process_videos_for_tracking(test_video_path, os.path.join(OUTPUT_TRACKS_PATH, "test"))

print("¡Procesamiento completado!")