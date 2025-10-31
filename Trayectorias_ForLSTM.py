import pandas as pd
import numpy as np
import os
from typing import Tuple

# --- CONFIGURACIÓN ---
# RUTA BASE DONDE ESTÁN TUS CSV PROCESADOS POR YOLO
# Asegúrate que esta ruta exista y contenga las carpetas 'train', 'val', 'test'
BASE_PROCESSED_TRACKS = "/Users/cristellemadrid/Desktop/Sistema-vial-inteligente/processed_tracks" 

# Parámetros de la ventana temporal
FPS = 30 # Asumiendo que BDD-Attention usa 30 FPS
T_HIST = 20  # Historia (Input): 20 frames (~0.67 segundos)
T_PRED = 10  # Predicción (Output): 10 frames (~0.33 segundos)
MIN_TRACK_LENGTH = T_HIST + T_PRED + 1 # Descartar trayectorias muy cortas

# Dimensiones de normalización (Común en BDD)
WIDTH, HEIGHT = 1280, 720


def load_and_sequence_trajectories(split_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Carga todos los CSV de un split, agrupa por track_id y crea las ventanas temporales."""
    
    all_trajectories = []
    
    # 1. Cargar y concatenar todos los CSV del split
    print(f"Cargando archivos CSV desde: {split_dir}")
    for csv_file in os.listdir(split_dir):
        if csv_file.endswith('_tracks.csv'):
            path = os.path.join(split_dir, csv_file)
            df = pd.read_csv(path)
            all_trajectories.append(df)
            
    if not all_trajectories:
        print("Error: No se encontraron archivos CSV. Verifica la ruta.")
        return np.array([]), np.array([])

    df_full = pd.concat(all_trajectories, ignore_index=True)
    
    # Solo necesitamos las coordenadas (x, y)
    df_coords = df_full[['track_id', 'frame', 'x_center', 'y_center']].copy()
    
    # 2. Normalizar coordenadas
    df_coords['x_center_norm'] = df_coords['x_center'] / WIDTH
    df_coords['y_center_norm'] = df_coords['y_center'] / HEIGHT
    
    X_sequences, Y_sequences = [], []

    # 3. Iterar por cada vehículo individual (track_id)
    unique_tracks = df_coords['track_id'].nunique()
    print(f"Total de trayectorias únicas detectadas: {unique_tracks}")
    
    for track_id, track_data in df_coords.groupby('track_id'):
        
        track_data = track_data.sort_values('frame').reset_index(drop=True)
        coords = track_data[['x_center_norm', 'y_center_norm']].values
        
        if len(coords) < MIN_TRACK_LENGTH:
            continue # Descartar trayectorias muy cortas
            
        # 4. Crear ventanas deslizantes (Sliding Windows)
        # El bucle itera a través de la trayectoria para crear múltiples ejemplos (pares Input/Output)
        for i in range(len(coords) - T_HIST - T_PRED + 1):
            
            # Input: T_HIST frames de historia (Posiciones pasadas)
            history = coords[i : i + T_HIST]
            
            # Output: T_PRED frames de futuro (Posiciones futuras, el ground truth)
            future = coords[i + T_HIST : i + T_HIST + T_PRED]
            
            X_sequences.append(history)
            Y_sequences.append(future)

    return np.array(X_sequences), np.array(Y_sequences)


def process_split(split_name: str):
    """Función auxiliar para procesar, guardar y reportar resultados."""
    output_file = f"{split_name}_lstm_sequences.npz"
    split_dir = os.path.join(BASE_PROCESSED_TRACKS, split_name)
    
    if not os.path.exists(split_dir):
        print(f"\n🚨 ERROR: El directorio '{split_dir}' no existe. Verifique BASE_PROCESSED_TRACKS.")
        return

    print(f"\n--- INICIANDO PROCESAMIENTO DE {split_name.upper()} ---")
    X, Y = load_and_sequence_trajectories(split_dir)
    
    if X.size > 0:
        np.savez_compressed(output_file, X=X, Y=Y)
        print(f"🎉 Datos de {split_name} guardados en {output_file}")
        print(f"Shape de Input (X): {X.shape}")
        print(f"Shape de Output (Y): {Y.shape}")
    else:
        print(f"⚠️ No se generaron secuencias para {split_name}.")

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    
    # 1. Procesar TRAIN
    process_split("train") 
    
    # 2. Procesar VALIDATION
    process_split("val")
    
    # 3. Procesar TEST
    process_split("test")

    print("\n✅ Proceso de creación de secuencias completado. Los archivos .npz están listos para el entrenamiento LSTM.")