import os
import shutil
import random
from typing import Dict

# --- CONFIGURACIÓN DE RUTAS Y TAMAÑOS ---

# Ruta base donde se encuentra tu dataset completo BDDA
BASE_SOURCE_PATH = "/Users/cristellemadrid/Desktop/Sistema-vial-inteligente/BDDA"

# Ruta donde se creará el dataset reducido
BASE_DEST_PATH = "/Users/cristellemadrid/Desktop/Sistema-vial-inteligente/BDDA_REDUCED"

# Definición de los tamaños deseados para cada split
SUBSET_SIZES: Dict[str, int] = {
    "train": 150,
    "val": 30,
    "test": 20
}

# La estructura de carpetas de origen (ajusta si es necesario)
SOURCE_PATHS: Dict[str, str] = {
    # Los videos de 'training' y 'validation' están en carpetas separadas
    "train": os.path.join(BASE_SOURCE_PATH, "training", "camera_videos"),
    "val": os.path.join(BASE_SOURCE_PATH, "validation", "camera_videos"),
    # Asumimos que los videos de 'test' están en una carpeta llamada 'test/camera_videos'
    "test": os.path.join(BASE_SOURCE_PATH, "test", "camera_videos") 
    # NOTA: Ajusta la ruta de 'test' si no existe la subcarpeta 'camera_videos'
}


def create_reduced_subset(split_name: str):
    """Crea un subconjunto de videos para un split dado y lo copia a la nueva carpeta."""
    
    source_dir = SOURCE_PATHS.get(split_name)
    target_size = SUBSET_SIZES.get(split_name)
    
    # Crea la ruta de destino: BDDA_REDUCED/train, BDDA_REDUCED/val, etc.
    dest_dir = os.path.join(BASE_DEST_PATH, split_name)
    
    if not source_dir or not os.path.exists(source_dir):
        print(f"⚠️ ERROR: El directorio de origen para '{split_name}' no se encontró: {source_dir}")
        return

    # Crear directorio de destino si no existe
    os.makedirs(dest_dir, exist_ok=True)
    
    # Listar todos los archivos .mp4 en el directorio de origen
    all_videos = [f for f in os.listdir(source_dir) if f.endswith('.mp4')]
    
    if len(all_videos) < target_size:
        print(f"🚨 ALERTA: Solo hay {len(all_videos)} videos en {source_dir}. Se copiarán todos.")
        videos_to_copy = all_videos
    else:
        # Seleccionar videos aleatoriamente sin reemplazo
        videos_to_copy = random.sample(all_videos, target_size)

    # Copiar los archivos seleccionados
    copied_count = 0
    for video_name in videos_to_copy:
        src_file = os.path.join(source_dir, video_name)
        dest_file = os.path.join(dest_dir, video_name)
        shutil.copy2(src_file, dest_file) # copy2 copia metadatos de archivo
        copied_count += 1
        
    print(f"✅ Split '{split_name}' completado. Copiados {copied_count} videos a {dest_dir}")


# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    print(f"Iniciando la reducción del dataset BDD-Attention. Destino: {BASE_DEST_PATH}\n")
    
    # Semilla (seed) para asegurar que la selección aleatoria sea reproducible
    random.seed(42) 
    
    # Ejecutar la reducción para cada split
    for split in ["train", "val", "test"]:
        create_reduced_subset(split)
        
    print("\n🎉 Reducción de dataset completada. Puedes comenzar a trabajar con BDDA_REDUCED.")