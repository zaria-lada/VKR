import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.csv')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUT_CSV = os.path.join(DATA_DIR, 'processed_metadata.csv')
TARGET_SIZE = 224

# Каскад Хаара для детекции лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_dataset():
    if not os.path.exists(METADATA_PATH):
        return

    df = pd.read_csv(METADATA_PATH, encoding='utf-8-sig')
    processed_data = []
    skipped = 0
    no_face = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Загрузка через OpenCV (теперь пути на латинице, проблем не будет)
        img = cv2.imread(row['path'])
        if img is None:
            skipped += 1
            continue

        if img.shape[0] < 50 or img.shape[1] < 50:
            skipped += 1
            continue

        # Детекция лица
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) > 0:
            # Берём самое крупное лицо
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Отступ 20%
            pad = int(max(w, h) * 0.2)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
            
            # Вырезаем и ресайзим
            crop = img[y1:y2, x1:x2]
            resized = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
            
            # Сохраняем
            out_path = os.path.join(PROCESSED_DIR, row['generator'], row['filename'])
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, resized)
            
            processed_data.append({
                'filename': row['filename'],
                'generator': row['generator'],
                'label': row['label'],
                'path': out_path
            })
        else:
            no_face += 1
            skipped += 1

    # Сохраняем метаданные обработанных файлов
    pd.DataFrame(processed_data).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    

if __name__ == "__main__":
    preprocess_dataset()
