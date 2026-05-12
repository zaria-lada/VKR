import os
import sys
import pandas as pd

# 🔧 НАДЕЖНЫЙ ПОИСК КОРНЯ ПРОЕКТА
# Скрипт лежит в src/, значит корень на уровень выше (..)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.csv')

def create_metadata():
    print(f"📂 Корень проекта: {PROJECT_ROOT}")
    print(f"🔍 Ищу фото в: {RAW_DIR}")

    if not os.path.exists(RAW_DIR):
        print("❌ Папка data/raw не найдена!")
        print("💡 Создай папку data/raw и положи туда подпапки: real, stylegan2, stable_diffusion, kandinsky")
        return

    data = []
    
    # 1. Реальные фото
    real_path = os.path.join(RAW_DIR, "real")
    if os.path.exists(real_path):
        for f in os.listdir(real_path):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                data.append({'filename': f, 'generator': 'real', 'label': 0, 'path': os.path.join(real_path, f)})
            
    # 2. Синтетика
    for folder in ['stylegan2', 'stable_diffusion', 'kandinsky']:
        folder_path = os.path.join(RAW_DIR, folder)
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    data.append({'filename': f, 'generator': folder, 'label': 1, 'path': os.path.join(folder_path, f)})

    if not data:
        print("❌ Фото не найдено! Проверь путь data/raw/")
        return

    df = pd.DataFrame(data)
    df.to_csv(METADATA_PATH, index=False)
    print(f"✅ Метаданные созданы: {len(df)} записей в {METADATA_PATH}")
    print(df['label'].value_counts())

if __name__ == "__main__":
    create_metadata()