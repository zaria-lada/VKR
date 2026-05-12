# split_data.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 🔧 АВТОМАТИЧЕСКИЙ ПОИСК КОРНЯ ПРОЕКТА
# Скрипт в src/, корень на уровень выше (..)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
INPUT_CSV = os.path.join(DATA_DIR, 'processed_metadata.csv')
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')

def split_data():
    print(f" Корень проекта: {PROJECT_ROOT}")
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Файл не найден: {INPUT_CSV}")
        print("💡 Сначала запусти: python preprocess_images.py")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"📊 Загружено {len(df)} обработанных изображений.")
    
    # Разбиваем: 70% Train, 15% Val, 15% Test (стратифицировано по меткам)
    # Шаг 1: отрезаем 30% (Val + Test)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['label']
    )
    # Шаг 2: делим оставшиеся 30% пополам (15% Val, 15% Test)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['label']
    )
    
    os.makedirs(SPLITS_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(SPLITS_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(SPLITS_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(SPLITS_DIR, 'test.csv'), index=False)
    
    print("\n✅ Данные успешно разбиты:")
    print(f"📂 Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"📂 Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"📂 Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    print("\n⚖️ Баланс классов в Train:")
    print(train_df['label'].value_counts())
    print(f"\n💾 CSV-файлы сохранены в: {SPLITS_DIR}")

if __name__ == "__main__":
    split_data()