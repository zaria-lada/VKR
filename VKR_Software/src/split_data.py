import os
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
INPUT_CSV = os.path.join(DATA_DIR, 'processed_metadata.csv')
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')

def split_data():
    if not os.path.exists(INPUT_CSV):
        return

    df = pd.read_csv(INPUT_CSV)
    
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['label']
    )
    
    os.makedirs(SPLITS_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(SPLITS_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(SPLITS_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(SPLITS_DIR, 'test.csv'), index=False)

if __name__ == "__main__":
    split_data()
