# src/evaluate_full.py
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

#  Автоматический поиск корня проекта
def find_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(current, 'data')) and os.path.isdir(os.path.join(current, 'src')):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = find_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Создаём структуру папок
os.makedirs(os.path.join(RESULTS_DIR, 'metrics'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'predictions'), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.conv(torch.cat([torch.mean(x, dim=1, keepdim=True), 
                                   torch.max(x, dim=1, keepdim=True)[0]], dim=1))
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x) * x
        return self.sa(x) * x

class EfficientNetCBAM(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        backbone = models.efficientnet_b4(weights=None)
        self.features = backbone.features
        self.cbam = CBAM(channels=1792)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(1792, num_classes), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        return self.classifier(x)

class FaceDataset(Dataset):
    def __init__(self, csv_path, img_size=224):
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = plt.imread(row['path'])
        if img.shape[2] == 4: img = img[:,:,:3]
        return self.transform(img), torch.tensor(row['label'], dtype=torch.float32), row['filename']

def save_table_as_image(df, title, filename, col_colors=None):
    fig, ax = plt.subplots(figsize=(10, len(df)*0.5 + 1.5))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=col_colors if col_colors else ['#2c3e50']*len(df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for i, cell in enumerate(table._cells.values()):
        if i < len(df.columns):
            cell.set_text_props(fontweight='bold', color='white')
        cell.set_facecolor('#ecf0f1' if (i // len(df.columns)) % 2 == 0 else '#ffffff')
        
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'metrics', filename), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_and_save():
    
    # 1. Загрузка модели
    print("\n Загрузка модели...")
    model_path = os.path.join(MODELS_DIR, 'final_model.pt')
    if not os.path.exists(model_path):
        print(f" Модель не найдена: {model_path}")
        return
    
    model = EfficientNetCBAM(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f" Модель загружена")
    
    # 2. Загрузка данных
    test_csv = os.path.join(SPLITS_DIR, 'test.csv')
    if not os.path.exists(test_csv):
        return
        
    test_ds = FaceDataset(test_csv)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    
    # 3. Инференс
    all_preds, all_probs, all_labels, all_filenames = [], [], [], []
    
    with torch.no_grad():
        for imgs, labels, filenames in test_loader:
            imgs = imgs.to(device)
            probs = model(imgs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_filenames.extend(filenames)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 4. Расчёт метрик
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = auc(roc_curve(all_labels, all_probs)[0], roc_curve(all_labels, all_probs)[1])
    
    metrics_df = pd.DataFrame({
        'Метрика': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Значение': [acc, prec, rec, f1, roc_auc],
        'Процент': [f"{acc*100:.2f}%", f"{prec*100:.2f}%", f"{rec*100:.2f}%", f"{f1*100:.2f}%", f"{roc_auc*100:.2f}%"]
    })
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'metrics', 'metrics_table.csv'), index=False, encoding='utf-8-sig')
    print(metrics_df.to_string(index=False))
    
    # 5. Визуализации
    # 5.1 Confusion Matrix (Heatmap)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'metrics', 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # 5.2 Confusion Matrix (Table via matplotlib)
    cm_df = pd.DataFrame(cm, columns=['Predicted Real', 'Predicted Fake'], index=['True Real', 'True Fake'])
    save_table_as_image(cm_df, 'Confusion Matrix Table', 'confusion_matrix_table.png', col_colors=['#3498db', '#e74c3c'])
    
    # 5.3 ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12); plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'metrics', 'roc_curve.png'), dpi=300)
    plt.close()
    
    # 5.4 Precision-Recall Curve
    prec_vals, rec_vals, _ = precision_recall_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(rec_vals, prec_vals, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall', fontsize=12); plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'metrics', 'precision_recall.png'), dpi=300)
    plt.close()
    
    # 5.5 Class Distribution
    counts = [np.sum(all_labels == 0), np.sum(all_labels == 1)]
    plt.figure(figsize=(8, 6))
    plt.bar(['Real', 'Fake'], counts, color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1.5)
    plt.title('Class Distribution in Test Set', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=12)
    for i, v in enumerate(counts):
        plt.text(i, v + 0.5, str(v), ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'metrics', 'class_distribution.png'), dpi=300)
    plt.close()
    
    # 5.6 Metrics Table (via matplotlib)
    save_table_as_image(metrics_df[['Метрика', 'Процент']], 'Model Performance Metrics', 'metrics_table_visual.png', col_colors=['#2c3e50', '#27ae60'])
    
    # 5.7 Classification Report Table
    report_dict = classification_report(all_labels, all_preds, target_names=['Real', 'Fake'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(4)
    save_table_as_image(report_df, 'Classification Report', 'classification_report_table.png', col_colors=['#2c3e50']*4)
    
    # 6. Сохранение предсказаний
    predictions_df = pd.DataFrame({
        'filename': all_filenames,
        'true_label': all_labels,
        'predicted_label': all_preds,
        'probability_fake': all_probs,
        'probability_real': 1 - all_probs,
        'correct': all_preds == all_labels
    })
    predictions_df.to_csv(os.path.join(RESULTS_DIR, 'predictions', 'predictions.csv'), index=False, encoding='utf-8-sig')
    
    # 7. Текстовый отчёт
    with open(os.path.join(RESULTS_DIR, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write("ОТЧЁТ ПО ОЦЕНКЕ МОДЕЛИ DETECTION AI-FACES\n")
        f.write("="*60 + "\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Модель: EfficientNet-B4 + CBAM\n")
        f.write(f"Тестовая выборка: {len(test_ds)} изображений\n\n")
        f.write("МЕТРИКИ:\n")
        f.write(metrics_df.to_string(index=False) + "\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(classification_report(all_labels, all_preds, target_names=['Real (0)', 'Fake (1)']))

if __name__ == "__main__":
    evaluate_and_save()
