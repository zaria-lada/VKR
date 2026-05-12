# src/train.py
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# 🔧 УНИВЕРСАЛЬНЫЙ ПОИСК КОРНЯ ПРОЕКТА
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

print(f"📍 Корень проекта: {PROJECT_ROOT}")
print(f"📍 Данные: {DATA_DIR}")

if not os.path.exists(SPLITS_DIR):
    print(f"❌ Папка splits не найдена: {SPLITS_DIR}")
    print("💡 Запусти сначала: python src/split_data.py")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Устройство: {device}")
if device.type == 'cuda':
    print(f"✅ Видеокарта: {torch.cuda.get_device_name(0)}")

# --- CBAM MODULES ---
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

# --- ЧИСТАЯ АРХИТЕКТУРА: EfficientNet + CBAM ---
class EfficientNetCBAM(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        # 1. Загружаем EfficientNet-B4
        backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # 2. Берём только convolutional features (без классификатора)
        self.features = backbone.features  # выход: [B, 1792, 7, 7] для 224x224 входа
        
        # 3. Добавляем CBAM
        self.cbam = CBAM(channels=1792)
        
        # 4. Adaptive pooling + классификатор
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1792, num_classes),
            nn.Sigmoid()
        )
        
        # Замораживаем features для начала
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)      # [B, 1792, 7, 7]
        x = self.cbam(x)           # [B, 1792, 7, 7]
        x = self.avgpool(x)        # [B, 1792, 1, 1]
        x = self.classifier(x)     # [B, 1]
        return x

# --- DATASET ---
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
        return self.transform(img), torch.tensor(row['label'], dtype=torch.float32)

# --- TRAIN/VAL ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss, acc, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        out = model(imgs)
        l = criterion(out, labels)
        l.backward(); optimizer.step()
        loss += l.item()
        acc += ((out >= 0.5).float() == labels).sum().item()
        total += labels.size(0)
    return loss/len(loader), acc/total

def validate(model, loader, criterion, device):
    model.eval()
    loss, acc, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val"):
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            out = model(imgs)
            loss += criterion(out, labels).item()
            acc += ((out >= 0.5).float() == labels).sum().item()
            total += labels.size(0)
    return loss/len(loader), acc/total

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    train_ds = FaceDataset(os.path.join(SPLITS_DIR, 'train.csv'))
    val_ds = FaceDataset(os.path.join(SPLITS_DIR, 'val.csv'))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    print(f"✅ Загружено: Train={len(train_ds)}, Val={len(val_ds)}")

    # Создаём модель с правильной архитектурой
    model = EfficientNetCBAM(num_classes=1, pretrained=True).to(device)
    criterion = nn.BCELoss()
    
    print("\n🔹 ЭТАП 1: Training Head + CBAM (backbone frozen)...")
    # Обучаем только CBAM и classifier
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    best_val = float('inf')
    for e in range(15):
        tl, ta = train_epoch(model, train_loader, criterion, opt, device)
        vl, va = validate(model, val_loader, criterion, device)
        print(f"Epoch {e+1}/15 | T:{tl:.4f}({ta*100:.1f}%) | V:{vl:.4f}({va*100:.1f}%)")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pt'))

    print("\n🔓 ЭТАП 2: Fine-Tuning (unfreeze backbone)...")
    # Размораживаем backbone для дообучения
    for param in model.features.parameters():
        param.requires_grad = True
    opt = optim.Adam(model.parameters(), lr=1e-5)
    
    for e in range(15, 45):
        tl, ta = train_epoch(model, train_loader, criterion, opt, device)
        vl, va = validate(model, val_loader, criterion, device)
        print(f"Epoch {e+1}/45 | T:{tl:.4f}({ta*100:.1f}%) | V:{vl:.4f}({va*100:.1f}%)")
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pt'))

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'final_model.pt'))
    print(f"\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО. Модель: {os.path.join(MODELS_DIR, 'final_model.pt')}")

if __name__ == "__main__":
    main()