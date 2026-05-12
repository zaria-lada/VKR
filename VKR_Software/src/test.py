# src/test_my_face.py
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt

# 🔧 Пути
def find_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(current, 'data')) and os.path.isdir(os.path.join(current, 'src')):
            return current
        parent = os.path.dirname(current)
        if parent == current: break
        current = parent
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = find_project_root()
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Устройство: {device}")

# --- МОДЕЛЬ (должна точно совпадать с обучением) ---
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

# --- ДЕТЕКТОР ЛИЦ ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_face(image_path, model):
    # 1. Загрузка
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"❌ Не удалось открыть: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    # 2. Поиск лица
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(max(w, h) * 0.2)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_rgb.shape[1], x + w + pad), min(img_rgb.shape[0], y + h + pad)
        face_crop = img_rgb[y1:y2, x1:x2]
        print("✅ Лицо обнаружено и обрезано.")
    else:
        print("⚠️ Лицо не найдено автоматически. Использую полное изображение (точность может снизиться).")
        face_crop = img_rgb

    # 3. Подготовка для модели
    face_resized = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(face_resized).unsqueeze(0).to(device)

    # 4. Инференс
    model.eval()
    with torch.no_grad():
        prob_fake = model(img_tensor).item()
    
    prob_real = 1 - prob_fake
    prediction = "FAKE (Синтетическое)" if prob_fake >= 0.5 else "REAL (Реальное)"

    # 5. Визуализация
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Оригинал")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(face_crop)
    plt.title("Обрезанное лицо")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(face_resized)
    plt.title(f"{prediction}\nFAKE: {prob_fake:.1%} | REAL: {prob_real:.1%}")
    plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "my_face_prediction.png")
    plt.savefig(save_path, dpi=150)
    print(f"🖼️ Результат сохранён: {save_path}")

    print("\n📊 ИТОГОВЫЙ ПРОГНОЗ:")
    print(f"   Класс: {prediction}")
    print(f"   Вероятность FAKE: {prob_fake:.4f} ({prob_fake*100:.2f}%)")
    print(f"   Вероятность REAL: {prob_real:.4f} ({prob_real*100:.2f}%)")

def main():
    print("🧪 Тестирование модели на твоём фото")
    image_path = input("📁 Введи путь к фото (например, C:\\Users\\retro\\Desktop\\my_photo.jpg): ").strip().strip('"')
    
    if not os.path.exists(image_path):
        print("❌ Файл не найден. Проверь путь и попробуй снова.")
        return

    print("📥 Загрузка модели...")
    model = EfficientNetCBAM(num_classes=1).to(device)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'final_model.pt'), map_location=device, weights_only=True))
    print("✅ Модель готова\n")

    try:
        predict_face(image_path, model)
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()