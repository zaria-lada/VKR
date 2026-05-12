# src/gradcam.py
import os
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'gradcam')
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- CBAM (должен совпадать с обучением) ---
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

# --- МОДЕЛЬ С ХУКАМИ ДЛЯ GRAD-CAM ---
class EfficientNetCBAM_GradCAM(nn.Module):
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
        self.gradients = None
        self.activations = None
        
    def forward(self, x):
        x = self.features(x)
        self.activations = x.clone().detach()  # Сохраняем для Grad-CAM
        x.register_hook(self.save_gradient)     # Хук для градиентов
        x = self.cbam(x)
        x = self.avgpool(x)
        return self.classifier(x)
    
    def save_gradient(self, grad):
        self.gradients = grad

    def get_cam_weights(self):
        """Вычисляет веса для Grad-CAM из градиентов"""
        return torch.mean(self.gradients, dim=[2, 3], keepdim=True)

# --- GRAD-CAM ФУНКЦИЯ ---
def apply_gradcam(model, img_tensor, target_class=1):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Прямой проход
    output = model(img_tensor)
    
    # Обратный проход для целевого класса
    model.zero_grad()
    if target_class == 1:
        loss = output[0, 0]  # Для класса "Fake"
    else:
        loss = 1 - output[0, 0]  # Для класса "Real"
    loss.backward()
    
    # Получаем веса и активации
    weights = model.get_cam_weights()  # [1, 1792, 1, 1]
    activations = model.activations    # [1, 1792, 7, 7]
    
    # Вычисляем карту внимания
    cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, 7, 7]
    cam = torch.relu(cam)  # ReLU для позитивных вкладов
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)  # Нормализация [0, 1]
    
    # Апсемплинг до размера изображения
    cam = nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    
    return cam, output.item()

# --- ВИЗУАЛИЗАЦИЯ ---
def visualize_gradcam(image_path, model):
    # Загрузка и препроцессинг
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_rgb)
    
    # Grad-CAM
    heatmap, prob = apply_gradcam(model, img_tensor, target_class=1)
    
    # Наложение тепловой карты
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Ресайз оригинала для совмещения
    img_resized = cv2.resize(img_rgb, (224, 224))
    overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
    
    return img_resized, heatmap, overlay, prob

# --- ГЛАВНЫЙ СКРИПТ ---
def main():
    print("🎨 Grad-CAM Визуализация")
    
    # Загрузка модели
    model = EfficientNetCBAM_GradCAM(num_classes=1).to(device)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'final_model.pt'), 
                                     map_location=device, weights_only=True))
    print("✅ Модель загружена")
    
    # Примеры для визуализации (возьми из test.csv)
    import pandas as pd
    test_csv = os.path.join(PROJECT_ROOT, 'data', 'splits', 'test.csv')
    test_df = pd.read_csv(test_csv, encoding='utf-8-sig')
    
    # Берём по 2 примера каждого класса
    real_samples = test_df[test_df['label'] == 0].head(2)['path'].tolist()
    fake_samples = test_df[test_df['label'] == 1].head(2)['path'].tolist()
    
    samples = [(p, 'REAL') for p in real_samples] + [(p, 'FAKE') for p in fake_samples]
    
    print(f"🖼️ Визуализация {len(samples)} изображений...")
    
    for i, (img_path, true_label) in enumerate(samples):
        try:
            orig, heatmap, overlay, prob = visualize_gradcam(img_path, model)
            pred_label = 'FAKE' if prob >= 0.5 else 'REAL'
            
            # Сохранение
            filename = os.path.basename(img_path)
            save_path = os.path.join(RESULTS_DIR, f'gradcam_{i}_{true_label}_{filename}')
            
            # Компоновка: оригинал | карта | результат
            combined = np.hstack([orig, heatmap, overlay])
            cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            
            print(f"  ✓ {true_label} → {pred_label} (prob={prob:.3f}) | {save_path}")
            
        except Exception as e:
            print(f"  ⚠️ Ошибка с {img_path}: {e}")
    
    print(f"\n✅ Grad-CAM визуализации сохранены в: {RESULTS_DIR}")
    print("💡 Открой эти изображения для диплома — они покажут, что модель смотрит на артефакты!")

if __name__ == "__main__":
    main()