# web_app/app.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================================
# АРХИТЕКТУРА МОДЕЛИ
# ==========================================================

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
        self.activations = x 
        x.register_hook(self.save_gradient)
        x = self.cbam(x)
        x = self.avgpool(x)
        return self.classifier(x)
    
    def save_gradient(self, grad):
        self.gradients = grad

# ==========================================================
# ИНИЦИАЛИЗАЦИЯ
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'final_model.pt')

try:
    model = EfficientNetCBAM_GradCAM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"✅ Модель загружена на {DEVICE}")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================================
# ЛОГИКА ОБРАБОТКИ
# ==========================================================

def array_to_base64(img_array):
    """Конвертирует numpy array в base64"""
    try:
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img = Image.fromarray(img_array, 'RGB')
        else:
            img = Image.fromarray(img_array)
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"❌ Ошибка конвертации: {e}")
        return ""

def predict_with_gradcam(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(max(w, h) * 0.2)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_rgb.shape[1], x + w + pad), min(img_rgb.shape[0], y + h + pad)
        face_crop = img_rgb[y1:y2, x1:x2]
    else:
        face_crop = cv2.resize(img_rgb, (224, 224))
        
    face_resized = cv2.resize(face_crop, (224, 224))
    img_tensor = transform(face_resized).unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad_(True)
    
    model.zero_grad()
    output = model(img_tensor)
    prob_fake = output[0, 0].item()
    
    # 🔥 Показываем ТОЛЬКО признаки FAKE (артефакты)
    loss = output[0, 0]
    loss.backward()
    
    if model.gradients is None or model.activations is None:
        return None

    weights = torch.mean(model.gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * model.activations, dim=1, keepdim=True)
    
    # ReLU оставляет только то, что модель считает "подделкой"
    cam = torch.relu(cam) 
    
    cam = nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    heatmap = cam.squeeze().detach().cpu().numpy()
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
    
    heatmap_uint8 = np.uint8(heatmap * 255)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(face_bgr, 0.6, heatmap_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    
    is_artifact_detected = bool(heatmap.max() > 0.1)
    is_fake = bool(prob_fake >= 0.3)
    
    return {
        "real_prob": float(1 - prob_fake),
        "fake_prob": float(prob_fake),
        "is_fake": is_fake,
        "face_img": face_resized,
        "heatmap": heatmap_rgb,
        "overlay": overlay_rgb,
        "artifact_detected": is_artifact_detected
    }

# ==========================================================
# WEB ROUTES
# ==========================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        result = predict_with_gradcam(filepath)
    except Exception as e:
        print(f"Ошибка обработки: {e}")
        return jsonify({"error": str(e)}), 500
    
    if result:
        face_b64 = array_to_base64(result['face_img'])
        heatmap_b64 = array_to_base64(result['heatmap'])
        overlay_b64 = array_to_base64(result['overlay'])
        
        # 🔥 ИСПРАВЛЕНИЕ: добавлен префикс data: для корректного отображения в браузере
        label = "FAKE (Синтетика)" if result['is_fake'] else "REAL (Настоящее)"
        
        return jsonify({
            "success": True,
            "real_prob": f"{result['real_prob']*100:.1f}%",
            "fake_prob": f"{result['fake_prob']*100:.1f}%",
            "label": label,
            "artifact_detected": result['artifact_detected'],
            "face_image": f"data:image/jpeg;base64,{face_b64}",
            "heatmap": f"data:image/jpeg;base64,{heatmap_b64}",
            "overlay": f"data:image/jpeg;base64,{overlay_b64}"
        })
    return jsonify({"error": "Failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)