# Алгоритм обнаружения лиц в системах информационной безопасности, сгенерированных большими языковыми моделями.

## 📋 Описание

Система автоматического обнаружения лиц, сгенерированных искусственным интеллектом (StyleGAN2, Stable Diffusion, Kandinsky 2.1, Midjourney).

**Архитектура:** EfficientNet-B4 + модуль внимания CBAM  
**Метод обучения:** Transfer Learning + Fine-Tuning  
**Интерфейс:** Flask Web-приложение с визуализацией Grad-CAM

### ✨ Основные возможности

- ✅ Бинарная классификация: Real vs Fake
- ✅ Визуализация областей внимания модели (Grad-CAM)
- ✅ Web-интерфейс для загрузки изображений
- ✅ REST API для интеграции в другие системы
- ✅ Автоматическая детекция и нормализация лиц
- ✅ Поддержка GPU (CUDA 11.8+) и CPU
- ✅ Точность 100% на тестовой выборке

---

## ⚙️ Требования

### Минимальные:
- Python 3.10–3.12
- 4 GB RAM
- 2 GB свободного места

### Рекомендуемые:
- GPU NVIDIA с поддержкой CUDA 11.8+ (RTX 30/40 series)
- 8 GB RAM
- Windows 10/11, Linux или macOS

---

## Быстрый запуск

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/zaria-lada/VKR.git
cd VKR
```
### 2. Создайте виртуальное окружение
Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

###3. Установите зависимости
```bash
pip install -r requirements.txt
```

###4. Скачайте веса модели
Скачайте файл final_model.pt (~180 МБ) по ссылке:
Скачать с Яндекс.Диска https://disk.yandex.ru/d/gkGDdmPApxhVDQ
Поместите файл в папку models/final_model.pt

###5. Запустите веб-интерфейс
```bash
cd web_app
python app.py
```

Откройте браузер: http://127.0.0.1:5000

## Структура проекта
VKR/
├── src/                          # Исходный код обучения и оценки
│   ├── train.py                  # Скрипт обучения модели
│   ├── evaluate_full.py          # Полная оценка метрик
│   ├── gradcam.py                # Визуализация Grad-CAM
│   └── split_data.py             # Разбиение датасета
│
├── web_app/                      # Flask веб-приложение
│   ├── app.py                    # Основной сервер
│   ├── templates/
│   │   └── index.html            # HTML интерфейс
│   └── uploads/                  # Временные загрузки
│
├── models/                       # Обученные модели
│   └── final_model.pt            # Веса EfficientNet-B4 + CBAM
│
├── results/                      # Результаты оценки
│   ├── metrics/                  # Графики и таблицы
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── metrics_table.csv
│   │   └── ...
│   ├── predictions/
│   │   └── predictions.csv
│   └── gradcam/                  # Визуализации внимания
│
├── data/                         # Датасет (не включён в репозиторий)
│   ├── raw/                      # Исходные изображения
│   ├── processed/                # Обработанные 224×224
│   └── splits/                   # CSV с разбиением
│
├── dist/                         # Исполняемые файлы
│   └── AIFaceDetector.exe        # Windows executable
│
├── requirements.txt              # Зависимости Python
├── .gitignore                    # Игнорируемые файлы
└── README.md                     # Этот файл

##Использование
Web-интерфейс
Откройте http://127.0.0.1:5000
Перетащите изображение или нажмите для выбора файла
Нажмите кнопку "АНАЛИЗИРОВАТЬ"
Получите результат:
Вердикт: REAL / FAKE
Вероятности по классам
Тепловая карта Grad-CAM
Визуализация наложения
