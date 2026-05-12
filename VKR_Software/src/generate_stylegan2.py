import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

def generate(output_dir='data/raw/stylegan2', num_images=100):
    """
    Генерирует реалистичные лица через локальную модель Realistic Vision.
    
    Аргументы:
        output_dir (str): Папка для сохранения изображений
        num_images (int): Количество изображений для генерации
    """
    
    # Создаём папку для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Модель, специализированная на фотореалистичных портретах
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    
    try:
        # Загружаем пайплайн
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,           # Более быстрый и безопасный формат весов
            safety_checker=None,            # Отключаем фильтр контента для исследований
            requires_safety_checker=False
        )
        
        # Определяем устройство: CUDA (GPU) или CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        # Оптимизация памяти для GPU (снижает потребление VRAM)
        if device == "cuda":
            pipe.enable_attention_slicing()
            print(f"Модель загружена на GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Модель загружена на CPU (будет медленнее)")
        
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        print("Проверь интернет-соединение и место на диске (~5 ГБ свободно)")
        return
    
    print(f"\nНАЧАЛО ГЕНЕРАЦИИ {num_images} ВЫСОКОКАЧЕСТВЕННЫХ ЛИЦ...")
    print("=" * 60)
    
    base_prompt = "portrait of a person, photorealistic, professional photography"
    
    details = [
        "studio lighting, sharp focus, 85mm lens, f/1.8, detailed skin texture, natural pores",
        "natural lighting, golden hour, soft shadows, detailed eyes, realistic hair",
        "cinematic lighting, bokeh background, professional retouching, high detail",
        "editorial portrait, clean background, perfect skin, detailed facial features",
        "fashion photography, dramatic lighting, sharp details, magazine quality",
    ]
    
    negative_prompt = (
        "blurry, low quality, pixelated, distorted, deformed, ugly, bad anatomy, "
        "disfigured, poorly drawn face, mutation, mutated, extra limb, extra hands, "
        "poorly drawn hands, missing limb, floating limbs, disconnected limbs, "
        "malformed hands, blurry, out of focus, long neck, long body, cartoon, "
        "anime, 3d, painting, drawing, illustration, sketch, watermark, text, signature"
    )
    
    # Вычисляем сколько изображений на каждый вариант деталей
    images_per_detail = num_images // len(details)
    
    count = 0  # Счётчик сгенерированных изображений

    for detail_idx, detail in enumerate(details):
        full_prompt = f"{base_prompt}, {detail}"
        print(f"\n📝 Вариант {detail_idx + 1}/{len(details)}: {detail[:50]}...")
        
        for i in range(images_per_detail):
            try:

                image = pipe(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,      
                    num_inference_steps=40,               
                    guidance_scale=7.5,                   
                    height=768,                           
                    width=768,                            
                    generator=torch.Generator(device).manual_seed(count) 
                ).images[0]
                
                # Сохранение в высоком качестве (PNG, без сжатия)
                save_path = os.path.join(output_dir, f'hq_{count:03d}.png')
                image.save(save_path)
                
                count += 1
                print(f"  ✓ Сгенерировано {count}/{num_images}", end='\r')
                
            except torch.cuda.OutOfMemoryError:

                try:
                    image = pipe(
                        prompt=full_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=30,      
                        guidance_scale=7.0,
                        height=512,             
                        width=512
                    ).images[0]
                    image.save(os.path.join(output_dir, f'hq_{count:03d}.png'))
                    count += 1
                    print(f"  ✓ {count}/{num_images} (512×512)", end='\r')
                except Exception as e2:
                    print(f"\n  Ошибка: {e2}")
                    
            except Exception as e:
                print(f"\n  Ошибка при генерации {count}: {e}")

if __name__ == '__main__':
    generate(num_images=100)
