import os
import torch
from PIL import Image, ImageEnhance
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def enhance_image(image, sharpness=1.15, contrast=1.08):
    """
    Лёгкая пост-обработка для улучшения детализации.
    """
    # Увеличение резкости
    if sharpness > 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
    # Лёгкое повышение контраста
    if contrast > 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    return image

def generate_sd_local(output_dir='data/raw/stable_diffusion', num_images=100):
    """
    Генерирует высококачественные лица через локальную модель.
    """
    os.makedirs(output_dir, exist_ok=True)

    
    # СПЕЦИАЛИЗИРОВАННАЯ МОДЕЛЬ ДЛЯ ФОТОРЕАЛИСТИЧНЫХ ПОРТРЕТОВ
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    
    try:
        # Загружаем пайплайн
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,           # Быстрее и безопаснее
            safety_checker=None,            # Отключаем фильтр для исследований
            requires_safety_checker=False
        )
        
        # 🔥 ЗАМЕНЯЕМ SCHEDULER НА БЫСТРЫЙ И КАЧЕСТВЕННЫЙ
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            final_sigmas_type="sigma_min"
        )
        
        # Определяем устройство
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        # Оптимизация памяти для GPU
        if device == "cuda":
            pipe.enable_attention_slicing()
            print(f" Модель загружена на GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(" Модель загружена на CPU (будет медленнее)")
        
    except Exception as e:
        print(f" Ошибка загрузки модели: {e}")
        return
    
    print(f"\n ГЕНЕРАЦИЯ {num_images} ВЫСОКОКАЧЕСТВЕННЫХ ЛИЦ...")
    print("=" * 60)
    
    # БАЗОВЫЙ ПРОМПТ
    base_prompt = "portrait of a person, photorealistic, professional photography"
    
    # ВАРИАЦИИ ДЕТАЛЕЙ
    details = [
        "studio lighting, sharp focus, 85mm lens, f/1.8, detailed skin texture, natural pores, realistic eyes with catchlight",
        "natural lighting, golden hour, soft shadows, detailed eyes, realistic hair strands, subsurface scattering skin",
        "cinematic lighting, bokeh background, professional retouching, high detail, magazine quality, perfect composition",
        "editorial portrait, clean background, flawless skin, detailed facial features, natural expression, high resolution",
        "fashion photography, dramatic lighting, sharp details, vogue style, professional color grading, 8k",
    ]
    
    # НЕГАТИВНЫЙ ПРОМПТ (что НЕ должно быть)
    negative_prompt = (
        "blurry, low quality, pixelated, distorted, deformed, ugly, bad anatomy, "
        "disfigured, poorly drawn face, mutation, mutated, extra limb, extra hands, "
        "poorly drawn hands, missing limb, floating limbs, disconnected limbs, "
        "malformed hands, out of focus, long neck, long body, cartoon, anime, "
        "3d, painting, drawing, illustration, sketch, watermark, text, signature, "
        "noise, grain, compression artifacts, oversaturated, undersaturated, "
        "plastic skin, doll-like, wax figure, mannequin"
    )
    
    NUM_STEPS = 50 
    GUIDANCE_SCALE = 7.5
    RESOLUTION = 768
    
    images_per_detail = num_images // len(details)
    count = 0
    
    for detail_idx, detail in enumerate(details):
        full_prompt = f"{base_prompt}, {detail}"
        print(f"\n📝 Вариант {detail_idx + 1}/{len(details)}")
        
        for i in range(images_per_detail):
            try:
                #  ГЕНЕРАЦИЯ С МАКСИМАЛЬНЫМИ НАСТРОЙКАМИ
                image = pipe(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,     
                    num_inference_steps=NUM_STEPS,    
                    guidance_scale=GUIDANCE_SCALE,     
                    height=RESOLUTION,             
                    width=RESOLUTION,
                    generator=torch.Generator(device).manual_seed(count)  # ← Воспроизводимость
                ).images[0]
                
                # ЛЁГКАЯ ПОСТ-ОБРАБОТКА
                image = enhance_image(image, sharpness=1.15, contrast=1.08)
                
                # Ресайз до 512×512 для единообразия датасета
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Сохранение в высоком качестве
                save_path = os.path.join(output_dir, f'sd_hq_{count:03d}.png')
                image.save(save_path)
                
                count += 1
                print(f"   {count}/{num_images}", end='\r')
                
            except torch.cuda.OutOfMemoryError:
                # 🔥 АВТО-ФАЛЛБЭК при нехватке памяти
                try:
                    image = pipe(
                        prompt=full_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=35,
                        guidance_scale=7.0,
                        height=512,
                        width=512
                    ).images[0]
                    image = enhance_image(image, sharpness=1.1, contrast=1.05)
                    image.save(os.path.join(output_dir, f'sd_hq_{count:03d}.png'))
                    count += 1
                    print(f"  ✓ {count}/{num_images} (512×512)", end='\r')
                except Exception as e2:
                    print(f"\n  Ошибка: {e2}")
                    
            except Exception as e:
                print(f"\n  Ошибка {count}: {e}")
    

if __name__ == '__main__':
    generate_sd_local(num_images=100)
