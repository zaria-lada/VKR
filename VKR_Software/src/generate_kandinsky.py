import os
import torch
from PIL import Image

def generate_kandinsky(output_dir='data/raw/kandinsky', num_images=100):
    """Генерирует лица через Kandinsky 2.1."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from diffusers import KandinskyPipeline, KandinskyPriorPipeline
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # 1. Загружаем Prior (текст → эмбеддинги)
        prior = KandinskyPriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-1-prior",
            torch_dtype=dtype
        )
        prior = prior.to(device)
        
        # 2. Загружаем основную модель
        pipe = KandinskyPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-1",
            torch_dtype=dtype
        )
        pipe = pipe.to(device)
        
        if device == "cuda":
            pipe.enable_attention_slicing()
            print(f" Модель на GPU: {torch.cuda.get_device_name(0)}")
        
    except Exception as e:
        print(f" Ошибка загрузки: {e}")
        return
    
    print(f"\n Генерация {num_images} лиц...")
    
    prompts = [
        "person face photo, detailed skin texture",
        "headshot of a person, studio lighting"
    ]
    
    negative_prompt = "blurry, low quality, distorted, deformed, ugly, cartoon, anime"
    
    # Настройки
    NUM_STEPS = 40
    GUIDANCE_SCALE = 5.0
    RESOLUTION = 512
    
    images_per_prompt = num_images // len(prompts)
    count = 0
    
    for prompt in prompts:
        print(f"\n Промпт: {prompt[:50]}...")
        
        try:
            prior_out = prior(
                prompt=prompt,
                negative_prompt=negative_prompt,
                output_type='pt'  # ← Ключевой параметр!
            )
            
            # Извлекаем эмбеддинги
            img_embeds = prior_out.image_embeds
            neg_embeds = prior_out.negative_image_embeds
            
            print(f"  Эмбеддинги получены: {img_embeds.shape}")
            
            for i in range(images_per_prompt):
                try:
                    image = pipe(
                        prompt=prompt,
                        image_embeds=img_embeds,
                        negative_image_embeds=neg_embeds,
                        num_inference_steps=NUM_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        height=RESOLUTION,
                        width=RESOLUTION,
                        generator=torch.Generator(device).manual_seed(count)
                    ).images[0]
                    
                    # Сохранение
                    save_path = os.path.join(output_dir, f'kan_{count:03d}.png')
                    image.save(save_path)
                    
                    count += 1
                    print(f"   ✓ {count}/{num_images}", end='\r')
                    
                except Exception as e:
                    print(f"\n   Ошибка генерации {count}: {e}")
                    
        except Exception as e:
            print(f"\n   Ошибка промпта: {e}")
            continue
    
    print(f"\n\nГОТОВО! {count} лиц сохранено в {output_dir}")

if __name__ == '__main__':
    generate_kandinsky(num_images=100)
