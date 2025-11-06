"""
Script para aumentar el dataset ANTES del entrenamiento
Genera múltiples versiones de cada imagen para tener más datos
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from tqdm import tqdm

class DatasetAugmenter:
    def __init__(self, source_dir, output_dir, target_per_class=500):
        """
        Args:
            source_dir: Carpeta con tu dataset original
            output_dir: Carpeta donde se guardará el dataset aumentado
            target_per_class: Número objetivo de imágenes por clase
        """
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.target_per_class = target_per_class
        
    def augment_image(self, image, aug_type):
        """Aplica una transformación específica a la imagen"""
        img = image.copy()
        
        if aug_type == 'rotate_90':
            img = img.rotate(90, expand=True)
        
        elif aug_type == 'rotate_180':
            img = img.rotate(180, expand=True)
        
        elif aug_type == 'rotate_270':
            img = img.rotate(270, expand=True)
        
        elif aug_type == 'rotate_random':
            angle = random.randint(-30, 30)
            img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        
        elif aug_type == 'flip_horizontal':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        elif aug_type == 'flip_vertical':
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        elif aug_type == 'brightness_up':
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.3)
        
        elif aug_type == 'brightness_down':
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(0.7)
        
        elif aug_type == 'contrast_up':
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.4)
        
        elif aug_type == 'contrast_down':
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(0.7)
        
        elif aug_type == 'saturation_up':
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.5)
        
        elif aug_type == 'saturation_down':
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(0.5)
        
        elif aug_type == 'blur':
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
        
        elif aug_type == 'sharpen':
            img = img.filter(ImageFilter.SHARPEN)
        
        elif aug_type == 'zoom_in':
            w, h = img.size
            crop_size = int(min(w, h) * 0.8)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            img = img.crop((left, top, left + crop_size, top + crop_size))
            img = img.resize((w, h), Image.LANCZOS)
        
        elif aug_type == 'zoom_out':
            w, h = img.size
            new_size = (int(w * 1.2), int(h * 1.2))
            new_img = Image.new('RGB', new_size, (255, 255, 255))
            paste_x = (new_size[0] - w) // 2
            paste_y = (new_size[1] - h) // 2
            new_img.paste(img, (paste_x, paste_y))
            img = new_img.resize((w, h), Image.LANCZOS)
        
        return img
    
    def augment_dataset(self):
        """Aumenta todo el dataset"""
        print("=" * 70)
        print("AUMENTO DE DATASET PARA ENTRENAMIENTO")
        print("=" * 70)
        
        # Tipos de augmentación disponibles
        augmentation_types = [
            'rotate_90', 'rotate_180', 'rotate_270', 'rotate_random',
            'flip_horizontal', 'flip_vertical',
            'brightness_up', 'brightness_down',
            'contrast_up', 'contrast_down',
            'saturation_up', 'saturation_down',
            'blur', 'sharpen',
            'zoom_in', 'zoom_out'
        ]
        
        # Procesar cada clase
        for class_name in ['morchella', 'no_morchella']:
            print(f"\n{'='*70}")
            print(f"Procesando clase: {class_name}")
            print(f"{'='*70}")
            
            source_class_dir = os.path.join(self.source_dir, class_name)
            output_class_dir = os.path.join(self.output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Obtener imágenes originales
            image_files = [f for f in os.listdir(source_class_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"📁 Imágenes originales: {len(image_files)}")
            print(f"🎯 Objetivo: {self.target_per_class} imágenes")
            
            # Calcular cuántas variaciones necesitamos por imagen
            variations_needed = self.target_per_class // len(image_files)
            print(f"🔄 Generando {variations_needed} variaciones por imagen")
            
            total_generated = 0
            
            # Procesar cada imagen
            for img_file in tqdm(image_files, desc=f"Aumentando {class_name}"):
                img_path = os.path.join(source_class_dir, img_file)
                img = Image.open(img_path).convert('RGB')
                
                # Guardar imagen original
                base_name = os.path.splitext(img_file)[0]
                output_path = os.path.join(output_class_dir, f"{base_name}_original.jpg")
                img.save(output_path, 'JPEG', quality=95)
                total_generated += 1
                
                # Seleccionar augmentaciones aleatorias
                selected_augs = random.sample(
                    augmentation_types, 
                    min(variations_needed - 1, len(augmentation_types))
                )
                
                # Generar variaciones
                for i, aug_type in enumerate(selected_augs):
                    try:
                        augmented_img = self.augment_image(img, aug_type)
                        output_path = os.path.join(
                            output_class_dir,
                            f"{base_name}_aug_{i}_{aug_type}.jpg"
                        )
                        augmented_img.save(output_path, 'JPEG', quality=95)
                        total_generated += 1
                    except Exception as e:
                        print(f"\n⚠️  Error en {aug_type} para {img_file}: {e}")
                
                # Si necesitamos más, combinar transformaciones
                remaining = variations_needed - len(selected_augs) - 1
                if remaining > 0:
                    for i in range(remaining):
                        try:
                            # Aplicar 2-3 transformaciones aleatorias
                            combined_augs = random.sample(augmentation_types, 
                                                         random.randint(2, 3))
                            augmented_img = img.copy()
                            for aug in combined_augs:
                                augmented_img = self.augment_image(augmented_img, aug)
                            
                            output_path = os.path.join(
                                output_class_dir,
                                f"{base_name}_combined_{i}.jpg"
                            )
                            augmented_img.save(output_path, 'JPEG', quality=95)
                            total_generated += 1
                        except Exception as e:
                            print(f"\n⚠️  Error en combinación para {img_file}: {e}")
            
            print(f"\n✅ Total generadas para {class_name}: {total_generated} imágenes")
        
        print("\n" + "=" * 70)
        print("✅ ¡AUMENTO DE DATASET COMPLETADO!")
        print("=" * 70)
        
        # Resumen final
        for class_name in ['morchella', 'no_morchella']:
            output_class_dir = os.path.join(self.output_dir, class_name)
            count = len([f for f in os.listdir(output_class_dir)
                        if f.lower().endswith('.jpg')])
            print(f"📊 {class_name}: {count} imágenes")


# EJEMPLO DE USO
if __name__ == "__main__":
    print("🚀 Iniciando aumento de dataset...")
    print()
    print("IMPORTANTE:")
    print("- Este script genera NUEVAS imágenes a partir de las originales")
    print("- Úsalo ANTES de entrenar el modelo")
    print("- El dataset original (81+81) se convertirá en uno más grande")
    print()
    
    # Configuración
    SOURCE_DIR = 'dataset'          # Tu dataset original (81+81)
    OUTPUT_DIR = 'dataset_augmented'  # Dataset aumentado
    TARGET_PER_CLASS = 500          # Objetivo: 500 imágenes por clase
    
    # Crear augmenter
    augmenter = DatasetAugmenter(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        target_per_class=TARGET_PER_CLASS
    )
    
    # Ejecutar aumento
    augmenter.augment_dataset()
    
    print("\n📋 SIGUIENTE PASO:")
    print(f"1. Verifica las imágenes en: {OUTPUT_DIR}")
    print("2. Cambia DATASET_PATH en el script de entrenamiento a 'dataset_augmented'")
    print("3. Ejecuta el entrenamiento normalmente")
    print("\n💡 CONSEJO:")
    print("Revisa visualmente algunas imágenes generadas para asegurarte")
    print("de que las transformaciones son realistas y útiles.")