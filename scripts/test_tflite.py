import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob

class MorchellaClassifier:
    def __init__(self, model_path=None):
        """
        Inicializa el clasificador con el modelo TFLite
        Si model_path es None, busca automáticamente el modelo
        """
        # Si no se proporciona ruta, buscar modelos disponibles
        if model_path is None:
            model_path = self._find_tflite_model()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ No se encontró el modelo: {model_path}\n"
                f"📁 Archivos .tflite disponibles:\n"
                f"{self._list_available_models()}"
            )
        
        print(f"📦 Cargando modelo: {model_path}")
        
        # Cargar el modelo TFLite
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Obtener detalles de entrada y salida
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Obtener el tamaño de entrada esperado
        self.input_shape = self.input_details[0]['shape']
        self.img_size = self.input_shape[1]
        
        print(f"✅ Modelo cargado correctamente")
        print(f"📐 Tamaño de entrada: {self.img_size}x{self.img_size}")
        print(f"📊 Tipo de entrada: {self.input_details[0]['dtype']}")
        print(f"📊 Tipo de salida: {self.output_details[0]['dtype']}")
    
    def _find_tflite_model(self):
        """Busca automáticamente un modelo .tflite en el directorio actual"""
        # Buscar en directorio actual y padre
        search_paths = [
            '.',
            '..',
            './models',
            '../models'
        ]
        
        for path in search_paths:
            tflite_files = glob.glob(os.path.join(path, '*.tflite'))
            if tflite_files:
                # Priorizar modelos con nombres específicos
                priority_names = [
                    'morchella_classifier_small.tflite',
                    'morchella_classifier.tflite',
                    'best_morchella'
                ]
                
                for priority_name in priority_names:
                    matches = [f for f in tflite_files if priority_name in f]
                    if matches:
                        return matches[0]
                
                # Si no hay nombres prioritarios, usar el primero encontrado
                return tflite_files[0]
        
        raise FileNotFoundError(
            "❌ No se encontró ningún modelo .tflite\n"
            "💡 Asegúrate de haber entrenado el modelo primero"
        )
    
    def _list_available_models(self):
        """Lista todos los modelos .tflite disponibles"""
        models = glob.glob('*.tflite') + glob.glob('**/*.tflite', recursive=True)
        if models:
            return "\n".join([f"  - {m}" for m in models])
        return "  (ninguno encontrado)"
    
    def preprocess_image(self, image_path):
        """Preprocesa la imagen para el modelo"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ No se encontró la imagen: {image_path}")
        
        # Cargar imagen
        img = Image.open(image_path).convert('RGB')
        
        # Redimensionar
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        # Convertir a array numpy
        img_array = np.array(img, dtype=np.float32)
        
        # Normalizar (0-255 a 0-1)
        img_array = img_array / 255.0
        
        # Añadir dimensión de batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path, threshold=0.7):
        """
        Realiza la predicción sobre una imagen
        
        Args:
            image_path: Ruta a la imagen
            threshold: Umbral de confianza (default: 0.7 = 70%)
        
        Returns:
            dict con resultados de la predicción
        """
        # Preprocesar imagen
        img_array = self.preprocess_image(image_path)
        
        # Establecer tensor de entrada
        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        
        # Ejecutar inferencia
        self.interpreter.invoke()
        
        # Obtener resultado
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        raw_output = float(output[0][0])
        
        # IMPORTANTE: El modelo devuelve:
        # - 0.0 = Morchella (clase 0)
        # - 1.0 = No Morchella (clase 1)
        # Por eso invertimos la interpretación
        
        # Determinar la clase (LÓGICA INVERTIDA)
        if raw_output <= (1 - threshold):  # Si es bajo (cercano a 0)
            is_morchella = True
            label = "Morchella"
            certainty = (1 - raw_output) * 100  # Confianza = qué tan cerca de 0
        elif raw_output >= threshold:  # Si es alto (cercano a 1)
            is_morchella = False
            label = "No Morchella"
            certainty = raw_output * 100  # Confianza = qué tan cerca de 1
        else:  # Entre 0.3 y 0.7
            is_morchella = None
            label = "Incierto"
            certainty = max(raw_output, 1 - raw_output) * 100
        
        return {
            'is_morchella': is_morchella,
            'label': label,
            'confidence_percentage': round(certainty, 2),
            'raw_score': round(raw_output, 4),
            'threshold': threshold,
            'image_path': image_path
        }
    
    def predict_batch(self, image_folder, threshold=0.7):
        """Predice múltiples imágenes en una carpeta"""
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"❌ No se encontró la carpeta: {image_folder}")
        
        results = []
        
        # Obtener todas las imágenes
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(valid_extensions)]
        
        if not image_files:
            print(f"⚠️  No se encontraron imágenes en: {image_folder}")
            return results
        
        print(f"\n🔍 Procesando {len(image_files)} imágenes...\n")
        
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            try:
                result = self.predict(img_path, threshold)
                result['filename'] = img_file
                results.append(result)
                
                # Mostrar resultado con emoji
                emoji = "✅" if result['is_morchella'] else "❌" if result['is_morchella'] == False else "⚠️"
                print(f"{emoji} {img_file}")
                print(f"   Resultado: {result['label']}")
                print(f"   Confianza: {result['confidence_percentage']}%")
                print(f"   Score raw: {result['raw_score']}\n")
            except Exception as e:
                print(f"❌ Error procesando {img_file}: {str(e)}\n")
        
        return results


def print_header(text):
    """Imprime un encabezado bonito"""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)


def print_result_card(result):
    """Imprime un resultado de forma visual"""
    emoji = "✅" if result['is_morchella'] else "❌" if result['is_morchella'] == False else "⚠️"
    
    print(f"\n{'─' * 70}")
    print(f"{emoji}  {result['label'].upper()}  {emoji}")
    print(f"{'─' * 70}")
    print(f"📷 Imagen:     {os.path.basename(result['image_path'])}")
    print(f"🎯 Confianza:  {result['confidence_percentage']}%")
    print(f"📊 Score raw:  {result['raw_score']}")
    print(f"⚖️  Umbral:     {result['threshold']}")
    
    if result['is_morchella']:
        print(f"\n💚 ¡Es un hongo Morchella!")
    elif result['is_morchella'] == False:
        print(f"\n❤️  NO es un hongo Morchella")
    else:
        print(f"\n💛 Resultado incierto - Se necesita más evidencia")
    
    print(f"{'─' * 70}")


# EJEMPLO DE USO
if __name__ == "__main__":
    print_header("🍄 CLASIFICADOR DE HONGOS MORCHELLA 🍄")
    
    try:
        # Inicializar clasificador (busca automáticamente el modelo)
        classifier = MorchellaClassifier()
        
        # Clasificar únicamente test_image.jpeg
        print_header("ANÁLISIS DE test_image.jpeg")
        
        image_path = 'test_image6.jpeg'
        
        if os.path.exists(image_path):
            print(f"📸 Analizando imagen: {image_path}\n")
            
            result = classifier.predict(image_path, threshold=0.7)
            print_result_card(result)
            
            print_header("✅ ANÁLISIS COMPLETADO")
           
        else:
            print(f"\n❌ No se encontró la imagen: {image_path}")
            print("💡 Coloca una imagen llamada 'test_image.jpeg' en el directorio del script")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\n💡 SOLUCIÓN:")
        print("   1. Asegúrate de haber entrenado el modelo primero")
        print("   2. El archivo .tflite debe estar en el mismo directorio")
        print("   3. O especifica la ruta: MorchellaClassifier('ruta/al/modelo.tflite')")
    
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()