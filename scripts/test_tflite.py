import tensorflow as tf
import numpy as np
from PIL import Image
import os

class MorchellaClassifier:
    def __init__(self, model_path='morchella_classifier.tflite'):
        """Inicializa el clasificador con el modelo TFLite"""
        # Cargar el modelo TFLite
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Obtener detalles de entrada y salida
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Obtener el tamaño de entrada esperado
        self.input_shape = self.input_details[0]['shape']
        self.img_size = self.input_shape[1]
        
        print(f"Modelo cargado correctamente")
        print(f"Tamaño de entrada: {self.img_size}x{self.img_size}")
    
    def preprocess_image(self, image_path):
        """Preprocesa la imagen para el modelo"""
        # Cargar imagen
        img = Image.open(image_path).convert('RGB')
        
        # Redimensionar
        img = img.resize((self.img_size, self.img_size))
        
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
        confidence = float(output[0][0])
        
        # Determinar la clase
        # Si confianza >= threshold, es morchella
        # Si confianza < (1-threshold), NO es morchella
        # En medio: incierto
        
        if confidence >= threshold:
            is_morchella = True
            label = "Morchella"
            certainty = confidence * 100
        elif confidence <= (1 - threshold):
            is_morchella = False
            label = "No Morchella"
            certainty = (1 - confidence) * 100
        else:
            is_morchella = None
            label = "Incierto"
            certainty = max(confidence, 1 - confidence) * 100
        
        return {
            'is_morchella': is_morchella,
            'label': label,
            'confidence_percentage': round(certainty, 2),
            'raw_score': round(confidence, 4),
            'threshold': threshold
        }
    
    def predict_batch(self, image_folder, threshold=0.7):
        """Predice múltiples imágenes en una carpeta"""
        results = []
        
        # Obtener todas las imágenes
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(valid_extensions)]
        
        print(f"\nProcesando {len(image_files)} imágenes...\n")
        
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            try:
                result = self.predict(img_path, threshold)
                result['filename'] = img_file
                results.append(result)
                
                # Mostrar resultado
                print(f"📷 {img_file}")
                print(f"   Resultado: {result['label']}")
                print(f"   Confianza: {result['confidence_percentage']}%")
                print(f"   Score raw: {result['raw_score']}\n")
            except Exception as e:
                print(f"❌ Error procesando {img_file}: {str(e)}\n")
        
        return results


# EJEMPLO DE USO
if __name__ == "__main__":
    print("=" * 60)
    print("CLASIFICADOR DE HONGOS MORCHELLA")
    print("=" * 60)
    
    # Inicializar clasificador
    classifier = MorchellaClassifier('morchella_classifier.tflite')
    
    # OPCIÓN 1: Clasificar una imagen individual
    print("\n--- PRUEBA CON IMAGEN INDIVIDUAL ---")
    image_path = 'test_image.jpeg'  # Cambia esto por tu imagen
    
    if os.path.exists(image_path):
        result = classifier.predict(image_path, threshold=0.7)
        
        print(f"\nResultado para: {image_path}")
        print(f"{'='*40}")
        print(f"Es Morchella: {result['is_morchella']}")
        print(f"Etiqueta: {result['label']}")
        print(f"Confianza: {result['confidence_percentage']}%")
        print(f"Score raw: {result['raw_score']}")
        print(f"Umbral usado: {result['threshold']}")
        
        if result['is_morchella']:
            print(f"\n✅ ¡Es un hongo Morchella con {result['confidence_percentage']}% de confianza!")
        elif result['is_morchella'] == False:
            print(f"\n❌ NO es un hongo Morchella ({result['confidence_percentage']}% seguro)")
        else:
            print(f"\n⚠️  Resultado incierto. Se necesita más evidencia.")
    else:
        print(f"❌ No se encontró la imagen: {image_path}")
    
 
    
    print("\n" + "="*60)
    print("¡Prueba completada!")
    print("="*60)