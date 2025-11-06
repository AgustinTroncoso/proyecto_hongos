# Contenido para: scripts/convert.py

import tensorflow as tf

KERAS_MODEL_PATH = 'models/morchella_detector.keras'
TFLITE_MODEL_PATH = 'models/morchella_detector.tflite'

def main():
    print(f"Cargando modelo Keras desde {KERAS_MODEL_PATH}...")
    # Es importante cargar el modelo SIN compilar para la conversión
    model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False) 
    
    print("Iniciando conversión a TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Aplicar optimizaciones (reduce el tamaño)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    print(f"Guardando modelo TFLite en {TFLITE_MODEL_PATH}...")
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
        
    print("¡Conversión completada!")

if __name__ == "__main__":
    main()