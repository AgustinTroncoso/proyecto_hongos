import numpy as np
import tensorflow as tf
from PIL import Image
import os

TFLITE = os.path.join(os.path.dirname(__file__), '..', 'morchella_classifier_small.tflite')
LABELS = os.path.join(os.path.dirname(__file__), '..', 'labels.txt')
IMG_PATH = os.path.join(os.path.dirname(__file__), '..', 'test_image4.jpeg')
IMG_SIZE = 224
TH = 0.7

def load_labels(path):
    d = {}
    if not os.path.exists(path):
        return {0: 'no_morchella', 1: 'morchella'}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                d[int(parts[0])] = parts[1]
    return d

def main():
    if not os.path.exists(TFLITE):
        print('Modelo tflite no encontrado en', TFLITE)
        return
    if not os.path.exists(IMG_PATH):
        print('Imagen de prueba no encontrada en', IMG_PATH)
        return

    labels = load_labels(LABELS)

    interpreter = tf.lite.Interpreter(model_path=TFLITE)
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']

    img = Image.open(IMG_PATH).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    interpreter.set_tensor(inp_idx, arr)
    interpreter.invoke()
    out = interpreter.get_tensor(out_idx)
    try:
        p = float(out[0][0])
    except Exception:
        p = float(np.asarray(out).flatten()[0])

    # p = probabilidad de la clase index 1 (morchella)
    prob_morchella = p
    prob_no = 1.0 - p
    pred_idx = 1 if prob_morchella > prob_no else 0
    conf = max(prob_morchella, prob_no)
    print(f"TFLITE -> Morchella: {prob_morchella*100:.2f}%  |  No-Morchella: {prob_no*100:.2f}%")
    print(f"Predicción: {labels.get(pred_idx, pred_idx)} con {conf*100:.2f}% de confianza")

if __name__ == '__main__':
    main()
