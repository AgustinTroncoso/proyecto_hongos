import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

MODEL = os.path.join(os.path.dirname(__file__), '..', 'best_morchella_small_dataset.h5')
LABELS = os.path.join(os.path.dirname(__file__), '..', 'labels.txt')
IMG_PATH = os.path.join(os.path.dirname(__file__), '..', 'test_image6.jpeg')
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
    if not os.path.exists(MODEL):
        print('Modelo .h5 no encontrado en', MODEL)
        return
    if not os.path.exists(IMG_PATH):
        print('Imagen de prueba no encontrada en', IMG_PATH)
        return

    model = tf.keras.models.load_model(MODEL)
    labels = load_labels(LABELS)

    img = image.load_img(IMG_PATH, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, 0)

    p = float(model.predict(x)[0][0])
    # p = probabilidad de la clase index 1 (morchella) en el modelo sigmoid
    prob_morchella = p
    prob_no = 1.0 - p
    pred_idx = 1 if prob_morchella > prob_no else 0
    conf = max(prob_morchella, prob_no)
    print(f"HDF5 -> Morchella: {prob_morchella*100:.2f}%  |  No-Morchella: {prob_no*100:.2f}%")
    print(f"Predicción: {labels.get(pred_idx, pred_idx)} con {conf*100:.2f}% de confianza")

if __name__ == '__main__':
    main()
