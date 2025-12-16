import tensorflowjs as tfjs
from tensorflow import keras

# Cargar el modelo Keras
model = keras.models.load_model('best_morchella_small_dataset.h5')

# Convertir y guardar en la carpeta 'tfjs_model'
tfjs.converters.save_keras_model(model, 'tfjs_model')
