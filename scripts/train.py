import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuración
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
DATASET_PATH = 'dataset'  # Ruta a tu carpeta dataset

# 1. PREPARACIÓN Y DIVISIÓN DEL DATASET
print("=" * 50)
print("PASO 1: Preparando el dataset")
print("=" * 50)

# Data Augmentation para evitar overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% entrenamiento, 20% validación
    rotation_range=30,      # Rotación aleatoria
    width_shift_range=0.2,  # Desplazamiento horizontal
    height_shift_range=0.2, # Desplazamiento vertical
    shear_range=0.2,        # Transformación de corte
    zoom_range=0.2,         # Zoom aleatorio
    horizontal_flip=True,   # Volteo horizontal
    fill_mode='nearest'
)

# Solo rescalado para validación (sin augmentation)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Generadores de datos
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Clasificación binaria
    subset='training',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print(f"\nClases encontradas: {train_generator.class_indices}")
print(f"Total imágenes entrenamiento: {train_generator.samples}")
print(f"Total imágenes validación: {validation_generator.samples}")

# 2. CREACIÓN DEL MODELO CON TRANSFER LEARNING
print("\n" + "=" * 50)
print("PASO 2: Creando modelo con Transfer Learning")
print("=" * 50)

# Cargar MobileNetV2 pre-entrenado (ideal para móviles)
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,  # Sin las capas de clasificación
    weights='imagenet'  # Pesos pre-entrenados
)

# Congelar las capas del modelo base
base_model.trainable = False

print(f"Modelo base: MobileNetV2")
print(f"Capas congeladas: {len(base_model.layers)}")

# Crear el modelo completo
model = keras.Sequential([
    # Capa de entrada
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Modelo pre-entrenado
    base_model,
    
    # Capas personalizadas para clasificación
    layers.GlobalAveragePooling2D(),
    
    # Dropout para prevenir overfitting
    layers.Dropout(0.5),
    
    # Capa densa con regularización
    layers.Dense(128, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(0.01)),
    
    layers.Dropout(0.3),
    
    # Capa de salida (sigmoid para clasificación binaria)
    layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

model.summary()

# 3. CALLBACKS PARA CONTROL DE OVERFITTING
print("\n" + "=" * 50)
print("PASO 3: Configurando callbacks")
print("=" * 50)

# Early Stopping: detiene si no mejora
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,  # Espera 10 épocas sin mejora
    restore_best_weights=True,
    verbose=1
)

# ReduceLROnPlateau: reduce learning rate si no mejora
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Reduce a la mitad
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# ModelCheckpoint: guarda el mejor modelo
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_morchella_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# 4. ENTRENAMIENTO INICIAL (capas congeladas)
print("\n" + "=" * 50)
print("PASO 4: Entrenamiento inicial (Transfer Learning)")
print("=" * 50)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# 5. FINE-TUNING (descongelar algunas capas)
print("\n" + "=" * 50)
print("PASO 5: Fine-tuning del modelo")
print("=" * 50)

# Descongelar las últimas 30 capas
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"Capas entrenables después de fine-tuning: {len([l for l in base_model.layers if l.trainable])}")

# Re-compilar con learning rate más bajo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# Continuar entrenamiento
history_fine = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# 6. EVALUACIÓN DEL MODELO
print("\n" + "=" * 50)
print("PASO 6: Evaluación final")
print("=" * 50)

# Cargar el mejor modelo guardado
model = keras.models.load_model('best_morchella_model.h5')

# Evaluar en el conjunto de validación
results = model.evaluate(validation_generator)
print(f"\nResultados en validación:")
print(f"Loss: {results[0]:.4f}")
print(f"Accuracy: {results[1]:.4f}")
print(f"Precision: {results[2]:.4f}")
print(f"Recall: {results[3]:.4f}")

# 7. CONVERSIÓN A TENSORFLOW LITE
print("\n" + "=" * 50)
print("PASO 7: Conversión a TensorFlow Lite")
print("=" * 50)

# Convertir a TFLite con optimizaciones
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convertir
tflite_model = converter.convert()

# Guardar el modelo .tflite
with open('morchella_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Modelo TFLite guardado: morchella_classifier.tflite")
print(f"Tamaño: {len(tflite_model) / 1024 / 1024:.2f} MB")

# 8. VISUALIZACIÓN DE RESULTADOS
print("\n" + "=" * 50)
print("PASO 8: Generando gráficas")
print("=" * 50)

# Combinar historiales
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

# Gráfica de accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.axvline(x=len(history.history['accuracy']), color='r', linestyle='--', label='Fine-tuning start')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.axvline(x=len(history.history['loss']), color='r', linestyle='--', label='Fine-tuning start')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.savefig('training_results.png')
print("Gráficas guardadas en: training_results.png")

# 9. CREAR ARCHIVO DE ETIQUETAS
with open('labels.txt', 'w') as f:
    for label, idx in train_generator.class_indices.items():
        f.write(f"{idx} {label}\n")

print("\nArchivo de etiquetas guardado: labels.txt")
print("\n" + "=" * 50)
print("¡ENTRENAMIENTO COMPLETADO!")
print("=" * 50)
print("\nArchivos generados:")
print("1. best_morchella_model.h5 (modelo Keras)")
print("2. morchella_classifier.tflite (modelo para móvil)")
print("3. labels.txt (etiquetas de clases)")
print("4. training_results.png (gráficas de entrenamiento)")
