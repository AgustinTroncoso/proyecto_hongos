import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import shutil

# Configuración optimizada para dataset pequeño
IMG_SIZE = 224
BATCH_SIZE = 16  # Más pequeño para dataset pequeño
EPOCHS = 100  # Más épocas con early stopping
LEARNING_RATE = 0.0001
DATASET_PATH = 'dataset_augmented'

print("=" * 60)
print("ENTRENAMIENTO OPTIMIZADO PARA DATASET PEQUEÑO (81+81 imágenes)")
print("=" * 60)

# 1. DIVISIÓN MANUAL DEL DATASET (80/10/10 train/val/test)
print("\n" + "=" * 60)
print("PASO 1: División estratégica del dataset")
print("=" * 60)

def create_split_dataset(source_dir, split_dir):
    """Divide el dataset en train/val/test manualmente"""
    
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    
    # Crear estructura
    for split in ['train', 'val', 'test']:
        for class_name in ['morchella', 'no_morchella']:
            os.makedirs(os.path.join(split_dir, split, class_name), exist_ok=True)
    
    # Procesar cada clase
    for class_name in ['morchella', 'no_morchella']:
        class_path = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Dividir: 70% train, 15% val, 15% test
        train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
        
        # Copiar archivos
        for img in train_imgs:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(split_dir, 'train', class_name, img)
            )
        for img in val_imgs:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(split_dir, 'val', class_name, img)
            )
        for img in test_imgs:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(split_dir, 'test', class_name, img)
            )
        
        print(f"\n{class_name}:")
        print(f"  Train: {len(train_imgs)} imágenes")
        print(f"  Val:   {len(val_imgs)} imágenes")
        print(f"  Test:  {len(test_imgs)} imágenes")

# Crear división
split_dir = 'dataset_split'
create_split_dataset(DATASET_PATH, split_dir)

# 2. DATA AUGMENTATION AGRESIVO para dataset pequeño
print("\n" + "=" * 60)
print("PASO 2: Data Augmentation agresivo")
print("=" * 60)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,           # Mayor rotación
    width_shift_range=0.3,       # Mayor desplazamiento
    height_shift_range=0.3,
    shear_range=0.3,            # Mayor transformación
    zoom_range=0.3,             # Mayor zoom
    horizontal_flip=True,
    vertical_flip=True,          # También vertical
    brightness_range=[0.7, 1.3], # Variación de brillo
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generadores
train_generator = train_datagen.flow_from_directory(
    os.path.join(split_dir, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(split_dir, 'val'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(split_dir, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\nDataset dividido:")
print(f"Train: {train_generator.samples} imágenes")
print(f"Val:   {validation_generator.samples} imágenes")
print(f"Test:  {test_generator.samples} imágenes")

# 3. MODELO CON TRANSFER LEARNING (MobileNetV2)
print("\n" + "=" * 60)
print("PASO 3: Creando modelo con Transfer Learning")
print("=" * 60)

base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Congelar modelo base inicialmente
base_model.trainable = False

# Modelo completo con regularización fuerte
model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Modelo base
    base_model,
    
    # Global Average Pooling
    layers.GlobalAveragePooling2D(),
    
    # Batch Normalization
    layers.BatchNormalization(),
    
    # Dropout fuerte
    layers.Dropout(0.6),
    
    # Capa densa con L2 regularization
    layers.Dense(256, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(0.01)),
    
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(128, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(0.01)),
    
    layers.Dropout(0.4),
    
    # Salida
    layers.Dense(1, activation='sigmoid')
])

# Compilar
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

model.summary()

# 4. CALLBACKS AVANZADOS
print("\n" + "=" * 60)
print("PASO 4: Configurando callbacks")
print("=" * 60)

callbacks = [
    # Early Stopping con paciencia alta
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Guardar mejor modelo
    keras.callbacks.ModelCheckpoint(
        'best_morchella_small_dataset.h5',
        monitor='val_auc',  # Usar AUC para dataset pequeño
        save_best_only=True,
        verbose=1
    ),
    
    # TensorBoard para visualización
    keras.callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1
    )
]

# 5. ENTRENAMIENTO FASE 1 (capas congeladas)
print("\n" + "=" * 60)
print("PASO 5: Fase 1 - Transfer Learning (capas congeladas)")
print("=" * 60)

history1 = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1,
    # Importante: class_weight para balancear
    class_weight={0: 1.0, 1: 1.0}
)

# 6. FINE-TUNING (descongelar capas)
print("\n" + "=" * 60)
print("PASO 6: Fase 2 - Fine-tuning")
print("=" * 60)

# Descongelar últimas 50 capas
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 50

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Capas entrenables: {sum([1 for layer in model.layers if layer.trainable])}")

# Re-compilar con LR más bajo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

# Continuar entrenamiento
history2 = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# 7. EVALUACIÓN EN TEST SET
print("\n" + "=" * 60)
print("PASO 7: Evaluación en Test Set")
print("=" * 60)

# Cargar mejor modelo
best_model = keras.models.load_model('best_morchella_small_dataset.h5')

# Evaluar
test_results = best_model.evaluate(test_generator)
print(f"\n📊 RESULTADOS EN TEST SET:")
print(f"Loss:      {test_results[0]:.4f}")
print(f"Accuracy:  {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
print(f"Precision: {test_results[2]:.4f}")
print(f"Recall:    {test_results[3]:.4f}")
print(f"AUC:       {test_results[4]:.4f}")

# Calcular F1-Score
f1_score = 2 * (test_results[2] * test_results[3]) / (test_results[2] + test_results[3])
print(f"F1-Score:  {f1_score:.4f}")

# 8. ANÁLISIS DE PREDICCIONES
print("\n" + "=" * 60)
print("PASO 8: Análisis de predicciones")
print("=" * 60)

# Obtener predicciones
test_generator.reset()
predictions = best_model.predict(test_generator, verbose=1)
y_pred = (predictions > 0.7).astype(int).flatten()  # Umbral 70%
y_true = test_generator.classes

# Matriz de confusión
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)
print("\n📈 Matriz de Confusión:")
print(f"                Pred: No-Morchella  Pred: Morchella")
print(f"True No-Morchella:   {cm[0][0]:3d}              {cm[0][1]:3d}")
print(f"True Morchella:      {cm[1][0]:3d}              {cm[1][1]:3d}")

print("\n📋 Reporte de Clasificación:")
print(classification_report(y_true, y_pred, 
                          target_names=['No-Morchella', 'Morchella']))

# 9. CONVERSIÓN A TENSORFLOW LITE
print("\n" + "=" * 60)
print("PASO 9: Conversión a TensorFlow Lite")
print("=" * 60)

converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Cuantización para dataset representativo
def representative_dataset():
    for i in range(min(100, train_generator.samples // BATCH_SIZE)):
        batch = next(iter(train_generator))
        yield [batch[0]]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

# Guardar
with open('morchella_classifier_small.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"✅ Modelo guardado: morchella_classifier_small.tflite")
print(f"📦 Tamaño: {len(tflite_model) / 1024 / 1024:.2f} MB")

# 10. VISUALIZACIÓN DE RESULTADOS
print("\n" + "=" * 60)
print("PASO 10: Generando visualizaciones")
print("=" * 60)

# Combinar historiales
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']
auc = history1.history['auc'] + history2.history['auc']
val_auc = history1.history['val_auc'] + history2.history['val_auc']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy
axes[0, 0].plot(acc, label='Train Accuracy', linewidth=2)
axes[0, 0].plot(val_acc, label='Val Accuracy', linewidth=2)
axes[0, 0].axvline(x=len(history1.history['accuracy']), color='r', 
                   linestyle='--', label='Fine-tuning')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title('Model Accuracy')

# Loss
axes[0, 1].plot(loss, label='Train Loss', linewidth=2)
axes[0, 1].plot(val_loss, label='Val Loss', linewidth=2)
axes[0, 1].axvline(x=len(history1.history['loss']), color='r', 
                   linestyle='--', label='Fine-tuning')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title('Model Loss')

# AUC
axes[1, 0].plot(auc, label='Train AUC', linewidth=2)
axes[1, 0].plot(val_auc, label='Val AUC', linewidth=2)
axes[1, 0].axvline(x=len(history1.history['auc']), color='r', 
                   linestyle='--', label='Fine-tuning')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('AUC')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title('Model AUC')

# Matriz de confusión
im = axes[1, 1].imshow(cm, cmap='Blues')
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_xticklabels(['No-Morchella', 'Morchella'])
axes[1, 1].set_yticklabels(['No-Morchella', 'Morchella'])
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('True')
axes[1, 1].set_title('Confusion Matrix (Test Set)')

for i in range(2):
    for j in range(2):
        text = axes[1, 1].text(j, i, cm[i, j],
                              ha="center", va="center", color="black", fontsize=20)

plt.colorbar(im, ax=axes[1, 1])
plt.tight_layout()
plt.savefig('training_results_small_dataset.png', dpi=300)
print("✅ Gráficas guardadas: training_results_small_dataset.png")

# 11. GUARDAR ETIQUETAS
with open('labels.txt', 'w') as f:
    for label, idx in train_generator.class_indices.items():
        f.write(f"{idx} {label}\n")

print("\n" + "=" * 60)
print("✅ ¡ENTRENAMIENTO COMPLETADO!")
print("=" * 60)
print("\n📁 Archivos generados:")
print("1. best_morchella_small_dataset.h5")
print("2. morchella_classifier_small.tflite")
print("3. labels.txt")
print("4. training_results_small_dataset.png")
print("\n💡 RECOMENDACIONES:")
print("- El modelo está optimizado para dataset pequeño")
print("- Se usó data augmentation agresivo")
print("- La métrica AUC es más confiable que accuracy")
print("- Considera recolectar más imágenes si es posible")
print("- Prueba el modelo extensivamente antes de usar en producción")