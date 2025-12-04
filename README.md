# 🍄 Proyecto de Clasificación de Hongos Morchella con MLflow

## 📖 Descripción
Proyecto de Deep Learning para clasificar hongos Morchella usando Transfer Learning (MobileNetV2) con tracking de experimentos mediante MLflow.

## 🚀 Inicio Rápido

### 1. Activar el entorno virtual
```powershell
cd "c:\Users\Tito\Desktop\Hongos tenserlite\proyecto_hongos"
.\tf-env\Scripts\Activate.ps1
```

### 2. Instalar dependencias (incluye MLflow)
```powershell
pip install -r requirements.txt
```

### 3. Entrenar el modelo
```powershell
python scripts\train.py
```

### 4. Ver resultados en MLflow
```powershell
mlflow ui
```
Luego abre: **http://localhost:5000**

---

## 📊 Características de MLflow Integradas

El proyecto ahora trackea automáticamente:
- ✅ **Hiperparámetros**: Learning rate, batch size, dropout, augmentation, etc.
- ✅ **Métricas**: Accuracy, AUC, F1-score, Precision, Recall
- ✅ **Artefactos**: Modelos (.h5, .tflite), gráficas, labels
- ✅ **Comparación**: Compara múltiples experimentos lado a lado

---

## 🧪 Experimentación

### Opción 1: Manual
1. Edita hiperparámetros en `scripts/train.py` (líneas 13-32)
2. Ejecuta: `python scripts\train.py`
3. Compara resultados en MLflow UI

### Opción 2: Automática
```powershell
python scripts\run_experiments.py
```
Esto ejecuta 5 experimentos predefinidos y muestra cuál funciona mejor.

---

## 📚 Documentación

- **[GUIA_MLFLOW.md](GUIA_MLFLOW.md)**: Guía completa de cómo usar MLflow para optimizar tu modelo

---

## 📁 Estructura del Proyecto

```
proyecto_hongos/
├── scripts/
│   ├── train.py              # Script de entrenamiento con MLflow
│   ├── run_experiments.py    # Experimentación automática
│   └── test_tflite.py        # Prueba del modelo TFLite
├── dataset_augmented/        # Dataset aumentado
├── dataset_split/            # Train/Val/Test splits
├── mlruns/                   # Experimentos de MLflow
├── requirements.txt          # Dependencias (incluye MLflow)
├── GUIA_MLFLOW.md           # Guía completa de MLflow
└── README.md                 # Este archivo
```

---

## 🎯 Métricas del Modelo

El modelo se evalúa con:
- **Accuracy**: % de clasificaciones correctas
- **AUC**: Capacidad de discriminación (métrica principal)
- **F1-Score**: Balance entre precisión y recall
- **Precision**: Evitar falsos positivos
- **Recall**: Detectar todos los positivos

---

## 💡 Tips

1. **Primera vez**: Usa la configuración por defecto para establecer baseline
2. **Experimentar**: Cambia UN hiperparámetro a la vez
3. **Comparar**: Usa MLflow UI para ver qué funciona mejor
4. **Optimizar**: Itera hasta alcanzar tus objetivos de performance

---

## 🔧 Hiperparámetros Principales

Modifica estos en `scripts/train.py`:

```python
# Básicos
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
EPOCHS = 100

# Data Augmentation
ROTATION_RANGE = 40
ZOOM_RANGE = 0.3

# Regularización
DROPOUT_RATE_1 = 0.6
L2_REGULARIZATION = 0.01

# Fine-tuning
FINE_TUNE_LAYERS = 50
```

Ver **GUIA_MLFLOW.md** para detalles sobre qué hace cada parámetro.

---

## 📞 Comandos Útiles

```powershell
# Activar entorno
.\tf-env\Scripts\Activate.ps1

# Entrenar
python scripts\train.py

# Experimentar automáticamente
python scripts\run_experiments.py

# Ver experimentos
mlflow ui

# Desactivar entorno
deactivate
```

---

## 🏆 ¿Cómo Mejorar el Modelo?

1. Lee **GUIA_MLFLOW.md** (guía completa)
2. Ejecuta baseline: `python scripts\train.py`
3. Ve resultados: `mlflow ui`
4. Experimenta con diferentes hiperparámetros
5. Compara en MLflow UI
6. Itera hasta alcanzar tu objetivo

---

## ✨ ¿Qué hay de nuevo con MLflow?

**Antes**:
- ❌ Perdías track de qué parámetros usaste
- ❌ No podías comparar experimentos fácilmente
- ❌ Difícil reproducir resultados
- ❌ No había historial de modelos

**Ahora con MLflow**:
- ✅ Todo se guarda automáticamente
- ✅ Interfaz web para comparar experimentos
- ✅ Reproducible 100%
- ✅ Historial completo de todos los modelos

---

**¡Éxito con tu proyecto! 🚀🍄**
