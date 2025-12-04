"""
🔬 Script de Experimentación Automática con MLflow
====================================================

Este script ejecuta múltiples experimentos automáticamente,
probando diferentes combinaciones de hiperparámetros.

Uso:
    python scripts/run_experiments.py

El script:
- Ejecuta 5 experimentos predefinidos
- Guarda todos los resultados en MLflow
- Al final, muestra un resumen comparativo
- Te dice cuál configuración funcionó mejor
"""

import subprocess
import time
import mlflow
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN DE EXPERIMENTOS
# ============================================================================

# Experimentos predefinidos (puedes modificar o agregar más)
EXPERIMENTS = [
    {
        "name": "Experimento 1: Baseline",
        "description": "Configuración inicial conservadora",
        "params": {
            "LEARNING_RATE": 0.0001,
            "BATCH_SIZE": 16,
            "ROTATION_RANGE": 40,
            "WIDTH_SHIFT": 0.3,
            "HEIGHT_SHIFT": 0.3,
            "SHEAR_RANGE": 0.3,
            "ZOOM_RANGE": 0.3,
            "DROPOUT_RATE_1": 0.6,
            "DROPOUT_RATE_2": 0.5,
            "DROPOUT_RATE_3": 0.4,
            "L2_REGULARIZATION": 0.01,
            "FINE_TUNE_LAYERS": 50,
        }
    },
    {
        "name": "Experimento 2: LR Alto",
        "description": "Learning rate más alto para aprendizaje rápido",
        "params": {
            "LEARNING_RATE": 0.001,  # 🔼 Cambio principal
            "BATCH_SIZE": 16,
            "ROTATION_RANGE": 40,
            "WIDTH_SHIFT": 0.3,
            "HEIGHT_SHIFT": 0.3,
            "SHEAR_RANGE": 0.3,
            "ZOOM_RANGE": 0.3,
            "DROPOUT_RATE_1": 0.6,
            "DROPOUT_RATE_2": 0.5,
            "DROPOUT_RATE_3": 0.4,
            "L2_REGULARIZATION": 0.01,
            "FINE_TUNE_LAYERS": 50,
        }
    },
    {
        "name": "Experimento 3: Menos Dropout",
        "description": "Reducir dropout para dar más capacidad al modelo",
        "params": {
            "LEARNING_RATE": 0.0001,
            "BATCH_SIZE": 16,
            "ROTATION_RANGE": 40,
            "WIDTH_SHIFT": 0.3,
            "HEIGHT_SHIFT": 0.3,
            "SHEAR_RANGE": 0.3,
            "ZOOM_RANGE": 0.3,
            "DROPOUT_RATE_1": 0.4,  # 🔽 Cambio
            "DROPOUT_RATE_2": 0.3,  # 🔽 Cambio
            "DROPOUT_RATE_3": 0.2,  # 🔽 Cambio
            "L2_REGULARIZATION": 0.01,
            "FINE_TUNE_LAYERS": 50,
        }
    },
    {
        "name": "Experimento 4: Augmentation Conservador",
        "description": "Menos augmentation para preservar características originales",
        "params": {
            "LEARNING_RATE": 0.0001,
            "BATCH_SIZE": 16,
            "ROTATION_RANGE": 20,   # 🔽 Cambio
            "WIDTH_SHIFT": 0.2,     # 🔽 Cambio
            "HEIGHT_SHIFT": 0.2,    # 🔽 Cambio
            "SHEAR_RANGE": 0.2,     # 🔽 Cambio
            "ZOOM_RANGE": 0.2,      # 🔽 Cambio
            "DROPOUT_RATE_1": 0.6,
            "DROPOUT_RATE_2": 0.5,
            "DROPOUT_RATE_3": 0.4,
            "L2_REGULARIZATION": 0.01,
            "FINE_TUNE_LAYERS": 50,
        }
    },
    {
        "name": "Experimento 5: Batch Size Pequeño",
        "description": "Batch size más pequeño para mejor generalización",
        "params": {
            "LEARNING_RATE": 0.0001,
            "BATCH_SIZE": 8,  # 🔽 Cambio principal
            "ROTATION_RANGE": 40,
            "WIDTH_SHIFT": 0.3,
            "HEIGHT_SHIFT": 0.3,
            "SHEAR_RANGE": 0.3,
            "ZOOM_RANGE": 0.3,
            "DROPOUT_RATE_1": 0.6,
            "DROPOUT_RATE_2": 0.5,
            "DROPOUT_RATE_3": 0.4,
            "L2_REGULARIZATION": 0.01,
            "FINE_TUNE_LAYERS": 50,
        }
    },
]

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def update_train_script(params):
    """Actualiza train.py con los parámetros del experimento"""
    
    with open('scripts/train.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Actualizar cada parámetro
    for param_name, param_value in params.items():
        # Buscar la línea del parámetro
        import re
        pattern = f'{param_name} = [^\n]+'
        replacement = f'{param_name} = {param_value}'
        content = re.sub(pattern, replacement, content)
    
    with open('scripts/train.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Script actualizado con nuevos parámetros")


def run_training():
    """Ejecuta el script de entrenamiento"""
    
    print("\n" + "="*70)
    print("🚀 Iniciando entrenamiento...")
    print("="*70)
    
    try:
        # Ejecutar train.py
        result = subprocess.run(
            ['python', 'scripts/train.py'],
            capture_output=True,
            text=True,
            timeout=7200  # Timeout de 2 horas
        )
        
        if result.returncode == 0:
            print("✅ Entrenamiento completado exitosamente")
            return True
        else:
            print("❌ Error durante el entrenamiento:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Entrenamiento excedió el tiempo límite (2 horas)")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False


def get_latest_run_metrics():
    """Obtiene las métricas del último run de MLflow"""
    
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("morchella_classification")
    
    if experiment is None:
        return None
    
    # Obtener el último run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        return None
    
    latest_run = runs[0]
    metrics = latest_run.data.metrics
    
    return {
        "run_id": latest_run.info.run_id,
        "test_accuracy": metrics.get("test_accuracy", 0),
        "test_auc": metrics.get("test_auc", 0),
        "test_f1_score": metrics.get("test_f1_score", 0),
        "test_precision": metrics.get("test_precision", 0),
        "test_recall": metrics.get("test_recall", 0),
    }


def print_experiment_summary(experiment, metrics):
    """Imprime resumen del experimento"""
    
    print("\n" + "="*70)
    print(f"📊 RESUMEN DEL EXPERIMENTO")
    print("="*70)
    print(f"Nombre: {experiment['name']}")
    print(f"Descripción: {experiment['description']}")
    print("\nMétricas:")
    print(f"  • Test Accuracy:  {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
    print(f"  • Test AUC:       {metrics['test_auc']:.4f}")
    print(f"  • Test F1-Score:  {metrics['test_f1_score']:.4f}")
    print(f"  • Test Precision: {metrics['test_precision']:.4f}")
    print(f"  • Test Recall:    {metrics['test_recall']:.4f}")
    print(f"\nRun ID: {metrics['run_id']}")
    print("="*70)


def print_final_comparison(results):
    """Imprime comparación final de todos los experimentos"""
    
    print("\n\n" + "="*80)
    print("🏆 COMPARACIÓN FINAL DE TODOS LOS EXPERIMENTOS")
    print("="*80)
    
    # Ordenar por AUC (métrica más importante)
    sorted_results = sorted(results, key=lambda x: x['metrics']['test_auc'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Experimento':<30} {'Accuracy':<12} {'AUC':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results, 1):
        exp_name = result['experiment']['name']
        metrics = result['metrics']
        
        # Emoji según ranking
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        
        print(f"{emoji} {i:<4} {exp_name:<30} "
              f"{metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:5.2f}%)  "
              f"{metrics['test_auc']:.4f}    "
              f"{metrics['test_f1_score']:.4f}")
    
    # Mejor experimento
    best = sorted_results[0]
    
    print("\n" + "="*80)
    print("🎯 MEJOR CONFIGURACIÓN ENCONTRADA:")
    print("="*80)
    print(f"\nExperimento: {best['experiment']['name']}")
    print(f"Descripción: {best['experiment']['description']}")
    print(f"\nMétricas:")
    print(f"  • Test AUC:       {best['metrics']['test_auc']:.4f}")
    print(f"  • Test Accuracy:  {best['metrics']['test_accuracy']:.4f} ({best['metrics']['test_accuracy']*100:.2f}%)")
    print(f"  • Test F1-Score:  {best['metrics']['test_f1_score']:.4f}")
    
    print(f"\n📋 Hiperparámetros:")
    for param, value in best['experiment']['params'].items():
        print(f"  • {param}: {value}")
    
    print(f"\n🔗 Run ID: {best['metrics']['run_id']}")
    print("\n💡 Tip: Ve a MLflow UI (mlflow ui) para ver más detalles y gráficas")
    print("="*80)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que ejecuta todos los experimentos"""
    
    print("\n" + "🔬"*30)
    print("EXPERIMENTACIÓN AUTOMÁTICA CON MLFLOW")
    print("🔬"*30)
    print(f"\n📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🧪 Total de experimentos: {len(EXPERIMENTS)}")
    print(f"⏱️  Tiempo estimado: {len(EXPERIMENTS) * 40} minutos (aprox.)")
    
    input("\n⚠️  Presiona ENTER para comenzar la experimentación...")
    
    results = []
    
    for i, experiment in enumerate(EXPERIMENTS, 1):
        print("\n\n" + "🔬"*35)
        print(f"EXPERIMENTO {i}/{len(EXPERIMENTS)}: {experiment['name']}")
        print("🔬"*35)
        print(f"Descripción: {experiment['description']}")
        print(f"\nParámetros clave:")
        
        # Mostrar solo parámetros que cambiaron respecto al baseline
        baseline = EXPERIMENTS[0]['params']
        for param, value in experiment['params'].items():
            if value != baseline.get(param):
                print(f"  🔄 {param}: {baseline.get(param)} → {value}")
        
        print("\n" + "-"*70)
        
        # Actualizar script
        update_train_script(experiment['params'])
        
        # Esperar un momento
        time.sleep(2)
        
        # Ejecutar entrenamiento
        start_time = time.time()
        success = run_training()
        elapsed_time = time.time() - start_time
        
        if success:
            # Obtener métricas
            metrics = get_latest_run_metrics()
            
            if metrics:
                print_experiment_summary(experiment, metrics)
                
                results.append({
                    'experiment': experiment,
                    'metrics': metrics,
                    'elapsed_time': elapsed_time
                })
            else:
                print("⚠️ No se pudieron obtener las métricas del experimento")
        else:
            print(f"❌ Experimento {i} falló. Continuando con el siguiente...")
        
        # Pausa entre experimentos
        if i < len(EXPERIMENTS):
            print(f"\n⏸️  Esperando 10 segundos antes del próximo experimento...")
            time.sleep(10)
    
    # Comparación final
    if results:
        print_final_comparison(results)
        
        # Guardar resumen en archivo
        with open('experiments_summary.txt', 'w', encoding='utf-8') as f:
            f.write(f"Resumen de Experimentación - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for i, result in enumerate(sorted(results, key=lambda x: x['metrics']['test_auc'], reverse=True), 1):
                f.write(f"{i}. {result['experiment']['name']}\n")
                f.write(f"   AUC: {result['metrics']['test_auc']:.4f}\n")
                f.write(f"   Accuracy: {result['metrics']['test_accuracy']:.4f}\n")
                f.write(f"   Tiempo: {result['elapsed_time']/60:.1f} minutos\n\n")
        
        print("\n📝 Resumen guardado en: experiments_summary.txt")
    else:
        print("\n❌ No se completó ningún experimento exitosamente")
    
    print("\n✅ Experimentación completada!")
    print("\n💡 Próximos pasos:")
    print("   1. Ejecuta 'mlflow ui' para ver todos los resultados")
    print("   2. Compara los experimentos en la interfaz web")
    print("   3. Usa la mejor configuración para entrenar tu modelo final")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Experimentación interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()
