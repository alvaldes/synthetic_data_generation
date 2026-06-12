# 📝 User Story to Development Tasks

Generador automático de tareas de desarrollo a partir de historias de usuario usando Ollama.

> ⚠️ **Documento actualizado**: Este pipeline ahora vive en `src/dataforge/use_cases/salony/scripts/`.
> Los notebooks y scripts CLI antiguos fueron reemplazados por pipelines más robustos.

## 🎯 Descripción

Este pipeline toma historias de usuario en formato estructurado y las descompone automáticamente en tareas de desarrollo específicas, accionables y no superpuestas.

### Input

CSV con columna `input` conteniendo JSON:

```json
{
  "project": "Nombre del proyecto",
  "summary": "Resumen de la historia",
  "description": "Descripción detallada de la historia de usuario"
}
```

### Output

CSV con tareas individuales:

- `user_story_id`: ID de la historia original
- `project`: Nombre del proyecto
- `user_story_summary`: Resumen de la historia
- `user_story_description`: Descripción completa
- `task_number`: Número de la tarea
- `task_summary`: Resumen de la tarea
- `task_description`: Descripción detallada de la tarea

## 🚀 Uso

### Pipeline Generador Simple

```bash
python src/dataforge/use_cases/salony/scripts/salony_single_generator_pipeline.py output.csv
```

**Con opciones:**

```bash
python src/dataforge/use_cases/salony/scripts/salony_single_generator_pipeline.py output.csv \
  --model llama3.2 \
  --batch-size 4 \
  --temperature 0.7 \
  --num-predict 1000 \
  --use-judge
```

### Pipeline Dual Generator con Comparación

```bash
python src/dataforge/use_cases/salony/scripts/salony_dual_generator_pipeline.py output.csv
```

Compara dos modelos (e.g., `llama3.1:8b` vs `qwen3:8b`) y usa un juez LLM para seleccionar el mejor resultado:

```bash
python src/dataforge/use_cases/salony/scripts/salony_dual_generator_pipeline.py output.csv \
  --model-a llama3.1:8b \
  --model-b qwen3:8b \
  --judge-model llama3.1:8b \
  --batch-size 2
```

### Análisis de Resultados

```bash
# Métricas agregadas (mean, std, pass rate, win rate)
python src/dataforge/use_cases/salony/scripts/aggregate_metrics.py resultados_judge.csv

# Desglose por criterio de evaluación
python src/dataforge/use_cases/salony/scripts/criterion_breakdown.py resultados_judge.csv

# Visualizaciones (boxplots, comparaciones)
python src/dataforge/use_cases/salony/scripts/plots.py resultados_judge.csv [--show]

# Consolidación de múltiples tests en reporte unificado
python src/dataforge/use_cases/salony/scripts/consolidate_results.py
```

## 📋 Preparar tu Dataset

### Formato Requerido

Tu CSV debe tener una columna llamada `input` con JSON estructurado:

```csv
input
"{""project"": ""E-commerce"", ""summary"": ""User login"", ""description"": ""As a user, I want to...""}"
"{""project"": ""CRM"", ""summary"": ""Contact management"", ""description"": ""As a sales rep, I want to...""}"
```

Para el dataset de ejemplo del proyecto Salony, los datos están en `data/raw/salony_train.csv`.

### Crear Dataset desde Excel

```python
import pandas as pd
import json

# Cargar desde Excel
df = pd.read_excel('historias.xlsx')

# Convertir a formato requerido
df['input'] = df.apply(lambda row: json.dumps({
    'project': row['Proyecto'],
    'summary': row['Titulo'],
    'description': row['Descripcion']
}), axis=1)

# Guardar
df[['input']].to_csv('historias_formateadas.csv', index=False)
```

## ⚙️ Configuración

### Modelos Recomendados

| Modelo | Uso | Velocidad | Calidad |
|--------|-----|-----------|---------|
| `llama3.2` | General | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| `llama3.1:8b` | Judge/Comparaciones | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| `qwen3:8b` | Generación dual | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| `mistral` | Rápido | ⚡⚡⚡⚡ | ⭐⭐⭐ |

### Parámetros

- **Batch Size**: 2-4 (recomendado)
- **Temperature**: 0.3-0.7 (más baja para judge, más alta para creatividad)
- **Num Predict**: 1000-1500 (suficiente para tareas detalladas)
- **Judge Threshold**: 35/50 (aprobación por defecto)

## 🐛 Troubleshooting

### "Ollama no encontrado"

```bash
ollama serve  # En una terminal
ollama pull llama3.2  # Descargar modelo
```

### "No tasks parsed"

- Reducir temperature a 0.5
- Aumentar num_predict a 1500
- Verificar que el modelo siga el formato

### "Out of memory"

- Reducir batch_size a 1
- Procesar en chunks más pequeños

## 💡 Tips

✅ **Buena historia:**

- Descripción clara y detallada
- Contexto del proyecto
- Criterios de aceptación

❌ **Mala historia:**

- Muy corta o vaga
- Sin contexto
- Ambigua

## 📚 Recursos

- [Documentación Principal](../README.md)
- [Pipeline Generador Simple](../src/dataforge/use_cases/salony/scripts/salony_single_generator_pipeline.py)
- [Pipeline Dual Generator](../src/dataforge/use_cases/salony/scripts/salony_dual_generator_pipeline.py)
- [Scripts de Análisis](../src/dataforge/use_cases/salony/scripts/)
