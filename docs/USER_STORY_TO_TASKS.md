# 📝 User Story to Development Tasks

Generador automático de tareas de desarrollo a partir de historias de usuario usando Ollama.

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

### Opción 1: Jupyter Notebook (Interactivo)

```bash
cd notebooks
jupyter notebook user_story_to_tasks.ipynb
```

**Ventajas:**
- Exploración interactiva paso a paso
- Visualizaciones y análisis incluidos
- Fácil de modificar y experimentar
- Ver resultados en tiempo real

### Opción 2: Script CLI (Producción)

```bash
cd examples
python user_story_to_tasks.py input.csv output.csv
```

**Con opciones:**
```bash
python user_story_to_tasks.py historias.csv tareas.csv \
  --model llama3.2 \
  --batch-size 4 \
  --temperature 0.7 \
  --num-predict 1000
```

## 📋 Preparar tu Dataset

### Formato Requerido

Tu CSV debe tener una columna llamada `input` con JSON estructurado:

```csv
input
"{""project"": ""E-commerce"", ""summary"": ""User login"", ""description"": ""As a user, I want to...""}"
"{""project"": ""CRM"", ""summary"": ""Contact management"", ""description"": ""As a sales rep, I want to...""}"
```

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
| `mistral` | Rápido | ⚡⚡⚡⚡ | ⭐⭐⭐ |
| `codellama` | Técnico | ⚡⚡ | ⭐⭐⭐⭐⭐ |

### Parámetros

- **Batch Size**: 2-4 (recomendado)
- **Temperature**: 0.6-0.7 (balanceado)
- **Num Predict**: 1000-1500 (suficiente para tareas detalladas)

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
- [Notebook Interactivo](../notebooks/user_story_to_tasks.ipynb)
- [Script CLI](../examples/user_story_to_tasks.py)
