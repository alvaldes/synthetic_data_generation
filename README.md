# LocalLLM-DataForge

**Data Pipeline Architecture** para descomposición de historias de usuario en tareas de desarrollo usando LLMs locales con Ollama.

Inspirado en [Distilabel](https://github.com/argilla-io/distilabel), organiza el procesamiento siguiendo el flujo natural de los datos: **Entrada ➔ Procesamiento LLM ➔ Validación/Reparación ➔ Salida**.

---

## 🚀 Quick Start

```bash
# 1. Install the package
pip install -e .

# 2. Install dev/test dependencies
pip install -r requirements.txt

# 3. Make sure Ollama is running
ollama serve

# 4. Pull a model
ollama pull llama3.2

# 5. Run example
python examples/demo_pipeline.py

# 6. Run tests
pytest tests/ -v
```

---

## 🐳 Docker

El proyecto incluye **Docker Compose** para correr el pipeline junto con Ollama en contenedores, sin necesidad de instalar Python ni Ollama en el host.

### Servicios

| Servicio | Imagen | Rol |
|----------|--------|-----|
| **ollama** | `ollama/ollama:latest` | Servicio LLM local (inferencia) |
| **app** | Build local desde `Dockerfile` | Ejecución de pipelines Python |

Comunicación por red interna `dataforge_network`. La app usa `OLLAMA_HOST=http://ollama:11434` para apuntar al servicio.

### Prerrequisitos

- Docker Engine 24+ con Docker Compose v2
- (Opcional) GPU para aceleración — depende de tu hardware:
  - **NVIDIA**: `nvidia-container-toolkit` instalado en el host
  - **AMD**: `rocm` y configuración de dispositivos en el compose
  - **Apple Silicon (macOS)**: Ollama usa Metal por defecto, la GPU se comparte con el host sin configuración extra

### Uso Básico

```bash
# Iniciar servicios (ollama + app)
docker compose up -d

# Ver logs de ambos servicios
docker compose logs -f
```

### Pull de Modelos

Ollama arranca sin modelos. Hay que descargarlos desde el contenedor:

```bash
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull qwen3:8b
```

### Ejecutar Pipelines

Una vez que los servicios están corriendo, usá `exec` (modo interactivo):

```bash
# Pipeline de ejemplo
docker compose exec app python examples/demo_pipeline.py

# Pipeline real con dataset Salony (single generator)
docker compose exec app python src/dataforge/use_cases/salony/scripts/salony_single_generator_pipeline.py \
  /app/data/outputs/result.csv \
  --model llama3.1:8b \
  --batch-size 4

# Pipeline dual generator
docker compose exec app python src/dataforge/use_cases/salony/scripts/salony_dual_generator_pipeline.py \
  /app/data/outputs/result.csv \
  --model-a llama3.1:8b \
  --model-b qwen3:8b \
  --judge-model llama3.1:8b
```

Para comandos únicos sin mantener el contenedor corriendo:

```bash
docker compose run --rm app python examples/demo_pipeline.py
```

### Aceleración por GPU

Ollama dentro del contenedor puede usar la GPU del host. La configuración depende de tu hardware.

#### NVIDIA (ejemplo)

Si tenés GPU NVIDIA con CUDA y `nvidia-container-toolkit` instalado, descomentá las líneas de GPU en el servicio `ollama` del `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Verificá que Docker tenga acceso:

```bash
docker compose exec ollama nvidia-smi
```

#### Otras GPUs

El mismo principio aplica para **AMD** (configurando dispositivos ROCm en el compose) o **Apple Silicon** (Ollama usa Metal automáticamente — en macOS la GPU se comparte con el host sin necesidad de configuración extra en Docker).

> **⚠️ Importante**: La aceleración por GPU es totalmente opcional. El pipeline funciona perfectamente en CPU, solo que es más lento con modelos grandes. Si no tenés GPU o preferís no configurarla, simplemente ignorá esta sección.

### Volúmenes

| Volumen | Mount en Contenedor | Propósito |
|---------|--------------------|-----------|
| `dataforge_ollama_models` | `/root/.ollama` | Modelos descargados (persistente entre reinicios) |
| `dataforge_pipeline_cache` | `/app/.cache` | Caché de pipelines (persistente) |
| `./data` (bind mount) | `/app/data` | Datos de entrada/salida — compartido con el host |

Los CSVs de entrada van en `data/raw/` y los resultados se escriben en `data/outputs/`, accesibles desde el host inmediatamente.

### Detener

```bash
# Bajar servicios (los volúmenes persisten)
docker compose down

# Bajar servicios y eliminar volúmenes (borra modelos descargados y caché)
docker compose down -v
```

---

## 📖 Features

- 🐼 **Native pandas DataFrame support** — Work with familiar data structures
- 🦙 **Ollama local model integration** — Run LLMs locally without API costs
- 🔄 **Step-based processing architecture** — Modular and extensible design
- 💾 **Automatic caching** — Speed up iterations with intelligent caching
- 🔁 **Parallel batch processing** — Concurrent row execution via `ThreadPoolExecutor`
- 🛠️ **Easy to extend** — Create custom steps with simple API
- ⚖️ **LLM-as-a-judge validation** — Quality control with judge models
- 🔄 **Dual generator comparison** — Compare two models and pick the best
- 🔧 **JSON repair** — Automatic recovery from malformed LLM outputs

---

## 🏗️ Architecture

```
                  ┌─────────────────────────────────────┐
                  │         DataForgePipeline            │
                  │  (Orquestador de pasos secuenciales) │
                  └─────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌──────────────────┐     ┌──────────────┐
│  transformers/ │     │     llm/         │     │  validators/ │
│  (carga y      │     │  (generación y   │     │  (reglas de  │
│   transforma)  │     │   evaluación)    │     │   negocio)   │
├───────────────┤     ├──────────────────┤     ├──────────────┤
│ LoadDataFrame │     │ OllamaLLMStep    │     │ValidateUser  │
│ AddColumn     │     │ OllamaJudgeStep  │     │Stories       │
│ KeepColumns   │     │ ComparisonJudge  │     └──────────────┘
│ ExplodeTasks  │     └──────────────────┘
│ json_repair   │
└───────────────┘
```

---

## 📚 Examples

### Chaining Steps with `>>`

Puedes encadenar pasos de dos formas equivalentes. La API explícita:

```python
pipeline = DataForgePipeline(name="example")
pipeline.add_step(LoadDataFrame(name="load", df=df))
pipeline.add_step(OllamaLLMStep(name="generate", ...))
```

O con el operador `>>` (syntactic sugar para `add_step()`):

```python
pipeline = DataForgePipeline(name="example")

(pipeline
    >> LoadDataFrame(name="load", df=df)
    >> OllamaLLMStep(name="generate", ...)
)
```

Ambos son equivalentes. El operador `>>` devuelve el pipeline, permitiendo encadenamiento fluido.

### Basic Pipeline

```python
import pandas as pd
from dataforge import DataForgePipeline
from dataforge.transformers import LoadDataFrame
from dataforge.llm import OllamaLLMStep

# Create data
df = pd.DataFrame({
    'input': ['As a user, I want to login so that I can access my account']
})

# Create pipeline
pipeline = DataForgePipeline(name="basic-example")

# Add steps
pipeline.add_step(LoadDataFrame(name="load", df=df))
pipeline.add_step(OllamaLLMStep(
    name="generate",
    model_name="llama3.2",
    prompt_column="input",
    output_column="tasks",
    batch_size=8,
    num_workers=4         # requests concurrentes dentro del batch
))

# Run
result = pipeline.run()
print(result)
```

### Pipeline with Validation

```python
from dataforge import DataForgePipeline
from dataforge.transformers import LoadDataFrame, AddColumn, ExplodeTasks
from dataforge.llm import OllamaLLMStep, OllamaJudgeStep
from dataforge.validators import ValidateUserStories

pipeline = DataForgePipeline(name="salony-pipeline")

(pipeline
    >> LoadDataFrame(name="load", df=stories_df)
    >> ValidateUserStories(name="validate", story_column="input")
    >> AddColumn(name="add_id", input_columns=[], output_column="us_id", func=lambda: counter())
    >> OllamaLLMStep(name="generate", model_name="llama3.2", ...)
    >> OllamaJudgeStep(name="judge", model_name="llama3.1:8b", ...)
    >> ExplodeTasks(name="explode", tasks_column="tasks", output_column="task")
)

result = pipeline.run(use_cache=True)
```

---

## 🏗️ Use Cases Reales

El directorio `src/dataforge/use_cases/` contiene pipelines completos y listos para correr sobre datasets reales.

### Salony Dataset

Pipelines para descomponer historias de usuario del dataset [Salony](data/raw/salony_train.csv) en tareas de desarrollo.

| Script | Descripción |
|--------|-------------|
| **`salony_single_generator_pipeline.py`** | Pipeline completo con un solo generador + judge opcional |
| **`salony_dual_generator_pipeline.py`** | Pipeline con dos generadores + ComparisonJudge para seleccionar el mejor |

#### Single Generator

```bash
python src/dataforge/use_cases/salony/scripts/salony_single_generator_pipeline.py output.csv \
  --model llama3.1:8b \
  --batch-size 4 \
  --use-judge
```

#### Dual Generator

```bash
python src/dataforge/use_cases/salony/scripts/salony_dual_generator_pipeline.py output.csv \
  --model-a llama3.1:8b \
  --model-b qwen3:8b \
  --judge-model llama3.1:8b
```

#### Análisis de Resultados

| Script | Descripción |
|--------|-------------|
| **`aggregate_metrics.py`** | Métricas agregadas: media, std, pass rate, win rate (con output LaTeX) |
| **`criterion_breakdown.py`** | Desglose de puntuaciones por criterio (coherencia, completitud, etc.) |
| **`plots.py`** | Visualizaciones: boxplots, comparación de scores entre generadores |
| **`consolidate_results.py`** | Consolidación de múltiples tests en reporte unificado con gráficas |
| **`fix_total_score_no_stage.py`** | Utilidad para recalcular totales si el judge no los devolvió |

---

## 🔧 Available Steps

### Data Loading & Transformation (`dataforge.transformers`)

| Step | Description |
|------|-------------|
| **LoadDataFrame** | Load data from DataFrame or CSV |
| **AddColumn** | Add computed column with custom function |
| **KeepColumns** | Select specific columns |
| **ExplodeTasks** | Split concatenated numbered tasks into rows |

### LLM Generation & Evaluation (`dataforge.llm`)

| Step | Description |
|------|-------------|
| **OllamaLLMStep** | Generate text with Ollama models (with retry logic and exponential backoff) |
| **OllamaJudgeStep** | Validate generated content using LLM-as-a-judge (5 criteria, 0-50 scale) |
| **ComparisonJudgeStep** | Compare outputs from dual generators, score both, select the best |

### Business Validation (`dataforge.validators`)

| Step | Description |
|------|-------------|
| **ValidateUserStories** | Filter user stories matching Agile format ("As a... I want... so that...") |

### Utilities (`dataforge.utils`)

| Module | Description |
|--------|-------------|
| **CacheManager** | Automatic pipeline step caching with content-hash keys |
| **setup_logger** | Configure consistent logging across steps (timestamped, named loggers) |
| **batch_dataframe** | Split DataFrames into batches for memory-efficient processing |
| **split_dataframe** | Split DataFrames for parallel processing |
| **get_num_batches** | Calculate number of batches given a batch size |

### JSON Repair (`dataforge.transformers.json_repair`)

| Function | Description |
|----------|-------------|
| **clean_json_response** | Extract JSON from LLM responses (removes markdown fences, finds boundaries) |
| **repair_json** | Fix common JSON errors: escape newlines in strings, add missing commas, remove trailing commas |
| **parse_json_with_repair** | Parse with automatic fallback: try direct parse, then repair and retry |

---

## ⚖️ LLM-as-a-Judge: Validation Criteria

`OllamaJudgeStep` evalúa cada tarea generada contra 5 criterios. Cada criterio se puntúa de 0 a 10, para un total de 0 a 50 puntos.

| Criterio | Rango | ¿Qué mide? |
|----------|-------|------------|
| **Coherencia** | 0-10 | ¿Las tareas están directamente relacionadas con la historia? ¿Hay tareas irrelevantes? |
| **Completitud** | 0-10 | ¿Cubren todos los aspectos necesarios? ¿Falta algo crítico? |
| **Viabilidad** | 0-10 | ¿Son técnicamente realizables? ¿Hay pasos imposibles o ilógicos? |
| **Formato** | 0-10 | ¿Cada tarea tiene título claro, descripción y está bien estructurada? |
| **Granularidad** | 0-10 | ¿El nivel de detalle es apropiado? ¿Muy amplias o demasiado atómicas? |

**Output columns** generadas por el judge:

| Columna | Descripción |
|---------|-------------|
| `validacion_coherencia` | Score 0-10 |
| `validacion_completitud` | Score 0-10 |
| `validacion_viabilidad` | Score 0-10 |
| `validacion_formato` | Score 0-10 |
| `validacion_granularidad` | Score 0-10 |
| `validacion_total` | Suma total (0-50) |
| `validacion_aprobado` | Boolean (`True` si total >= threshold) |
| `validacion_problemas` | Lista de problemas críticos detectados |
| `validacion_recomendaciones` | Sugerencias de mejora |

```python
judge = OllamaJudgeStep(
    name="validate_tasks",
    model_name="llama3.1:8b",
    historia_usuario_column="input",
    tareas_generadas_column="tasks",
    approval_threshold=35.0,  # Mínimo 35/50 para aprobar
    batch_size=2,
    generation_kwargs={"temperature": 0.2},  # Baja temperatura = juicio consistente
)
```

---

## 🔧 JSON Repair: Manejo Robusto de LLM Outputs

Los LLMs frecuentemente devuelven JSON malformado. El módulo `json_repair` resuelve los problemas más comunes automáticamente.

### Problemas que Repara

| Problema | Ejemplo | Solución |
|----------|---------|----------|
| Raw newlines en strings | `"desc": "line 1\nline 2"` | Escapa a `\n` |
| Faltan comas entre campos | `} "key"` | Agrega `,` |
| Trailing commas | `},]` | Remueve la coma extra |
| Markdown fences | `` ```json `` | Limpieza automática |
| Caracteres de control | tabs, null bytes | Escape o remoción |

### Uso

```python
from dataforge.transformers.json_repair import repair_json, parse_json_with_repair

# Limpiar y extraer JSON de una respuesta
cleaned = clean_json_response(llm_response)

# Reparar errores comunes
repaired = repair_json(cleaned)

# Parse con fallback automático (intenta directo, luego repara)
result = parse_json_with_repair(llm_response)
if result is None:
    print("No se pudo parsear ni reparando")
```

---

## ⚡ Performance Tuning

### `batch_size` + `num_workers`: Cómo Afinar el Throughput

Cada LLM step acepta dos parámetros que controlan el rendimiento:

| Parámetro | Default (LLM / Judge) | Qué Controla |
|-----------|----------------------|--------------|
| `batch_size` | 8 / 4 | Filas cargadas por lote en memoria. También limita la concurrencia máxima. |
| `num_workers` | 1 | Requests concurrentes a Ollama **dentro** de cada batch. |

**Regla de oro:** `batch_size >= num_workers`. Si `num_workers > batch_size`, sobran workers que nunca se usan.

**¿`batch_size` es placebo?** No, pero su rol cambió con `num_workers`:
- **Sin paralelismo** (`num_workers=1`): controla memoria y granularidad de caché
- **Con paralelismo** (`num_workers > 1`): actúa como tope de concurrencia

### Guía por Hardware

Valores de partida recomendados para **modelos 7B-8B** (llama3.1:8b, qwen3:8b, deepseek-r1:8b):

| Hardware | `num_workers` | `batch_size` | Notas |
|----------|:------------:|:------------:|-------|
| **Solo CPU** (8GB RAM) | 1 | 4-8 | CPU encola requests secuencialmente. Paralelismo no da ganancia real. |
| **Solo CPU** (16GB+ RAM) | 1-2 | 4-8 | Con 2 workers puede ayudar si hay múltiples cores físicos. |
| **GPU 4GB VRAM** | 1 | 2-4 | Sin espacio para más de 1 modelo en VRAM. Batches chicos. |
| **GPU 6GB VRAM** | 2 | 4 | Point inicial para modelos 7B con cuantización Q4. |
| **GPU 8GB VRAM** | 2-3 | 4-6 | Ollama puede mantener 1-2 requests simultáneos en GPU. |
| **GPU 12GB VRAM** | 4 | 8 | **Balance ideal** para 7B-8B. GPU aprovechada sin saturar. |
| **GPU 16GB VRAM** | 4-6 | 8 | Podés usar modelos 13B y mantener paralelismo. |
| **GPU 24GB+ VRAM** | 6-8 | 8-12 | Múltiples requests caben en VRAM. Probá de a poco. |

### Recomendaciones por Tamaño de Modelo

| Modelo | Ejemplos | `num_workers` | Notas |
|--------|----------|:------------:|-------|
| **1B-3B** | `llama3.2:3b`, `qwen3:4b`, `phi4-mini` | 4-8 | Modelos chicos, mucha concurrencia posible. No saturan VRAM. |
| **7B-8B** | `llama3.1:8b`, `qwen3:8b`, `deepseek-r1:8b` | 2-4 | El punto dulce. Buena calidad sin consumir mucha VRAM. |
| **13B-14B** | `llama3.1:14b`, `qwen3:14b` | 1-3 | Cada request ocupa más VRAM. Reducí workers respecto a 7B. |
| **30B+** | `qwen3:30b`, `llama3.1:70b` | 1-2 | Modelos grandes. Necesitás 24GB+ VRAM. Paralelismo mínimo. |

### Cómo Ajustar

```python
# GPU 12GB VRAM + modelo 7B-8B → punto dulce
generator = OllamaLLMStep(
    name="generate",
    model_name="qwen3:8b",
    prompt_column="prompt",
    output_column="tasks",
    batch_size=8,
    num_workers=4,
)

# GPU 8GB VRAM + modelo 13B → conservador
judge = OllamaJudgeStep(
    name="validate",
    model_name="llama3.1:14b",
    historia_usuario_column="input",
    tareas_generadas_column="tasks",
    batch_size=4,
    num_workers=2,
)
```

### Ajuste Fino con `ollama ps`

```bash
# Mientras corre el pipeline, monitoreá:
ollama ps

# Si ves cola de requests sin procesar → subí num_workers
# Si ves errores CUDA OOM → bajá num_workers o batch_size
# Si el output es lento pero GPU al 100% → num_workers suficiente
```

**Estrategia:** Empezá con `num_workers=2` y subí de a 1 mientras monitoreás con `ollama ps`. Cuando veas errores de memoria o latencia que no mejora, ese es tu límite.

---

## 🎨 Creating Custom Steps

```python
from dataforge.base_step import BaseStep
import pandas as pd


class MyCustomStep(BaseStep):
    def __init__(self, name: str, param: str, **kwargs):
        super().__init__(name, **kwargs)
        self.param = param

    @property
    def inputs(self):
        return ["input_column"]

    @property
    def outputs(self):
        return ["output_column"]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["output_column"] = df["input_column"].apply(
            lambda x: self.transform(x)
        )
        return df

    def transform(self, value):
        return value.upper()
```

---

## 📊 Caching

Caching is automatic and based on step configuration + input data hash:

```python
# Use cache (default)
result = pipeline.run(use_cache=True)

# Skip cache
result = pipeline.run(use_cache=False)

# Clear cache
pipeline.clear_cache()
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=dataforge
```

---

## 📁 Project Structure

```
LocalLLM-DataForge/
├── data/                    # Input datasets and pipeline outputs
│   ├── raw/                 #   CSV inputs (user stories, etc.)
│   └── outputs/             #   Pipeline results and benchmarks
├── examples/                # Usage demonstrations
│   └── demo_pipeline.py
├── scripts/                 # CLI utilities
│   ├── clear_cache.py
│   └── run_pipeline.py
├── src/                     # 🌟 Source code
│   └── dataforge/           #   Main package
│       ├── pipeline.py      #   Pipeline orchestrator
│       ├── base_step.py     #   Abstract step base class
│       ├── llm/             #   🦙 LLM generation and validation steps
│       ├── transformers/    #   🔄 Data transformation steps
│       ├── validators/      #   ✅ Business rule validators
│       ├── use_cases/       #   🏗️ Real-world pipeline implementations
│       │   └── salony/      #       Salony dataset pipelines & analysis
│       └── utils/           #   🛠️ Shared utilities (cache, logging, batching)
├── tests/                   # Automated tests
├── docs/                    # Documentation
│   ├── USER_STORY_TO_TASKS.md
│   └── springer/            # Academic paper and research tests
├── README.md
└── AGENTS.md                # AI agent guide
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your code and tests
4. Update documentation
5. Submit a pull request

**¿Por dónde empezar?** Revisá los pipelines en `src/dataforge/use_cases/` para ver ejemplos reales de cómo se arman pipelines completos.

---

## 📄 License

MIT License

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/alvaldes/LocalLLM-DataForge/issues)
- **Documentation**: See `examples/`, `docs/`, and `AGENTS.md`
- **Pipelines reales**: Explorá `src/dataforge/use_cases/salony/scripts/`
