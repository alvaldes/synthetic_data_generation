# simple_pipeline/steps/ollama_judge_step.py

import ollama
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import time
import json
import re
from ..utils.logging import setup_logger
import logging

from ..base_step import BaseStep
from ..utils.batching import batch_dataframe, get_num_batches


class OllamaJudgeStep(BaseStep):
    """
    Step que valida tareas generadas usando un LLM como juez.

    Evalúa las tareas generadas según criterios de:
    - Coherencia: Relación con la historia de usuario original
    - Completitud: Cobertura de todos los aspectos necesarios
    - Viabilidad: Factibilidad técnica de las tareas
    - Formato: Estructura correcta de cada tarea
    - Granularidad: Nivel de detalle apropiado

    Retorna puntuaciones estructuradas y estado de aprobación.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        historia_usuario_column: str,
        tareas_generadas_column: str,
        approval_threshold: float = 35.0,
        batch_size: int = 4,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        ollama_host: str = "http://localhost:11434",
        max_retries: int = 3,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.model_name = model_name
        self.historia_usuario_column = historia_usuario_column
        self.tareas_generadas_column = tareas_generadas_column
        self.approval_threshold = approval_threshold
        self.batch_size = batch_size
        self.generation_kwargs = generation_kwargs or {"temperature": 0.2}
        self.ollama_host = ollama_host
        self.max_retries = max_retries
        self.client = None
        self.logger = setup_logger(
            name=f"OllamaJudgeStep.{name}",
            level=logging.INFO
        )

    # -------- Propiedades requeridas --------
    @property
    def inputs(self) -> List[str]:
        return [self.historia_usuario_column, self.tareas_generadas_column]

    @property
    def outputs(self) -> List[str]:
        return [
            "validacion_coherencia",
            "validacion_completitud",
            "validacion_viabilidad",
            "validacion_formato",
            "validacion_granularidad",
            "validacion_total",
            "validacion_aprobado",
            "validacion_problemas",
            "validacion_recomendaciones"
        ]

    # -------- Ciclo de vida --------
    def load(self) -> None:
        """Inicializa el cliente de Ollama."""
        self.client = ollama.Client(host=self.ollama_host)
        super().load()

    def unload(self) -> None:
        """Libera recursos del cliente."""
        self.client = None
        super().unload()

    # -------- Utilidades internas --------
    def _create_judge_prompt(self, historia_usuario: str, tareas_generadas: str) -> str:
        """Crea el prompt optimizado para el LLM juez."""
        prompt = f"""Eres un experto QA que valida si una lista de tareas derivadas de una historia de usuario es correcta y completa.

**HISTORIA DE USUARIO:**
{historia_usuario}

**TAREAS GENERADAS:**
{tareas_generadas}

**INSTRUCCIONES DE VALIDACIÓN:**

Evalúa las tareas generadas según estos criterios:

1. **COHERENCIA (0-10):** ¿Todas las tareas están relacionadas directamente con la historia de usuario? ¿Hay tareas irrelevantes o fuera de alcance?

2. **COMPLETITUD (0-10):** ¿Las tareas cubren todos los aspectos necesarios para completar la historia? ¿Falta algo crítico?

3. **VIABILIDAD (0-10):** ¿Son las tareas técnicamente realizables? ¿Hay pasos imposibles o ilógicos?

4. **FORMATO (0-10):** ¿Cada tarea tiene: título claro, descripción, criterios de aceptación? ¿Está bien estructurada?

5. **GRANULARIDAD (0-10):** ¿El nivel de detalle es apropiado? ¿Las tareas son muy amplias o demasiado atómicas?

**REGLAS CRÍTICAS:**
- Si la puntuación total es < {self.approval_threshold}/50, marca aprobado=false
- Si falta algún elemento crítico del formato, marca aprobado=false
- Si hay tareas completamente irrelevantes, marca aprobado=false
- Sé estricto pero justo: no rechaces por detalles menores

RESPONDE ÚNICAMENTE CON ESTE JSON VÁLIDO (sin markdown, sin explicaciones):

{{
  "coherencia": {{
    "puntuacion": 0-10,
    "justificacion": "texto"
  }},
  "completitud": {{
    "puntuacion": 0-10,
    "justificacion": "texto",
    "tareas_faltantes": []
  }},
  "viabilidad": {{
    "puntuacion": 0-10,
    "justificacion": "texto"
  }},
  "formato": {{
    "puntuacion": 0-10,
    "justificacion": "texto"
  }},
  "granularidad": {{
    "puntuacion": 0-10,
    "justificacion": "texto"
  }},
  "puntuacion_total": 0-50,
  "aprobado": true/false,
  "problemas_criticos": [],
  "recomendaciones": []
}}"""
        return prompt

    def _clean_json_response(self, response: str) -> str:
        """Limpia la respuesta para extraer JSON válido."""
        # Remover markdown si existe
        response = response.replace("```json", "").replace("```", "").strip()

        # Buscar JSON usando regex
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            return match.group(0)

        return response

    def _parse_validation_result(self, raw_response: str) -> Dict[str, Any]:
        """Parsea la respuesta JSON del juez con manejo robusto de errores."""
        try:
            cleaned_response = self._clean_json_response(raw_response)
            result = json.loads(cleaned_response)

            # Validar estructura mínima requerida
            required_fields = ['coherencia', 'completitud', 'viabilidad', 'formato', 'granularidad']
            for field in required_fields:
                if field not in result or 'puntuacion' not in result[field]:
                    raise ValueError(f"Campo requerido faltante: {field}")

            # Calcular puntuación total si no está presente
            if 'puntuacion_total' not in result:
                result['puntuacion_total'] = sum(
                    result[field]['puntuacion'] for field in required_fields
                )

            # Determinar aprobación si no está presente
            if 'aprobado' not in result:
                result['aprobado'] = result['puntuacion_total'] >= self.approval_threshold

            # Asegurar campos opcionales
            result.setdefault('problemas_criticos', [])
            result.setdefault('recomendaciones', [])

            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"Error parseando validación: {e}")
            self.logger.error(f"Respuesta cruda: {raw_response}")

            # Retornar resultado de fallo
            return {
                'coherencia': {'puntuacion': 0, 'justificacion': 'Error de parsing'},
                'completitud': {'puntuacion': 0, 'justificacion': 'Error de parsing', 'tareas_faltantes': []},
                'viabilidad': {'puntuacion': 0, 'justificacion': 'Error de parsing'},
                'formato': {'puntuacion': 0, 'justificacion': 'Error de parsing'},
                'granularidad': {'puntuacion': 0, 'justificacion': 'Error de parsing'},
                'puntuacion_total': 0,
                'aprobado': False,
                'problemas_criticos': ['Error en parsing de respuesta del juez'],
                'recomendaciones': ['Revisar manualmente']
            }

    def _validate_with_retry(
        self,
        historia_usuario: str,
        tareas_generadas: str,
        retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Llama al modelo juez con reintentos en caso de error."""
        try:
            prompt = self._create_judge_prompt(historia_usuario, tareas_generadas)

            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                format="json",
                options=self.generation_kwargs
            )

            raw_content = response['message']['content']
            validation_result = self._parse_validation_result(raw_content)
            return validation_result

        except ollama.ResponseError as e:
            self.logger.warning(f"Ollama API error during validation: {e.error}")
            if e.status_code == 404:
                self.logger.error(f"Judge model {self.model_name} not found. Try: ollama pull {self.model_name}")
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count  # exponential backoff
                time.sleep(wait_time)
                return self._validate_with_retry(historia_usuario, tareas_generadas, retry_count + 1)
            else:
                self.logger.error(f"Judge validation failed after {self.max_retries} retries")
                return None
        except ConnectionError as e:
            self.logger.error(f"Connection error during validation: {e}")
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                time.sleep(wait_time)
                return self._validate_with_retry(historia_usuario, tareas_generadas, retry_count + 1)
            else:
                self.logger.error("Ollama server appears to be down. Check with: ollama serve")
                return None
        except Exception as e:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                time.sleep(wait_time)
                return self._validate_with_retry(historia_usuario, tareas_generadas, retry_count + 1)
            else:
                self.logger.error(f"Unexpected error in judge validation after {self.max_retries} retries: {e}")
                return None

    def _process_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Procesa un batch de validaciones con Ollama."""
        results = []

        for _, row in batch_df.iterrows():
            historia = str(row[self.historia_usuario_column])
            tareas = str(row[self.tareas_generadas_column])

            validation = self._validate_with_retry(historia, tareas)

            if validation is None:
                # Usar valores por defecto en caso de error total
                validation = {
                    'coherencia': {'puntuacion': 0, 'justificacion': 'Error de conexión'},
                    'completitud': {'puntuacion': 0, 'justificacion': 'Error de conexión', 'tareas_faltantes': []},
                    'viabilidad': {'puntuacion': 0, 'justificacion': 'Error de conexión'},
                    'formato': {'puntuacion': 0, 'justificacion': 'Error de conexión'},
                    'granularidad': {'puntuacion': 0, 'justificacion': 'Error de conexión'},
                    'puntuacion_total': 0,
                    'aprobado': False,
                    'problemas_criticos': ['Error de conexión con modelo'],
                    'recomendaciones': ['Reintentar validación']
                }

            self.logger.info(f"Validación para fila {row.name}: aprobado={validation['aprobado']}, total={validation['puntuacion_total']}")
            results.append(validation)

        # Crear DataFrame resultado con todas las columnas de validación
        result_df = batch_df.copy()

        # Agregar columnas de validación
        result_df["validacion_coherencia"] = [r['coherencia']['puntuacion'] for r in results]
        result_df["validacion_completitud"] = [r['completitud']['puntuacion'] for r in results]
        result_df["validacion_viabilidad"] = [r['viabilidad']['puntuacion'] for r in results]
        result_df["validacion_formato"] = [r['formato']['puntuacion'] for r in results]
        result_df["validacion_granularidad"] = [r['granularidad']['puntuacion'] for r in results]
        result_df["validacion_total"] = [r['puntuacion_total'] for r in results]
        result_df["validacion_aprobado"] = [r['aprobado'] for r in results]
        result_df["validacion_problemas"] = [str(r['problemas_criticos']) for r in results]
        result_df["validacion_recomendaciones"] = [str(r['recomendaciones']) for r in results]

        return result_df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa el DataFrame en batches para validación."""
        results = []

        # Verificar que las columnas existen y tienen datos válidos
        for col in [self.historia_usuario_column, self.tareas_generadas_column]:
            if col not in df.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en DataFrame")

        # Filtrar filas con datos faltantes
        valid_rows = df.dropna(subset=[self.historia_usuario_column, self.tareas_generadas_column])
        if len(valid_rows) < len(df):
            self.logger.warning(f"Omitiendo {len(df) - len(valid_rows)} filas con datos faltantes")

        # Procesar en batches
        with tqdm(total=len(valid_rows), desc=f"Validating {self.name}") as pbar:
            for batch in batch_dataframe(valid_rows, self.batch_size):
                processed_batch = self._process_batch(batch)
                results.append(processed_batch)
                pbar.update(len(batch))

        return pd.concat(results, ignore_index=False)