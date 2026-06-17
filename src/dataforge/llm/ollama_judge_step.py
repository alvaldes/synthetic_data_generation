# dataforge/steps/ollama_judge_step.py

import ollama
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..base_step import BaseStep
from ..config import get_settings, PromptLoader
from ..utils.batching import batch_dataframe
from ..utils.logging import setup_logger
from ..transformers.json_repair import (
    parse_json_with_repair,
    fuzzy_normalize_dict,
)

_thread_local = threading.local()


class OllamaJudgeStep(BaseStep):
    """
    Step that validates generated tasks using an LLM as judge.

    Evaluates generated tasks according to criteria:
    - Coherence: Relationship with original user story
    - Completeness: Coverage of all necessary aspects
    - Feasibility: Technical feasibility of tasks
    - Format: Correct structure of each task
    - Granularity: Appropriate level of detail

    The judge prompt is loaded from :file:`config/prompts/single_judge.j2`.
    Criteria, thresholds, and column names come from the centralised
    :ref:`config <DataForgeSettings>`.
    """

    PROMPT_TEMPLATE = "single_judge.j2"

    def __init__(
        self,
        name: str,
        model_name: Optional[str] = None,
        historia_usuario_column: Optional[str] = None,
        tareas_generadas_column: Optional[str] = None,
        approval_threshold: Optional[float] = None,
        batch_size: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        ollama_host: Optional[str] = None,
        max_retries: Optional[int] = None,
        num_workers: Optional[int] = None,
        max_zero_retries: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        cfg = get_settings()
        llm_cfg = cfg.llm
        judge_cfg = cfg.judge

        self.model_name = model_name or judge_cfg.model or llm_cfg.default_model
        self.historia_usuario_column = historia_usuario_column
        self.tareas_generadas_column = tareas_generadas_column
        self.approval_threshold = (
            approval_threshold if approval_threshold is not None
            else judge_cfg.approval_threshold
        )
        self.batch_size = batch_size if batch_size is not None else judge_cfg.batch_size
        self.generation_kwargs = generation_kwargs or {
            "temperature": judge_cfg.temperature,
            "num_predict": judge_cfg.num_predict,
        }
        self.ollama_host = ollama_host or llm_cfg.ollama_host
        self.max_retries = max_retries if max_retries is not None else llm_cfg.max_retries
        self.num_workers = num_workers if num_workers is not None else judge_cfg.num_workers
        self.max_zero_retries = max_zero_retries if max_zero_retries is not None else judge_cfg.max_zero_retries
        self.client = None
        self.logger = setup_logger(name=f"OllamaJudgeStep.{name}", level=logging.INFO)

    # -------- Propiedades requeridas --------
    @property
    def inputs(self) -> List[str]:
        return [self.historia_usuario_column, self.tareas_generadas_column]

    @property
    def outputs(self) -> List[str]:
        cfg = get_settings()
        p = cfg.judge.column_prefix
        return [
            f"{p}coherencia",
            f"{p}completitud",
            f"{p}viabilidad",
            f"{p}formato",
            f"{p}granularidad",
            f"{p}total",
            f"{p}aprobado",
            f"{p}problemas",
            f"{p}recomendaciones",
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
    def _create_judge_prompt(self, historia_usuario: str, tareas_generadas: str, zero_retry_count: int = 0) -> str:
        """Render the judge prompt template with row data."""
        prompt = PromptLoader.render(
            self.PROMPT_TEMPLATE,
            {
                "historia_usuario": historia_usuario,
                "tareas_generadas": tareas_generadas,
                "approval_threshold": str(self.approval_threshold),
            },
        )

        if zero_retry_count > 0:
            prompt += (
                f"\n\n**⚠️ REINTENTO {zero_retry_count} — CRÍTICO:**\n"
                "En tu intento anterior diste puntuación 0 en al menos un criterio.\n"
                "CADA criterio debe tener un valor entre 1 y 10. 0 NO está permitido.\n"
                "Vuelve a evaluar con cuidado, asegurándote de dar valores entre 1 y 10.\n"
                "Si un criterio realmente no aplica, usa 1 como mínimo.\n"
            )

        return prompt

    def _has_zero_score(self, result: Dict[str, Any]) -> bool:
        """Check if any individual criterion has a score of 0."""
        required_fields = ["coherence", "completeness", "feasibility", "format", "granularity"]
        for field in required_fields:
            if field in result and result[field].get("puntuacion", 0) == 0:
                return True
        return False

    def _parse_validation_result(self, raw_response: str) -> Dict[str, Any]:
        """Parsea la respuesta JSON del juez con manejo robusto de errores."""
        required_fields = [
            "coherence",
            "completeness",
            "feasibility",
            "format",
            "granularity",
        ]

        result = parse_json_with_repair(raw_response, logger=self.logger)

        if result is not None:
            try:
                # Fuzzy-normalize criterion keys for typo tolerance
                result = fuzzy_normalize_dict(result, required_fields)

                for field in required_fields:
                    if field not in result or "puntuacion" not in result[field]:
                        raise ValueError(f"Campo requerido faltante: {field}")

                # Calculate total in code — always override LLM's value
                code_total = sum(
                    result[field]["puntuacion"] for field in required_fields
                )

                if "puntuacion_total" in result and result["puntuacion_total"] != code_total:
                    self.logger.warning(
                        f"Discrepancia en puntuacion_total: LLM={result['puntuacion_total']}, "
                        f"código={code_total}. Usando valor del código."
                    )

                result["puntuacion_total"] = code_total

                if "aprobado" not in result:
                    result["aprobado"] = (
                        result["puntuacion_total"] >= self.approval_threshold
                    )

                result.setdefault("problemas_criticos", [])
                result.setdefault("recomendaciones", [])

                return result

            except (ValueError, KeyError) as e:
                self.logger.warning(f"Validación del resultado del juez falló: {e}")

        self.logger.error(f"Error parseando validación. Respuesta cruda: {raw_response}")

        return {
            "coherence": {"puntuacion": -1, "justificacion": "Error de parsing"},
            "completeness": {
                "puntuacion": -1,
                "justificacion": "Error de parsing",
                "tareas_faltantes": [],
            },
            "feasibility": {"puntuacion": -1, "justificacion": "Error de parsing"},
            "format": {"puntuacion": -1, "justificacion": "Error de parsing"},
            "granularity": {"puntuacion": -1, "justificacion": "Error de parsing"},
            "puntuacion_total": -5,
            "aprobado": False,
            "problemas_criticos": ["Error en parsing de respuesta del juez"],
            "recomendaciones": ["Revisar manualmente"],
            "parse_error": True,
        }

    def _get_thread_client(self) -> ollama.Client:
        """Get thread-local Ollama client for parallel execution."""
        if not hasattr(_thread_local, "client"):
            _thread_local.client = ollama.Client(host=self.ollama_host)
        return _thread_local.client

    def _validate_with_retry(
        self, historia_usuario: str, tareas_generadas: str, retry_count: int = 0,
        zero_retry_count: int = 0, client: Optional[ollama.Client] = None,
    ) -> Optional[Dict[str, Any]]:
        """Llama al modelo juez con reintentos en caso de error o scores cero.

        Args:
            historia_usuario: User story text.
            tareas_generadas: Generated tasks text.
            retry_count: Current retry attempt number for connection/API errors.
            zero_retry_count: Current retry attempt number for zero-score retries.
            client: Optional Ollama client instance. Falls back to self.client if None.
        """
        client = client or self.client
        try:
            prompt = self._create_judge_prompt(
                historia_usuario, tareas_generadas, zero_retry_count=zero_retry_count,
            )

            response = client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                format="json",
                options=self.generation_kwargs,
            )

            raw_content = response["message"]["content"]
            validation_result = self._parse_validation_result(raw_content)

            # Zero-score retry: if any criterion is 0, retry with stricter prompt
            if (
                "parse_error" not in validation_result
                and zero_retry_count < self.max_zero_retries
                and self._has_zero_score(validation_result)
            ):
                self.logger.warning(
                    f"Detectado score 0 en validación (reintento {zero_retry_count + 1}/"
                    f"{self.max_zero_retries}). Reintentando con prompt corregido."
                )
                return self._validate_with_retry(
                    historia_usuario, tareas_generadas,
                    retry_count=0,
                    zero_retry_count=zero_retry_count + 1,
                    client=client,
                )

            return validation_result

        except ollama.ResponseError as e:
            self.logger.warning(f"Ollama API error during validation: {e.error}")
            if e.status_code == 404:
                self.logger.error(
                    f"Judge model {self.model_name} not found. Try: ollama pull {self.model_name}"
                )
            if retry_count < self.max_retries:
                wait_time = 2**retry_count
                time.sleep(wait_time)
                return self._validate_with_retry(
                    historia_usuario, tareas_generadas, retry_count + 1, zero_retry_count, client
                )
            else:
                self.logger.error(f"Judge validation failed after {self.max_retries} retries")
                return None
        except ConnectionError as e:
            self.logger.error(f"Connection error during validation: {e}")
            if retry_count < self.max_retries:
                wait_time = 2**retry_count
                time.sleep(wait_time)
                return self._validate_with_retry(
                    historia_usuario, tareas_generadas, retry_count + 1, zero_retry_count, client
                )
            else:
                self.logger.error("Ollama server appears to be down. Check with: ollama serve")
                return None
        except Exception as e:
            if retry_count < self.max_retries:
                wait_time = 2**retry_count
                time.sleep(wait_time)
                return self._validate_with_retry(
                    historia_usuario, tareas_generadas, retry_count + 1, zero_retry_count, client
                )
            else:
                self.logger.error(
                    f"Unexpected error in judge validation after {self.max_retries} retries: {e}"
                )
                return None

    def _validate_with_retry_parallel(
        self, historia_usuario: str, tareas_generadas: str
    ) -> Optional[Dict[str, Any]]:
        """Wrapper for parallel execution: uses thread-local client."""
        client = self._get_thread_client()
        return self._validate_with_retry(historia_usuario, tareas_generadas, client=client)

    def _process_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Procesa un batch de validaciones con Ollama."""
        results = []

        if self.num_workers <= 1:
            # Sequential path — uses self.client
            for _, row in batch_df.iterrows():
                historia = str(row[self.historia_usuario_column])
                tareas = str(row[self.tareas_generadas_column])

                validation = self._validate_with_retry(historia, tareas)

                if validation is None:
                    validation = {
                        "coherence": {"puntuacion": -1, "justificacion": "Error de conexión"},
                        "completeness": {
                            "puntuacion": -1,
                            "justificacion": "Error de conexión",
                            "tareas_faltantes": [],
                        },
                        "feasibility": {"puntuacion": -1, "justificacion": "Error de conexión"},
                        "format": {"puntuacion": -1, "justificacion": "Error de conexión"},
                        "granularity": {"puntuacion": -1, "justificacion": "Error de conexión"},
                        "puntuacion_total": -5,
                        "aprobado": False,
                        "problemas_criticos": ["Error de conexión con modelo"],
                        "recomendaciones": ["Reintentar validación"],
                        "parse_error": True,
                    }

                self.logger.info(
                    f"Validación para fila {row.name}: aprobado={validation['aprobado']}, "
                    f"total={validation['puntuacion_total']}"
                )
                results.append(validation)
        else:
            # Parallel path — thread-local clients via ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                future_to_idx: Dict[Any, int] = {}
                for idx, (_, row) in enumerate(batch_df.iterrows()):
                    historia = str(row[self.historia_usuario_column])
                    tareas = str(row[self.tareas_generadas_column])
                    future = pool.submit(
                        self._validate_with_retry_parallel, historia, tareas
                    )
                    future_to_idx[future] = idx

                ordered: List[Optional[Dict[str, Any]]] = [None] * len(batch_df)

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    validation = future.result()

                    if validation is None:
                        validation = {
                            "coherence": {"puntuacion": -1, "justificacion": "Error de conexión"},
                            "completeness": {
                                "puntuacion": -1,
                                "justificacion": "Error de conexión",
                                "tareas_faltantes": [],
                            },
                            "feasibility": {"puntuacion": -1, "justificacion": "Error de conexión"},
                            "format": {"puntuacion": -1, "justificacion": "Error de conexión"},
                            "granularity": {"puntuacion": -1, "justificacion": "Error de conexión"},
                            "puntuacion_total": -5,
                            "aprobado": False,
                            "problemas_criticos": ["Error de conexión con modelo"],
                            "recomendaciones": ["Reintentar validación"],
                            "parse_error": True,
                        }

                    self.logger.info(
                        f"Validación para fila {idx}: aprobado={validation['aprobado']}, "
                        f"total={validation['puntuacion_total']}"
                    )
                    ordered[idx] = validation

                results = ordered

        result_df = batch_df.copy()

        p = get_settings().judge.column_prefix
        result_df[f"{p}coherencia"] = [r["coherence"]["puntuacion"] for r in results]
        result_df[f"{p}completitud"] = [r["completeness"]["puntuacion"] for r in results]
        result_df[f"{p}viabilidad"] = [r["feasibility"]["puntuacion"] for r in results]
        result_df[f"{p}formato"] = [r["format"]["puntuacion"] for r in results]
        result_df[f"{p}granularidad"] = [r["granularity"]["puntuacion"] for r in results]
        result_df[f"{p}total"] = [r["puntuacion_total"] for r in results]
        result_df[f"{p}aprobado"] = [r["aprobado"] for r in results]
        result_df[f"{p}problemas"] = [str(r["problemas_criticos"]) for r in results]
        result_df[f"{p}recomendaciones"] = [str(r["recomendaciones"]) for r in results]

        return result_df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes the DataFrame in batches for validation."""
        results = []

        for col in [self.historia_usuario_column, self.tareas_generadas_column]:
            if col not in df.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en DataFrame")

        valid_rows = df.dropna(
            subset=[self.historia_usuario_column, self.tareas_generadas_column]
        )
        if len(valid_rows) < len(df):
            self.logger.warning(
                f"Omitiendo {len(df) - len(valid_rows)} filas con datos faltantes"
            )

        with tqdm(total=len(valid_rows), desc=f"Validating {self.name}") as pbar:
            for batch in batch_dataframe(valid_rows, self.batch_size):
                processed_batch = self._process_batch(batch)
                results.append(processed_batch)
                pbar.update(len(batch))

        return pd.concat(results, ignore_index=False)
