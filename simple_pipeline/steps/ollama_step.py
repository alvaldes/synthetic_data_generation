# simple_pipeline/steps/ollama_step.py

import ollama
import pandas as pd
from typing import Callable, List, Dict, Any, Optional
from tqdm import tqdm
import time
import re
from ..utils.logging import setup_logger
import logging

from ..base_step import BaseStep
from ..utils.batching import batch_dataframe, get_num_batches


class OllamaLLMStep(BaseStep):
    """
    Step que llama a un modelo LLM en Ollama para inferencia sobre filas de un DataFrame.
    - Usa una columna del DataFrame como prompt de entrada
    - Genera una nueva columna con la respuesta del modelo
    - Permite definir un system_prompt y plantillas de prompt
    - Soporta procesamiento en batches y reintentos automáticos
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        prompt_column: str,
        output_column: str = "generation",
        system_prompt: Optional[str] = None,
        prompt_template: Optional[Callable[[Dict], str]] = None,
        batch_size: int = 8,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        ollama_host: str = "http://localhost:11434",
        max_retries: int = 3,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.model_name = model_name
        self.prompt_column = prompt_column
        self.output_column = output_column
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        self.generation_kwargs = generation_kwargs or {}
        self.ollama_host = ollama_host
        self.max_retries = max_retries
        self.client = None
        self.logger = setup_logger(
            name=f"OllamaLLMStep.{name}",
            level=logging.INFO
        )

    # -------- Propiedades requeridas --------
    @property
    def inputs(self) -> List[str]:
        return [self.prompt_column]

    @property
    def outputs(self) -> List[str]:
        return [self.output_column, "model_name"]

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
    def _format_prompt(self, row: Dict[str, Any]) -> str:
        """Construye el prompt usando la plantilla o directamente la columna."""
        if self.prompt_template:
            return self.prompt_template(row)
        return str(row[self.prompt_column])

    def _clean_generation(self, text: Optional[str]) -> Optional[str]:
        """Limpia la salida del modelo removiendo secciones <think> y texto irrelevante."""
        if not text:
            return text

        # Elimina bloques <think>...</think>
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # Para eliminar cualquier bloque o etiquetas HTML/XML
        # cleaned = re.sub(r"<.*?>.*?</.*?>", "", text, flags=re.DOTALL)

        # Limpia espacios en blanco, saltos de línea redundantes
        cleaned = cleaned.strip()

        return cleaned

    def _generate_with_retry(
        self,
        prompt: str,
        retry_count: int = 0
    ) -> Optional[str]:
        """Llama al modelo con reintentos en caso de error."""
        try:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            messages.append({"role": "user", "content": prompt})

            # Todo: hay que tratar con format para que devuelva un json con lo que excactamente queremos
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
                options=self.generation_kwargs
            )
            raw_content = response['message']['content']
            return self._clean_generation(raw_content)

        except Exception as e:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count  # backoff exponencial
                time.sleep(wait_time)
                return self._generate_with_retry(prompt, retry_count + 1)
            else:
                self.logger.error(f"Error tras {self.max_retries} reintentos: {e}")
                return None

    def _process_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Procesa un batch de filas con Ollama."""
        results = []
        for _, row in batch_df.iterrows():
            prompt = self._format_prompt(row.to_dict())
            generation = self._generate_with_retry(prompt)
            results.append(generation)
        
        result_df = batch_df.copy()
        result_df[self.output_column] = results
        result_df["model_name"] = self.model_name
        
        return result_df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa el DataFrame en batches a través de Ollama."""
        results = []
        
        # Usar utilidades de batching ✅
        with tqdm(total=len(df), desc=f"Processing {self.name}") as pbar:
            for batch in batch_dataframe(df, self.batch_size):  # ✅ Iterator
                processed_batch = self._process_batch(batch)
                results.append(processed_batch)
                pbar.update(len(batch))
        
        return pd.concat(results, ignore_index=False)
