# simple_pipeline/steps/ollama_step.py

import ollama
import pandas as pd
from typing import Callable, List, Dict, Any, Optional
from tqdm import tqdm
import time

from ..base_step import BaseStep


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
        return row[self.prompt_column]

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

            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
                **self.generation_kwargs
            )
            return response['message']['content']

        except Exception as e:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count  # backoff exponencial
                time.sleep(wait_time)
                return self._generate_with_retry(prompt, retry_count + 1)
            else:
                print(f"Error tras {self.max_retries} reintentos: {e}")
                return None

    def _process_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Procesa un batch de filas con Ollama."""
        results = []
        for _, row in batch_df.iterrows():
            prompt = self._format_prompt(row.to_dict())
            generation = self._generate_with_retry(prompt)
            results.append({
                self.output_column: generation,
                "model_name": self.model_name
            })

        # Devuelve DataFrame con nuevas columnas añadidas
        result_df = pd.DataFrame(results, index=batch_df.index)
        return pd.concat([batch_df, result_df], axis=1)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa el DataFrame en batches a través de Ollama."""
        results = []
        num_batches = (len(df) + self.batch_size - 1) // self.batch_size

        with tqdm(total=len(df), desc=f"Processing {self.name}") as pbar:
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(df))
                batch = df.iloc[start_idx:end_idx]

                processed_batch = self._process_batch(batch)
                results.append(processed_batch)
                pbar.update(len(batch))

        return pd.concat(results, ignore_index=False)