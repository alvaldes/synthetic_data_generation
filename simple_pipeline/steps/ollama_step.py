# simple_pipeline/steps/ollama_step.py

import ollama
import pandas as pd
from typing import Callable, List, Dict, Any, Optional
from tqdm import tqdm
import time
import re
from ..utils.logging import setup_logger
import logging
from pydantic import BaseModel

from ..base_step import BaseStep
from ..utils.batching import batch_dataframe, get_num_batches

class ResponseOutput(BaseModel):
    response: Optional[str]


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
        output_format: Optional[BaseModel] = ResponseOutput,
        system_prompt: Optional[str] = None,
        prompt_template: Callable[[Dict], str] = None,
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
        self.output_format = output_format
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
        return [self.output_column]

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
        return self.prompt_template(row)


    def _generate_with_retry(
        self,
        prompt: str,
        retry_count: int = 0
    ) -> Optional[str]:
        """Call model with retry logic on error, following ollama best practices."""
        try:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
                options=self.generation_kwargs
            )
            raw_content = response['message']['content']
            return raw_content

        except ollama.ResponseError as e:
            self.logger.warning(f"Ollama API error: {e.error}")
            if e.status_code == 404:
                self.logger.error(f"Model {self.model_name} not found. Try: ollama pull {self.model_name}")
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count  # exponential backoff
                time.sleep(wait_time)
                return self._generate_with_retry(prompt, retry_count + 1)
            else:
                self.logger.error(f"Failed after {self.max_retries} retries")
                return None
        except ConnectionError as e:
            self.logger.error(f"Connection error: {e}")
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                time.sleep(wait_time)
                return self._generate_with_retry(prompt, retry_count + 1)
            else:
                self.logger.error("Ollama server appears to be down. Check with: ollama serve")
                return None
        except Exception as e:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                time.sleep(wait_time)
                return self._generate_with_retry(prompt, retry_count + 1)
            else:
                self.logger.error(f"Unexpected error after {self.max_retries} retries: {e}")
                return None

    def _process_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of rows with Ollama."""
        results = []
        for _, row in batch_df.iterrows():
            prompt = self._format_prompt(row.to_dict())
            generation = self._generate_with_retry(prompt)
            results.append(generation)

        result_df = batch_df.copy()
        result_df[self.output_column] = results

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
