# dataforge/steps/ollama_step.py

import ollama
import pandas as pd
from typing import Callable, List, Dict, Any, Optional, Union
from tqdm import tqdm
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel

from ..base_step import BaseStep
from ..config import get_settings
from ..utils.batching import batch_dataframe
from ..utils.logging import setup_logger
import logging

_thread_local = threading.local()

class ResponseOutput(BaseModel):
    response: Optional[str]


class OllamaLLMStep(BaseStep):
    """
    Step that calls an LLM model in Ollama for inference on DataFrame rows.

    - Uses a DataFrame column as input prompt
    - Generates a new column with the model response
    - Allows defining system_prompt and prompt templates
    - Supports batch processing and automatic retries

    Default values (model, host, retries, etc.) come from the centralised
    :ref:`config <DataForgeSettings>`.  Explicit constructor arguments always
    take precedence.
    """

    def __init__(
        self,
        name: str,
        model_name: Optional[str] = None,
        prompt_column: Optional[str] = None,
        output_column: str = "generation",
        output_format: Optional[BaseModel] = ResponseOutput,
        system_prompt: Optional[str] = None,
        prompt_template: Callable[[Dict], str] = None,
        batch_size: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        ollama_host: Optional[str] = None,
        max_retries: Optional[int] = None,
        track_time: bool = False,
        time_column: Optional[str] = None,
        num_workers: Optional[int] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        cfg = get_settings().llm

        self.model_name = model_name or cfg.default_model
        self.prompt_column = prompt_column
        self.output_column = output_column
        self.output_format = output_format
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.batch_size = batch_size if batch_size is not None else cfg.batch_size
        self.generation_kwargs = generation_kwargs or {
            "temperature": cfg.temperature,
            "num_predict": cfg.num_predict,
        }
        self.ollama_host = ollama_host or cfg.ollama_host
        self.max_retries = max_retries if max_retries is not None else cfg.max_retries
        self.track_time = track_time
        self.time_column = time_column or f"{output_column}_time"
        self.num_workers = num_workers if num_workers is not None else cfg.num_workers
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
        if self.track_time:
            return [self.output_column, self.time_column]
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

    def _get_thread_client(self) -> ollama.Client:
        """Get thread-local Ollama client for parallel execution."""
        if not hasattr(_thread_local, "client"):
            _thread_local.client = ollama.Client(host=self.ollama_host)
        return _thread_local.client

    def _generate_with_retry(
        self,
        prompt: str,
        retry_count: int = 0,
        client: Optional[ollama.Client] = None,
    ) -> Optional[str]:
        """Call model with retry logic on error, following ollama best practices.

        Args:
            prompt: The prompt to send to the model.
            retry_count: Current retry attempt number.
            client: Optional Ollama client instance. Falls back to self.client if None.
        """
        client = client or self.client
        try:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = client.chat(
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

    def _generate_with_retry_parallel(self, prompt: str) -> Any:
        """Wrapper for parallel execution: uses thread-local client and tracks time."""
        client = self._get_thread_client()
        if self.track_time:
            start = time.time()
            generation = self._generate_with_retry(prompt, client=client)
            elapsed = time.time() - start
            return generation, elapsed
        return self._generate_with_retry(prompt, client=client)

    def _process_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of rows with Ollama."""
        results = []
        times = []

        if self.num_workers <= 1:
            # Sequential path — uses self.client
            for _, row in batch_df.iterrows():
                prompt = self._format_prompt(row.to_dict())

                if self.track_time:
                    start_time = time.time()
                    generation = self._generate_with_retry(prompt)
                    elapsed_time = time.time() - start_time
                    times.append(elapsed_time)
                else:
                    generation = self._generate_with_retry(prompt)

                results.append(generation)
        else:
            # Parallel path — thread-local clients via ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                future_to_idx: Dict[Any, int] = {}
                for idx, (_, row) in enumerate(batch_df.iterrows()):
                    prompt = self._format_prompt(row.to_dict())
                    future = pool.submit(self._generate_with_retry_parallel, prompt)
                    future_to_idx[future] = idx

                ordered_results: List[Optional[str]] = [None] * len(batch_df)
                ordered_times: List[Optional[float]] = [None] * len(batch_df)

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result = future.result()
                    if self.track_time:
                        generation, elapsed = result
                        ordered_results[idx] = generation
                        ordered_times[idx] = elapsed
                    else:
                        ordered_results[idx] = result

                results = ordered_results
                times = ordered_times

        result_df = batch_df.copy()
        result_df[self.output_column] = results

        if self.track_time:
            result_df[self.time_column] = times

        return result_df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes the DataFrame in batches through Ollama."""
        results = []

        with tqdm(total=len(df), desc=f"Processing {self.name}") as pbar:
            for batch in batch_dataframe(df, self.batch_size):
                processed_batch = self._process_batch(batch)
                results.append(processed_batch)
                pbar.update(len(batch))

        return pd.concat(results, ignore_index=False)
