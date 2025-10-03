# simple_pipeline/pipeline.py

from typing import List, Optional, Dict
from pathlib import Path
import pandas as pd

from .base_step import BaseStep
from .utils.cache import CacheManager
from .utils.logging import setup_logger
import logging

class SimplePipeline:
    """Pipeline simplificado para procesar DataFrames paso a paso."""

    def __init__(
        self,
        name: str,
        description: str = "",
        cache_dir: str = ".cache/simple_pipeline",
        log_level: str = "INFO"
    ):
        self.name = name
        self.description = description
        
        # Usar CacheManager ✅
        self.cache_manager = CacheManager(cache_dir=f"{cache_dir}/{name}")
        
        # Setup logging
        self.logger = setup_logger(
            name=f"SimplePipeline.{name}",
            level=getattr(logging, log_level.upper())
        )
        
        self.steps: List[BaseStep] = []
        self._step_outputs: Dict[str, pd.DataFrame] = {}

    def add_step(self, step: BaseStep) -> 'SimplePipeline':
        """Añadir un step a la secuencia del pipeline."""
        self.steps.append(step)
        self.logger.info(f"Added step: {step.name}")
        return self

    def __rshift__(self, step: BaseStep) -> 'SimplePipeline':
        """Permite usar la sintaxis pipeline >> step."""
        return self.add_step(step)

    def run(
        self,
        input_df: Optional[pd.DataFrame] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Ejecuta el pipeline completo."""
        self.logger.info(f"Starting pipeline: {self.name}")
        self.logger.info(f"Number of steps: {len(self.steps)}")

        if not self.steps:
            raise ValueError("Pipeline has no steps")

        # Inicializar con input o primer step generador
        if input_df is not None:
            current_df = input_df.copy()
            steps_to_process = self.steps
        else:
            first_step = self.steps[0]
            self.logger.info(f"Executing generator step: {first_step.name}")
            current_df = first_step(pd.DataFrame())
            self._step_outputs[first_step.name] = current_df
            steps_to_process = self.steps[1:]

        # Procesar steps restantes
        for step in steps_to_process:
            self.logger.info(f"Executing step: {step.name}")

            # Usar CacheManager ✅
            if use_cache and step.cache:
                input_hash = self.cache_manager.get_df_hash(current_df)
                cache_key = self.cache_manager.get_cache_key(
                    step_name=step.name,
                    step_class=step.__class__.__name__,
                    inputs=step.inputs,
                    outputs=step.outputs,
                    input_hash=input_hash
                )
                
                cached_df = self.cache_manager.load(cache_key)
                if cached_df is not None:
                    self.logger.info(f"  ✓ Loaded from cache")
                    current_df = cached_df
                    self._step_outputs[step.name] = current_df
                    continue

            # Ejecutar step
            try:
                current_df = step(current_df)
                self._step_outputs[step.name] = current_df

                # Guardar en cache usando CacheManager ✅
                if use_cache and step.cache:
                    self.cache_manager.save(cache_key, current_df)

                self.logger.info(
                    f"  ✓ Complete ({len(current_df)} rows, "
                    f"{len(current_df.columns)} columns)"
                )

            except Exception as e:
                self.logger.error(f"  ✗ Error in step {step.name}: {e}")
                raise
            finally:
                step.unload()

        self.logger.info("Pipeline execution complete!")
        return current_df

    def clear_cache(self) -> None:
        """Elimina toda la cache del pipeline."""
        self.cache_manager.clear()
        self.logger.info("Cache cleared")

    def get_step_output(self, step_name: str) -> Optional[pd.DataFrame]:
        """Obtiene la salida de un step específico."""
        return self._step_outputs.get(step_name)