# simple_pipeline/pipeline.py

from typing import List, Optional, Dict
from pathlib import Path
import pandas as pd
import pickle
import hashlib
import json
import shutil

from .base_step import BaseStep

class SimplePipeline:
    """
    Pipeline simplificado para procesar DataFrames paso a paso.
    - Permite añadir steps en secuencia
    - Maneja ejecución, cacheo y recuperación
    - Devuelve un DataFrame final transformado
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        cache_dir: str = ".cache/simple_pipeline"
    ):
        self.name = name
        self.description = description
        self.cache_dir = Path(cache_dir) / name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.steps: List[BaseStep] = []          # Lista de steps
        self._step_outputs: Dict[str, pd.DataFrame] = {}  # Resultados intermedios

    # -------- Añadir steps --------
    def add_step(self, step: BaseStep) -> 'SimplePipeline':
        """Añadir un step a la secuencia del pipeline."""
        self.steps.append(step)
        return self

    def __rshift__(self, step: BaseStep) -> 'SimplePipeline':
        """Permite usar la sintaxis pipeline >> step."""
        return self.add_step(step)

    # -------- Cache --------
    def _get_cache_key(self, step: BaseStep, input_hash: str) -> str:
        """Genera una clave de cache única para un step + datos de entrada."""
        step_config = {
            "name": step.name,
            "class": step.__class__.__name__,
            "inputs": step.inputs,
            "outputs": step.outputs,
        }
        step_str = json.dumps(step_config, sort_keys=True)
        combined = f"{step_str}_{input_hash}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_df_hash(self, df: pd.DataFrame) -> str:
        """Hash rápido del DataFrame para detectar cambios."""
        return hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()[:16]

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Intenta cargar un DataFrame desde la cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, cache_key: str, df: pd.DataFrame) -> None:
        """Guarda un DataFrame en la cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)

    # -------- Ejecución --------
    def run(
        self,
        input_df: Optional[pd.DataFrame] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo sobre un DataFrame inicial o desde un step generador.
        """
        # Paso inicial: con DataFrame externo o el primer step como generador
        if input_df is not None:
            current_df = input_df.copy()
            steps_to_process = self.steps
        else:
            if not self.steps:
                raise ValueError("Pipeline vacío: no hay steps definidos")

            first_step = self.steps[0]
            print(f"Loading data from {first_step.name}...")
            current_df = first_step(pd.DataFrame())  # Generadores no necesitan input
            self._step_outputs[first_step.name] = current_df
            steps_to_process = self.steps[1:]

        # Procesar cada step en orden
        for step in steps_to_process:
            print(f"\nExecuting step: {step.name}")

            input_hash = self._get_df_hash(current_df)
            cache_key = self._get_cache_key(step, input_hash)

            # Revisar cache
            if use_cache and step.cache:
                cached_df = self._load_from_cache(cache_key)
                if cached_df is not None:
                    print(f"  ✓ Loaded from cache")
                    current_df = cached_df
                    self._step_outputs[step.name] = current_df
                    continue

            # Ejecutar el step
            try:
                current_df = step(current_df)
                self._step_outputs[step.name] = current_df

                if use_cache and step.cache:
                    self._save_to_cache(cache_key, current_df)

                print(f"  ✓ Complete ({len(current_df)} rows, {len(current_df.columns)} columns)")

            except Exception as e:
                print(f"  ✗ Error in step {step.name}: {e}")
                raise
            finally:
                step.unload()

        return current_df

    # -------- Utilidades --------
    def clear_cache(self) -> None:
        """Elimina toda la cache del pipeline."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)