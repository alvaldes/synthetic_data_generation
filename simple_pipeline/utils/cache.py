# simple_pipeline/utils/cache.py

import pandas as pd
import pickle
import hashlib
import json
from pathlib import Path
from typing import Optional


class CacheManager:
    """
    Maneja el almacenamiento en caché de DataFrames para steps del pipeline.
    - Genera claves únicas por combinación de step + entrada
    - Guarda y carga resultados en disco
    """

    def __init__(self, cache_dir: str = ".cache/simple_pipeline"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -------- Generar claves únicas --------
    def get_cache_key(self, step_name: str, step_class: str, inputs, outputs, input_hash: str) -> str:
        """Crea un hash único basado en config del step + hash de entrada."""
        step_config = {
            "name": step_name,
            "class": step_class,
            "inputs": inputs,
            "outputs": outputs,
        }
        step_str = json.dumps(step_config, sort_keys=True)
        combined = f"{step_str}_{input_hash}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get_df_hash(self, df: pd.DataFrame) -> str:
        """Crea un hash del DataFrame (contenido + índice)."""
        return hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()[:16]

    # -------- Guardar / cargar --------
    def load(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Carga un DataFrame desde cache si existe."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def save(self, cache_key: str, df: pd.DataFrame) -> None:
        """Guarda un DataFrame en cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)

    def clear(self) -> None:
        """Elimina todo el directorio de cache."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)