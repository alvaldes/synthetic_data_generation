# simple_pipeline/base_step.py

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import pandas as pd

class BaseStep(ABC):
    """
    Clase abstracta base para todos los pasos (steps) del pipeline.
    Define el ciclo de vida de un step:
    - Validación de entradas
    - Transformación de datos
    - Mapeo de columnas de entrada y salida
    """

    def __init__(
        self,
        name: str,
        input_mappings: Optional[Dict[str, str]] = None,
        output_mappings: Optional[Dict[str, str]] = None,
        cache: bool = True
    ):
        self.name = name
        self.input_mappings = input_mappings or {}
        self.output_mappings = output_mappings or {}
        self.cache = cache
        self._loaded = False

    # --------- Propiedades abstractas ---------
    @property
    @abstractmethod
    def inputs(self) -> List[str]:
        """Nombres de columnas que este step requiere como entrada."""
        pass

    @property
    @abstractmethod
    def outputs(self) -> List[str]:
        """Nombres de columnas que este step produce como salida."""
        pass

    # --------- Ciclo de vida del step ---------
    def load(self) -> None:
        """Inicializa recursos (ej. cargar modelos, conectar a APIs)."""
        self._loaded = True

    def unload(self) -> None:
        """Libera recursos (ej. cerrar conexiones)."""
        self._loaded = False

    # --------- Validaciones y mapeos ---------
    def _apply_input_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renombra columnas de entrada según input_mappings."""
        if self.input_mappings:
            df = df.rename(columns=self.input_mappings)
        return df

    def _apply_output_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renombra columnas de salida según output_mappings."""
        if self.output_mappings:
            df = df.rename(columns=self.output_mappings)
        return df

    def _validate_inputs(self, df: pd.DataFrame) -> None:
        """Verifica que las columnas requeridas existen en el DataFrame."""
        missing = set(self.inputs) - set(df.columns)
        if missing:
            raise ValueError(
                f"Step '{self.name}' no tiene columnas de entrada requeridas: {missing}. "
                f"Disponibles: {list(df.columns)}"
            )

    # --------- Método abstracto de proceso ---------
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforma el DataFrame y devuelve el resultado."""
        pass

    # --------- Ejecución del step ---------
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta el step con validación, mapeos y transformación.
        """
        if not self._loaded:
            self.load()

        # Aplicar mapeos de entrada
        df = self._apply_input_mappings(df)

        # Validar entradas
        self._validate_inputs(df)

        # Procesar datos
        df = self.process(df)

        # Aplicar mapeos de salida
        df = self._apply_output_mappings(df)

        return df
    
    def get_config(self) -> Dict[str, Any]:
        """Get step configuration for caching."""
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
