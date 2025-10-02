# simple_pipeline/steps/load_dataframe.py

from typing import List, Optional
import pandas as pd
from ..base_step import BaseStep

class LoadDataFrame(BaseStep):
    """
    Step generador que carga un DataFrame inicial para el pipeline.
    Puede recibir:
    - Un DataFrame directamente
    - Una ruta a un CSV para cargar los datos
    """

    def __init__(
        self,
        name: str,
        df: Optional[pd.DataFrame] = None,
        csv_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.df = df
        self.csv_path = csv_path

    @property
    def inputs(self) -> List[str]:
        """Un generador no requiere columnas de entrada."""
        return []

    @property
    def outputs(self) -> List[str]:
        """Las columnas disponibles dependen del DataFrame cargado."""
        if self.df is not None:
            return list(self.df.columns)
        return []

    def load(self) -> None:
        """Carga el DataFrame desde CSV si no se pasó uno directamente."""
        if self.csv_path and self.df is None:
            self.df = pd.read_csv(self.csv_path)
        super().load()

    def process(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Devuelve una copia del DataFrame cargado."""
        if self.df is None:
            raise ValueError(f"LoadDataFrame '{self.name}' no tiene datos cargados.")
        return self.df.copy()