# simple_pipeline/steps/sample_rows.py

from typing import List, Optional
import pandas as pd
from ..base_step import BaseStep


class SampleRows(BaseStep):
    """
    Step que toma una muestra aleatoria del DataFrame.
    Útil para prototipar con subconjuntos de datos grandes.
    """

    def __init__(
        self,
        name: str,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            name: Nombre del step
            n: Número de filas a muestrear
            frac: Fracción de filas a muestrear (e.g., 0.1 para 10%)
            random_state: Semilla para reproducibilidad
            
        Examples:
            # Tomar 100 filas aleatorias
            SampleRows(name="sample", n=100, random_state=42)
            
            # Tomar 10% del dataset
            SampleRows(name="sample", frac=0.1, random_state=42)
        """
        super().__init__(name, **kwargs)
        
        if n is None and frac is None:
            raise ValueError("Must provide either 'n' or 'frac'")
        
        if n is not None and frac is not None:
            raise ValueError("Cannot provide both 'n' and 'frac'")
        
        self.n = n
        self.frac = frac
        self.random_state = random_state

    @property
    def inputs(self) -> List[str]:
        return []  # Usa todas las columnas

    @property
    def outputs(self) -> List[str]:
        return []  # No agrega columnas

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Toma una muestra del DataFrame."""
        initial_count = len(df)
        
        result = df.sample(
            n=self.n,
            frac=self.frac,
            random_state=self.random_state
        ).reset_index(drop=True)
        
        final_count = len(result)
        percentage = (final_count / initial_count * 100) if initial_count > 0 else 0
        
        print(f"  Sampled {final_count} rows from {initial_count} ({percentage:.1f}%)")
        
        return result
