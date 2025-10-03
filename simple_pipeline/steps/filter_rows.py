# simple_pipeline/steps/filter_rows.py

from typing import List, Callable, Optional
import pandas as pd
from ..base_step import BaseStep


class FilterRows(BaseStep):
    """
    Step que filtra filas del DataFrame basándose en una condición.
    Permite usar expresiones lambda o funciones personalizadas.
    """

    def __init__(
        self,
        name: str,
        filter_column: Optional[str] = None,
        filter_func: Optional[Callable] = None,
        condition: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            name: Nombre del step
            filter_column: Columna sobre la que aplicar el filtro (para condition)
            filter_func: Función que recibe una fila y devuelve True/False
            condition: Condición como string (e.g., "> 5", "== 'Python'")
            
        Examples:
            # Usando función:
            FilterRows(name="filter", filter_func=lambda row: row['age'] > 18)
            
            # Usando condición:
            FilterRows(name="filter", filter_column="age", condition="> 18")
        """
        super().__init__(name, **kwargs)
        self.filter_column = filter_column
        self.filter_func = filter_func
        self.condition = condition
        
        # Validar que se proporcione al menos un método de filtrado
        if filter_func is None and condition is None:
            raise ValueError("Must provide either 'filter_func' or 'condition'")
        
        if condition and not filter_column:
            raise ValueError("'filter_column' required when using 'condition'")

    @property
    def inputs(self) -> List[str]:
        if self.filter_column:
            return [self.filter_column]
        return []  # filter_func puede usar cualquier columna

    @property
    def outputs(self) -> List[str]:
        return []  # No agrega columnas, solo filtra

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica el filtro al DataFrame."""
        initial_count = len(df)
        
        if self.filter_func:
            # Usar función personalizada
            mask = df.apply(self.filter_func, axis=1)
            result = df[mask].copy()
        else:
            # Usar condición en columna específica
            # Construir la expresión
            query_str = f"`{self.filter_column}` {self.condition}"
            result = df.query(query_str).copy()
        
        final_count = len(result)
        filtered_count = initial_count - final_count
        
        print(f"  Filtered {filtered_count} rows ({initial_count} → {final_count})")
        
        return result
