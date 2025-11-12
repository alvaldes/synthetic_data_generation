# simple_pipeline/steps/sort_rows.py

from typing import List, Union
import pandas as pd
from ..base_step import BaseStep


class SortRows(BaseStep):
    """
    Step that sorts DataFrame rows by one or more columns.
    """

    def __init__(
        self,
        name: str,
        by: Union[str, List[str]],
        ascending: Union[bool, List[bool]] = True,
        **kwargs
    ):
        """
        Args:
            name: Nombre del step
            by: Columna(s) por la(s) que ordenar
            ascending: Si ordenar ascendente (True) o descendente (False)
            
        Examples:
            # Ordenar por una columna
            SortRows(name="sort", by="age", ascending=False)
            
            # Sort by multiple columns
            SortRows(name="sort", by=["category", "price"], ascending=[True, False])
        """
        super().__init__(name, **kwargs)
        self.by = by if isinstance(by, list) else [by]
        self.ascending = ascending

    @property
    def inputs(self) -> List[str]:
        return self.by

    @property
    def outputs(self) -> List[str]:
        return []  # No agrega columnas, solo reordena

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordena el DataFrame."""
        result = df.sort_values(
            by=self.by,
            ascending=self.ascending
        ).reset_index(drop=True)
        
        print(f"  Sorted by: {', '.join(self.by)}")
        
        return result
