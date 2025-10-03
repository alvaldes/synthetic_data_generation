# simple_pipeline/steps/add_column.py

from typing import List, Callable
import pandas as pd
from ..base_step import BaseStep

class AddColumn(BaseStep):
    """
    Step que agrega una nueva columna calculada a partir de otras columnas.
    """

    def __init__(
        self,
        name: str,
        input_columns: List[str],
        output_column: str,
        func: Callable,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.input_columns = input_columns
        self.output_column = output_column
        self.func = func

    @property
    def inputs(self) -> List[str]:
        return self.input_columns

    @property
    def outputs(self) -> List[str]:
        return [self.output_column]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.output_column] = df.apply(
            lambda row: self.func(*[row[col] for col in self.input_columns]),
            axis=1
        )
        return df