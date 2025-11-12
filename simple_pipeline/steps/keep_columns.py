# simple_pipeline/steps/keep_columns.py

from typing import List
import pandas as pd
from ..base_step import BaseStep

class KeepColumns(BaseStep):
    """
    Step that keeps only the specified columns.
    """

    def __init__(self, name: str, columns: List[str], **kwargs):
        super().__init__(name, **kwargs)
        self.columns = columns

    @property
    def inputs(self) -> List[str]:
        return self.columns

    @property
    def outputs(self) -> List[str]:
        return self.columns

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.columns].copy()