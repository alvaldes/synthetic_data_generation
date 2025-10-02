# simple_pipeline/utils/batching.py
import pandas as pd
from typing import Iterator, List


def batch_dataframe(
    df: pd.DataFrame,
    batch_size: int
) -> Iterator[pd.DataFrame]:
    """Yield batches (memory efficient)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size]


def split_dataframe(
    df: pd.DataFrame,
    n_splits: int
) -> List[pd.DataFrame]:
    """Split for parallelization."""
    if n_splits <= 0:
        raise ValueError("n_splits must be greater than 0")
    
    return [df.iloc[i::n_splits] for i in range(n_splits)]


def get_num_batches(df: pd.DataFrame, batch_size: int) -> int:
    """Calculate number of batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    
    return (len(df) + batch_size - 1) // batch_size