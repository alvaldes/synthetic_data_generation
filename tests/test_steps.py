# tests/test_steps.py

import pandas as pd
import pytest

from simple_pipeline.steps.load_dataframe import LoadDataFrame
from simple_pipeline.steps.add_column import AddColumn
from simple_pipeline.steps.keep_columns import KeepColumns


def test_load_dataframe_from_df():
    df = pd.DataFrame({"x": [1, 2]})
    step = LoadDataFrame(name="load", df=df)
    result = step(pd.DataFrame())   # empty input because it's a generator
    assert "x" in result.columns
    assert result.equals(df)


def test_add_column_simple():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    step = AddColumn(
        name="sum",
        input_columns=["a", "b"],
        output_column="c",
        func=lambda a, b: a + b,
    )
    result = step(df)
    assert "c" in result.columns
    assert result["c"].tolist() == [4, 6]


def test_add_column_with_missing_input():
    df = pd.DataFrame({"a": [1]})
    step = AddColumn(
        name="bad",
        input_columns=["nonexistent"],
        output_column="y",
        func=lambda v: v * 2,
    )
    with pytest.raises(Exception):
        step(df)


def test_keep_columns():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    step = KeepColumns(name="keep", columns=["a", "c"])
    result = step(df)
    assert list(result.columns) == ["a", "c"]