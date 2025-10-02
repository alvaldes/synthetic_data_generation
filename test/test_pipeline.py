# tests/test_pipeline.py

import pandas as pd
import pytest

from simple_pipeline.pipeline import SimplePipeline
from simple_pipeline.steps.load_dataframe import LoadDataFrame
from simple_pipeline.steps.keep_columns import KeepColumns
from simple_pipeline.steps.add_column import AddColumn


def test_pipeline_runs_with_basic_steps():
    # Datos iniciales
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    # Crear pipeline
    pipeline = SimplePipeline(name="test-pipeline")

    # Step 1: cargar datos
    load = LoadDataFrame(name="load", df=df)

    # Step 2: crear nueva columna
    add = AddColumn(
        name="sum",
        input_columns=["a", "b"],
        output_column="c",
        func=lambda a, b: a + b,
    )

    # Step 3: quedarnos solo con columnas relevantes
    keep = KeepColumns(name="keep", columns=["a", "c"])

    # Construir pipeline
    pipeline.add_step(load)
    pipeline.add_step(add)
    pipeline.add_step(keep)

    # Ejecutar
    result = pipeline.run(use_cache=False)

    # Validar resultados
    assert "a" in result.columns
    assert "c" in result.columns
    assert result["c"].tolist() == [4, 6]


def test_pipeline_fails_with_wrong_column():
    df = pd.DataFrame({"x": [1]})
    pipeline = SimplePipeline(name="fail-pipeline")

    load = LoadDataFrame(name="load", df=df)

    add = AddColumn(
        name="wrong",
        input_columns=["nonexistent"],   # columna que no existe
        output_column="y",
        func=lambda v: v * 2,
    )

    pipeline.add_step(load)
    pipeline.add_step(add)

    # Ejecutar y esperar error
    with pytest.raises(Exception):
        pipeline.run(use_cache=False)