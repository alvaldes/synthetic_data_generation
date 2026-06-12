# examples/demo_pipeline.py
from framework.dataforge import DataForgePipeline
from framework.dataforge.steps import LoadDataFrame, AddColumn
import pandas as pd

print("--- Running Simple Framework Demo ---")
df = pd.DataFrame({"input": ["hello", "world", "dataforge"]})
pipeline = DataForgePipeline(name="simple_framework_test")
(
    pipeline
    >> LoadDataFrame(name="load_data", df=df)
    >> AddColumn(
        name="to_upper",
        input_columns=["input"],
        output_column="output",
        # ¡ACÁ ESTÁ EL CAMBIO CLAVE! 'text_value' recibe DIRECTAMENTE el valor de 'input'
        func=lambda text_value: text_value.upper(),
    )
)
result = pipeline.run(use_cache=False)
print("\nResultados del pipeline simple:")
print(result)
print("--- Demo Finalizada ---")
