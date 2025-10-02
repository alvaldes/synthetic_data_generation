# simple_pipeline/steps/robust_ollama.py

import pandas as pd
from typing import Any, Dict
from .ollama_step import OllamaLLMStep


class RobustOllamaStep(OllamaLLMStep):
    """
    Versión robusta de OllamaLLMStep que maneja errores fila por fila.
    - Si una fila falla, no detiene el pipeline.
    - Guarda las filas fallidas en un CSV para reprocesarlas después.
    """

    def __init__(self, save_failures: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.save_failures = save_failures
        self.failed_rows = []

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa con control de errores fila por fila."""
        results = []

        for idx, row in df.iterrows():
            try:
                prompt = self._format_prompt(row.to_dict())
                generation = self._generate_with_retry(prompt)

                result_row = row.copy()
                result_row[self.output_column] = generation
                result_row["model_name"] = self.model_name
                result_row["error"] = None
                results.append(result_row)

            except Exception as e:
                # Manejo de error: guardar fila con detalle del fallo
                failed_row = row.copy()
                failed_row[self.output_column] = None
                failed_row["model_name"] = self.model_name
                failed_row["error"] = str(e)
                results.append(failed_row)

                # Acumular en lista de fallos
                self.failed_rows.append((idx, row, e))

        result_df = pd.DataFrame(results)

        # Guardar filas fallidas si corresponde
        if self.save_failures and self.failed_rows:
            failed_df = pd.DataFrame([row for _, row, _ in self.failed_rows])
            failed_df.to_csv(f"{self.name}_failed.csv", index=False)
            print(f"⚠ {len(self.failed_rows)} filas fallaron. Guardadas en {self.name}_failed.csv")

        return result_df