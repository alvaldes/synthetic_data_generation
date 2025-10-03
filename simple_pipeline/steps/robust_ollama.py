# simple_pipeline/steps/robust_ollama.py

from typing import List, Dict, Any, Optional, Callable, Tuple
import pandas as pd
from pathlib import Path

from .ollama_step import OllamaLLMStep


class RobustOllamaStep(OllamaLLMStep):
    """
    Versión mejorada de OllamaLLMStep con:
    - Dead letter queue para filas que fallan
    - Guardado de errores en CSV
    - Métricas de éxito/fallo
    - Tracking detallado de problemas
    """

    def __init__(
        self,
        name: str,
        save_failures: bool = True,
        failure_dir: str = "./failures",
        continue_on_error: bool = True,
        **kwargs
    ):
        """
        Args:
            name: Nombre del step
            save_failures: Si guardar filas fallidas en CSV
            failure_dir: Directorio para guardar fallos
            continue_on_error: Si continuar procesando tras un error
            **kwargs: Argumentos para OllamaLLMStep
        """
        super().__init__(name=name, **kwargs)
        self.save_failures = save_failures
        self.failure_dir = Path(failure_dir)
        self.continue_on_error = continue_on_error
        self.failed_rows: List[Tuple[int, pd.Series, Exception]] = []
        self.success_count = 0
        self.failure_count = 0

    def load(self) -> None:
        """Inicializa el cliente y crea directorio de fallos."""
        super().load()
        if self.save_failures:
            self.failure_dir.mkdir(parents=True, exist_ok=True)

    def _process_batch(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Procesa un batch con tracking de errores."""
        results = []

        for idx, row in batch_df.iterrows():
            try:
                prompt = self._format_prompt(row.to_dict())
                generation = self._generate_with_retry(prompt)

                if generation is None:
                    raise ValueError("Generation returned None after retries")

                # Agregar resultado exitoso
                result_row = row.to_dict()
                result_row[self.output_column] = generation
                result_row["model_name"] = self.model_name
                result_row["error"] = None
                result_row["status"] = "success"
                results.append(result_row)
                self.success_count += 1

            except Exception as e:
                # Agregar a dead letter queue
                failed_row = row.to_dict()
                failed_row[self.output_column] = None
                failed_row["model_name"] = self.model_name
                failed_row["error"] = str(e)
                failed_row["status"] = "failed"
                
                results.append(failed_row)
                self.failed_rows.append((idx, row, e))
                self.failure_count += 1
                
                if not self.continue_on_error:
                    raise

        return pd.DataFrame(results)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa con tracking y guardado de fallos."""
        # Reset de métricas
        self.failed_rows = []
        self.success_count = 0
        self.failure_count = 0
        
        # Procesar usando el método padre
        result_df = super().process(df)

        # Guardar filas fallidas si existen
        if self.save_failures and self.failed_rows:
            failed_df = pd.DataFrame([row.to_dict() for _, row, _ in self.failed_rows])
            
            # Agregar columna con el error
            failed_df['error_detail'] = [str(e) for _, _, e in self.failed_rows]
            
            # Guardar
            failed_file = self.failure_dir / f"{self.name}_failed.csv"
            failed_df.to_csv(failed_file, index=False)
            
            print(f"\n⚠️  {self.failure_count} rows failed")
            print(f"   Saved to: {failed_file}")

        # Mostrar métricas
        total = self.success_count + self.failure_count
        if total > 0:
            success_rate = (self.success_count / total * 100)
            print(f"\n📊 Metrics:")
            print(f"   Success: {self.success_count}/{total} ({success_rate:.1f}%)")
            print(f"   Failed:  {self.failure_count}/{total} ({100-success_rate:.1f}%)")

        return result_df

    def get_failed_rows(self) -> List[Tuple[int, pd.Series, Exception]]:
        """Retorna lista de (index, row, error) para filas fallidas."""
        return self.failed_rows
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Retorna un resumen de los fallos."""
        error_types = {}
        for _, _, error in self.failed_rows:
            error_type = type(error).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_failures": self.failure_count,
            "total_successes": self.success_count,
            "error_types": error_types,
            "failure_rate": (self.failure_count / (self.success_count + self.failure_count) * 100) 
                           if (self.success_count + self.failure_count) > 0 else 0
        }
