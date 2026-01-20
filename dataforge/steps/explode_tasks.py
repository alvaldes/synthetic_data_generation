# dataforge/steps/explode_tasks.py

from typing import List
import pandas as pd
import re
from ..base_step import BaseStep


class ExplodeTasks(BaseStep):
    """
    Step que separa tareas numeradas en filas individuales.
    
    Toma una columna con múltiples tareas en formato:
    "1. summary: Tarea 1
    description: Descripción 1
    
    2. summary: Tarea 2
    description: Descripción 2"
    
    Y las separa en filas individuales, duplicando las otras columnas.
    """

    def __init__(
        self,
        name: str,
        tasks_column: str = "tasks",
        output_column: str = "task",
        **kwargs
    ):
        """
        Args:
            name: Nombre del step
            tasks_column: Columna que contiene las tareas concatenadas
            output_column: Nombre de la columna para la tarea individual
        """
        super().__init__(name, **kwargs)
        self.tasks_column = tasks_column
        self.output_column = output_column

    @property
    def inputs(self) -> List[str]:
        return [self.tasks_column]

    @property
    def outputs(self) -> List[str]:
        return [self.output_column]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa el DataFrame separando las tareas en filas individuales.
        """
        # Crear una lista para almacenar las filas expandidas
        expanded_rows = []

        for idx, row in df.iterrows():
            tasks_text = row[self.tasks_column]
            
            # Separar las tareas usando el patrón de numeración
            # Busca patrones como "1. summary:", "2. summary:", etc.
            tasks = self._split_tasks(tasks_text)
            
            # Si no se encontraron tareas numeradas, mantener el texto completo
            if not tasks:
                new_row = row.to_dict()
                new_row[self.output_column] = tasks_text
                expanded_rows.append(new_row)
            else:
                # Crear una fila por cada tarea
                for task in tasks:
                    new_row = row.to_dict()
                    new_row[self.output_column] = task.strip()
                    expanded_rows.append(new_row)

        # Crear nuevo DataFrame con las filas expandidas
        result_df = pd.DataFrame(expanded_rows)
        
        # Eliminar la columna original de tareas concatenadas si es diferente
        if self.tasks_column != self.output_column and self.tasks_column in result_df.columns:
            result_df = result_df.drop(columns=[self.tasks_column])
        
        return result_df

    def _split_tasks(self, tasks_text: str) -> List[str]:
        """
        Separa el texto de tareas en una lista de tareas individuales.
        
        Busca patrones como:
        "1. summary: ..."
        "2. summary: ..."
        etc.
        """
        # Patrón para detectar el inicio de cada tarea (número seguido de punto)
        pattern = r'\n(?=\d+\.\s+summary:)'
        
        # Separar por el patrón
        tasks = re.split(pattern, tasks_text.strip())
        
        # Filtrar tareas vacías
        tasks = [task.strip() for task in tasks if task.strip()]
        
        return tasks
