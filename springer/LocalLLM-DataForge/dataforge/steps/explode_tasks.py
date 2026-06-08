# dataforge/steps/explode_tasks.py

from typing import List, Optional
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

    Y las separa en filas individuales con:
    - task_id: Identificador de la tarea (contador global o por grupo)
    - task: Contenido de la tarea (sin el número inicial)
    """

    def __init__(
        self,
        name: str,
        tasks_column: str = "tasks",
        output_column: str = "task",
        task_id_column: str = "task_id",
        group_by_column: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            name: Nombre del step
            tasks_column: Columna que contiene las tareas concatenadas
            output_column: Nombre de la columna para la tarea individual
            task_id_column: Nombre de la columna para el ID de la tarea
            group_by_column: Si se especifica, task_id se resetea por cada grupo (ej: 'us_id')
                           Si es None, task_id es un contador global
        """
        super().__init__(name, **kwargs)
        self.tasks_column = tasks_column
        self.output_column = output_column
        self.task_id_column = task_id_column
        self.group_by_column = group_by_column

    @property
    def inputs(self) -> List[str]:
        inputs_list = [self.tasks_column]
        if self.group_by_column:
            inputs_list.append(self.group_by_column)
        return inputs_list

    @property
    def outputs(self) -> List[str]:
        return [self.task_id_column, self.output_column]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa el DataFrame separando las tareas en filas individuales.
        Agrega task_id y limpia el número inicial de cada tarea.

        Si group_by_column está especificado, task_id se resetea por cada grupo.
        Si no, task_id es un contador global.
        """
        expanded_rows = []

        task_counter = 0

        if self.group_by_column and self.group_by_column in df.columns:
            for group_value in df[self.group_by_column].unique():
                group_df = df[df[self.group_by_column] == group_value]

                group_task_counter = 0

                for idx, row in group_df.iterrows():
                    tasks_text = str(row[self.tasks_column])

                    tasks = self._split_tasks(tasks_text)

                    if not tasks:
                        group_task_counter += 1
                        new_row = row.to_dict()
                        new_row[self.task_id_column] = group_task_counter
                        new_row[self.output_column] = self._clean_task_number(
                            tasks_text
                        )
                        expanded_rows.append(new_row)
                    else:
                        for task in tasks:
                            group_task_counter += 1
                            new_row = row.to_dict()
                            new_row[self.task_id_column] = group_task_counter
                            new_row[self.output_column] = self._clean_task_number(
                                task.strip()
                            )
                            expanded_rows.append(new_row)
        else:
            for idx, row in df.iterrows():
                tasks_text = str(row[self.tasks_column])

                tasks = self._split_tasks(tasks_text)

                if not tasks:
                    task_counter += 1
                    new_row = row.to_dict()
                    new_row[self.task_id_column] = task_counter
                    new_row[self.output_column] = self._clean_task_number(tasks_text)
                    expanded_rows.append(new_row)
                else:
                    for task in tasks:
                        task_counter += 1
                        new_row = row.to_dict()
                        new_row[self.task_id_column] = task_counter
                        new_row[self.output_column] = self._clean_task_number(
                            task.strip()
                        )
                        expanded_rows.append(new_row)

        result_df = pd.DataFrame(expanded_rows)

        if (
            self.tasks_column != self.output_column
            and self.tasks_column in result_df.columns
        ):
            result_df = result_df.drop(columns=[self.tasks_column])

        cols = list(result_df.columns)
        if self.task_id_column in cols and self.output_column in cols:
            cols.remove(self.task_id_column)
            cols.remove(self.output_column)

            cols.extend([self.task_id_column, self.output_column])
            result_df = result_df[cols]

        return result_df

    def _split_tasks(self, tasks_text: str) -> List[str]:
        """
        Separa el texto de tareas en una lista de tareas individuales.

        Busca patrones como:
        "1. summary: ..."
        "2. summary: ..."
        etc.
        """
        pattern = r"\n(?=\d+\.\s+summary:)"

        tasks = re.split(pattern, tasks_text.strip())

        tasks = [task.strip() for task in tasks if task.strip()]

        return tasks

    def _clean_task_number(self, task_text: str) -> str:
        """
        Limpia el número inicial de la tarea.

        Convierte:
        "1. summary: Crear base de datos\ndescription: ..."

        En:
        "summary: Crear base de datos\ndescription: ..."
        """
        pattern = r"^\d+\.\s+"

        cleaned_text = re.sub(pattern, "", task_text.strip())

        return cleaned_text
