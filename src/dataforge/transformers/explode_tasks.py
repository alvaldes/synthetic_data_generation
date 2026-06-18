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
        **kwargs
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
        # Crear una lista para almacenar las filas expandidas
        expanded_rows = []
        
        # Contador global para task_id (solo si no hay grouping)
        task_counter = 0
        
        # Si hay grouping, procesar por grupos
        if self.group_by_column and self.group_by_column in df.columns:
            # Procesar cada grupo por separado
            for group_value in df[self.group_by_column].unique():
                group_df = df[df[self.group_by_column] == group_value]
                
                # Resetear contador para este grupo
                group_task_counter = 0
                
                for idx, row in group_df.iterrows():
                    tasks_text = str(row[self.tasks_column])
                    
                    # Separar las tareas usando el patrón de numeración
                    tasks = self._split_tasks(tasks_text)
                    
                    # Si no se encontraron tareas numeradas, mantener el texto completo
                    if not tasks:
                        group_task_counter += 1
                        new_row = row.to_dict()
                        new_row[self.task_id_column] = group_task_counter
                        new_row[self.output_column] = self._clean_task_number(tasks_text)
                        expanded_rows.append(new_row)
                    else:
                        # Crear una fila por cada tarea
                        for task in tasks:
                            group_task_counter += 1
                            new_row = row.to_dict()
                            new_row[self.task_id_column] = group_task_counter
                            # Limpiar el número inicial de la tarea
                            new_row[self.output_column] = self._clean_task_number(task.strip())
                            expanded_rows.append(new_row)
        else:
            # Procesamiento sin grouping (contador global)
            for idx, row in df.iterrows():
                tasks_text = str(row[self.tasks_column])
                
                # Separar las tareas usando el patrón de numeración
                tasks = self._split_tasks(tasks_text)
                
                # Si no se encontraron tareas numeradas, mantener el texto completo
                if not tasks:
                    task_counter += 1
                    new_row = row.to_dict()
                    new_row[self.task_id_column] = task_counter
                    new_row[self.output_column] = self._clean_task_number(tasks_text)
                    expanded_rows.append(new_row)
                else:
                    # Crear una fila por cada tarea
                    for task in tasks:
                        task_counter += 1
                        new_row = row.to_dict()
                        new_row[self.task_id_column] = task_counter
                        # Limpiar el número inicial de la tarea
                        new_row[self.output_column] = self._clean_task_number(task.strip())
                        expanded_rows.append(new_row)

        # Crear nuevo DataFrame con las filas expandidas
        result_df = pd.DataFrame(expanded_rows)
        
        # Eliminar la columna original de tareas concatenadas si es diferente
        if self.tasks_column != self.output_column and self.tasks_column in result_df.columns:
            result_df = result_df.drop(columns=[self.tasks_column])
        
        # Reordenar columnas para poner task_id antes de task
        cols = list(result_df.columns)
        if self.task_id_column in cols and self.output_column in cols:
            # Remover task_id y task de su posición actual
            cols.remove(self.task_id_column)
            cols.remove(self.output_column)
            
            # Insertar task_id y task justo antes de la posición donde estaba task
            # o al final si no existía
            cols.extend([self.task_id_column, self.output_column])
            result_df = result_df[cols]
        
        return result_df

    def _split_tasks(self, tasks_text: str) -> List[str]:
        """
        Separa el texto de tareas en una lista de tareas individuales.

        Busca patrones como:
        "1. summary: ..."
        "2. summary: ..."

        Descarta cualquier texto introductorio (preamble) que aparezca
        antes de la primera tarea numerada.

        Fallback: si no encuentra tareas numeradas, divide por líneas que
        comienzan con ``summary:`` (case-insensitive). Esto cubre modelos
        que no siguen el formato exacto de numeración.
        """
        # --- Primary method: numbered split (case-insensitive) ----------
        numbered_split_pattern = r'\n(?=\d+\.\s+[Ss]ummary\s*:)'
        numbered_start_pattern = re.compile(r'^\d+\.\s+[Ss]ummary\s*:', re.IGNORECASE)

        tasks = re.split(numbered_split_pattern, tasks_text.strip())
        tasks = [t.strip() for t in tasks if t.strip()]

        # Si la primera "tarea" no empieza con un número, es preamble
        if tasks and not numbered_start_pattern.match(tasks[0]):
            tasks = tasks[1:]

        # Si el split numerado encontró 2+ tareas, usarlo
        if len(tasks) >= 2:
            return tasks

        # --- Fallback: split por any line starting with summary: --------
        fallback_pattern = r'\n(?=[Ss]ummary\s*:)'
        fallback_tasks = re.split(fallback_pattern, tasks_text.strip())
        fallback_tasks = [t.strip() for t in fallback_tasks if t.strip()]

        # Remover preamble (líneas antes del primer summary:)
        if fallback_tasks and not re.match(r'^[Ss]ummary\s*:', fallback_tasks[0]):
            fallback_tasks = fallback_tasks[1:]

        if len(fallback_tasks) > len(tasks):
            return fallback_tasks

        return tasks

    def _clean_task_number(self, task_text: str) -> str:
        """
        Limpia el número inicial de la tarea.
        
        Convierte:
        "1. summary: Crear base de datos\ndescription: ..."
        
        En:
        "summary: Crear base de datos\ndescription: ..."
        """
        # Patrón para detectar y remover el número inicial (ej: "1. ", "2. ", etc.)
        # Busca: inicio de línea, dígitos, punto, espacios
        pattern = r'^\d+\.\s+'
        
        # Remover el número inicial
        cleaned_text = re.sub(pattern, '', task_text.strip())
        
        return cleaned_text
