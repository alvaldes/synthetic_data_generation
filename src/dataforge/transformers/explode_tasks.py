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
        # Supports four marker styles before ``summary:``:
        #   1) ``N.``  — standard numbered        (e.g. "2. summary:")
        #   2) ``N)``  — paren-numbered            (e.g. "2) summary:")
        #   3) ``x)``  — lettered paren            (e.g. "a) summary:")
        #   4) ``x.``  — lettered period           (e.g. "a. summary:")
        #
        # Also allows optional:
        #   - ``**`` markdown bold markers around summary:  (e.g. **summary:**)
        #   - parenthetical labels between number and summary:  (e.g. (Optional))
        #   - missing space after period  (e.g. "8.summary:")
        #   - leading whitespace before the marker (indented subtasks)
        #   - numbered tasks without summary: are NOT matched and stay attached
        #     (the subtasks inside them, if lettered+summary: DO match)
        numbered_split_pattern = (
            r'\n(?=\s*'
            r'(?:\d+\.|\d+\)|[a-z]\)|[a-z]\.)'  # markers: "5.", "2)", "a)", "a."
            r'\s*(?:\([^)]*\)\s*)?'              # optional (Optional) etc
            r'\*{0,2}[Ss]ummary'                 # optional ** bold
            r'\s*\*{0,2}\s*:)'                   # optional ** before colon
        )
        numbered_start_pattern = re.compile(
            r'^\s*'
            r'(?:\d+\.|\d+\)|[a-z]\)|[a-z]\.)?'  # marker is OPTIONAL — allows plain "summary:" as first task
            r'\s*(?:\([^)]*\)\s*)?'
            r'\*{0,2}[Ss]ummary'
            r'\s*\*{0,2}\s*:',
            re.IGNORECASE,
        )

        tasks = re.split(numbered_split_pattern, tasks_text.strip())
        tasks = [t.strip() for t in tasks if t.strip()]

        # Si la primera "tarea" no empieza con un marcador válido seguido de summary:,
        # ni con summary: directamente (primera tarea sin numerar), es preamble y se descarta.
        if tasks and not numbered_start_pattern.match(tasks[0]):
            tasks = tasks[1:]

        # Si el split numerado encontró 2+ tareas, usarlo
        if len(tasks) >= 2:
            return tasks

        # --- Fallback: split por any line starting with summary: --------
        # Allow optional ** markdown bold markers
        fallback_pattern = r'\n(?=\*{0,2}[Ss]ummary\s*\*{0,2}\s*:)'
        fallback_tasks = re.split(fallback_pattern, tasks_text.strip())
        fallback_tasks = [t.strip() for t in fallback_tasks if t.strip()]

        # Remover preamble (líneas antes del primer summary:)
        # Allow optional ** markdown bold markers
        if fallback_tasks and not re.match(r'^\*{0,2}[Ss]ummary\s*\*{0,2}\s*:', fallback_tasks[0]):
            fallback_tasks = fallback_tasks[1:]

        # El fallback debe encontrar AL MENOS 2 tareas para ser útil.
        # Si solo encuentra 1 (ej. el texto completo pegado), dejamos
        # que el 2do fallback intente un split más agresivo.
        if len(fallback_tasks) > len(tasks) and len(fallback_tasks) >= 2:
            return fallback_tasks

        # --- 2nd fallback: lettered subtasks with sub-numbering -----------
        # Some models output things like:
        #   a) 4.1. Create user authentication...
        #   b) 4.2. Develop data models...
        # (no ``summary:`` keyword at all in the subtasks)
        lettered_subnumber_pattern = r'\n(?=\s*[a-z]\)\s+\d+\.\d+\.\s+)'
        lettered_tasks = re.split(lettered_subnumber_pattern, tasks_text.strip())
        lettered_tasks = [t.strip() for t in lettered_tasks if t.strip()]

        # Remover preamble — texto que NO arranca con letra ni con summary:
        # (preserva parents como "summary: Implement core..." que tienen
        # subtareas con sub-números tipo "a) 4.1. Create...")
        if lettered_tasks:
            first = lettered_tasks[0].strip()
            is_preamble = not re.match(r'^[a-z]\)', first) \
                          and not re.match(r'^(summary|Summary)\s*:', first)
            if is_preamble:
                lettered_tasks = lettered_tasks[1:]

        if len(lettered_tasks) > len(tasks) and len(lettered_tasks) >= 2:
            return lettered_tasks

        return tasks

    def _clean_task_number(self, task_text: str) -> str:
        """
        Limpia el número inicial de la tarea.
        
        Convierte:
        "1. summary: Crear base de datos\ndescription: ..."
        
        En:
        "summary: Crear base de datos\ndescription: ..."

        También remueve:
        - Marcadores ``\d+)`` (ej: "2) summary:")
        - Marcadores con letras ``x)`` (ej: "a) summary:")
        - Marcadores con letras ``x.`` (ej: "a. summary:")
        - Sub-números ``X.Y.`` (ej: "4.1. Create...")
        - Marcadores markdown ** alrededor de ``summary:`` y ``description:``
        """
        # Remover el número inicial (ej: "1. ", "12. ", etc.)
        cleaned_text = re.sub(r'^\d+\.\s+', '', task_text.strip())
        
        # Remover marcador con paréntesis (ej: "2) summary:")
        cleaned_text = re.sub(r'^\d+\)\s+', '', cleaned_text)
        
        # Remover marcador con letra (ej: "a) summary:")
        cleaned_text = re.sub(r'^[a-z]\)\s+', '', cleaned_text)
        
        # Remover marcador con letra y punto (ej: "a. summary:")
        cleaned_text = re.sub(r'^[a-z]\.\s+', '', cleaned_text)
        
        # Remover sub-número jerárquico (ej: "4.1. Create...", "12.3.4. Title")
        # Esto se ejecuta DESPUÉS de remover los marcadores con letras, ya que
        # el patrón típico es "a) 4.1. Create..." → tras limpiar "a) " queda "4.1. Create..."
        cleaned_text = re.sub(r'^\d+\.\d+(?:\.\d+)*\s+', '', cleaned_text)
        
        # Remover marcadores markdown ** alrededor de summary: y description:
        # Esto cubre **summary:**, **summary:, **summary:**text**, **description:... etc.
        # 1) **summary**: → summary: (bold cierra justo antes del colon)
        cleaned_text = re.sub(
            r'\*\*(summary|Summary)\*\*\s*:', r'\1:', cleaned_text
        )
        # 2) **summary:** → summary: (bold wrapping todo el label con colon adentro)
        cleaned_text = re.sub(
            r'\*\*(summary|Summary)\s*:\s*\*\*', r'\1:', cleaned_text
        )
        # 3) **summary:  → summary:  (bold solo antes del label)
        cleaned_text = re.sub(
            r'\*\*(summary|Summary)\s*:', r'\1:', cleaned_text
        )
        # 4) **description**: → description:
        cleaned_text = re.sub(
            r'\*\*(description|Description)\*\*\s*:', r'\1:', cleaned_text
        )
        # 5) **description:** → description:
        cleaned_text = re.sub(
            r'\*\*(description|Description)\s*:\s*\*\*', r'\1:', cleaned_text
        )
        # 6) **description: → description:
        cleaned_text = re.sub(
            r'\*\*(description|Description)\s*:', r'\1:', cleaned_text
        )
        # 5) Strip trailing ** al final de cualquier línea (bold closing)
        cleaned_text = re.sub(r'\*\*\s*$', '', cleaned_text, flags=re.MULTILINE)
        
        return cleaned_text
