#!/usr/bin/env python3
"""
Script para generar tareas de desarrollo a partir de historias de usuario.

Uso:
    python user_story_to_tasks.py input.csv output.csv
    python user_story_to_tasks.py input.csv output.csv --model llama3.1:8b --batch-size 4
"""

import pandas as pd
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List

from simple_pipeline import SimplePipeline
from simple_pipeline.steps import LoadDataFrame, RobustOllamaStep, OllamaLLMStep


def create_task_generation_prompt(row: Dict) -> str:
    """
    Crea el prompt para generar tareas a partir de una historia de usuario.
    
    Args:
        row: Fila del DataFrame con la columna 'input' que contiene el JSON
    
    Returns:
        Prompt formateado
    """
    try:
        user_story = json.loads(row['input'])
        project = user_story.get('project', 'Unknown Project')
        summary = user_story.get('summary', 'No summary')
        description = user_story.get('description', 'No description')
    except Exception as e:
        return f"Error parsing user story: {e}"
    
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides a user story with this format:
"project": "Name of the project"
"summary": "Summary of the user story"
"description": "Description of the user story"

Write a response that appropriately completes the request.


Instruction:


Break this user story into smaller development tasks to help the developers implement it efficiently. You can divide this user story into as many tasks as needed, depending on its complexity. Each task must be unique, actionable, and non-overlapping.

Use the following format for the response:

1. summary: ‹task summary 1›
description: ‹task description 1›
2. summary: ‹task summary 2›
description: ‹task description 2›


N. summary: ‹task summary N›
description: ‹task description N›


Input:


project: {project}
summary: {summary}
description: {description}


Response:"""
    
    return prompt


def parse_tasks_from_response(response_text: str) -> List[Dict[str, str]]:
    """
    Parsea las tareas del formato de respuesta.
    """
    tasks = []
    
    if not response_text or pd.isna(response_text):
        return tasks
    
    pattern = r'(\d+)\.\s*summary:\s*(.+?)\s*description:\s*(.+?)(?=\d+\.\s*summary:|$)'
    matches = re.finditer(pattern, response_text, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        tasks.append({
            'task_number': int(match.group(1)),
            'summary': match.group(2).strip(),
            'description': match.group(3).strip()
        })
    
    return tasks


def run_pipeline(
    input_csv: str,
    output_csv: str,
    model_name: str = "llama3.1:8b",
    batch_size: int = 2,
    temperature: float = 0.3,
    num_predict: int = 1000
):
    """
    Ejecuta el pipeline de generación de tareas.
    
    Args:
        input_csv: Ruta al CSV con historias de usuario
        output_csv: Ruta donde guardar el resultado
        model_name: Modelo de Ollama a usar
        batch_size: Número de historias a procesar simultáneamente
        temperature: Temperatura para generación
        num_predict: Tokens máximos a generar
    """
    
    print(f"\n{'='*80}")
    print("🚀 USER STORY TO TASKS PIPELINE")
    print(f"{'='*80}\n")
    
    # Cargar datos
    print(f"📥 Cargando datos desde: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   ✓ {len(df)} historias cargadas")
    
    # Verificar columna 'input'
    if 'input' not in df.columns:
        raise ValueError(
            "El CSV debe tener una columna 'input' con JSON formato:\n"
            '{"project": "...", "summary": "...", "description": "..."}'
        )
    
    # Crear pipeline
    print(f"\n⚙️ Configurando pipeline:")
    print(f"   Modelo: {model_name}")
    print(f"   Batch size: {batch_size}")
    print(f"   Temperature: {temperature}")
    
    pipeline = SimplePipeline(
        name="user-story-to-tasks-cli",
        description="CLI pipeline para generación de tareas"
    )
    
    pipeline.add_step(
        LoadDataFrame(name="load", df=df[['input']])
    )
    
    pipeline.add_step(
        OllamaLLMStep(
            name="generate_tasks",
            model_name=model_name,
            prompt_column="input",
            output_column="tasks_raw",
            prompt_template=create_task_generation_prompt,
            system_prompt="You are an expert software development lead who excels at breaking down user stories into clear, actionable development tasks.",
            batch_size=batch_size,
            generation_kwargs={
                "temperature": temperature,
                "num_predict": num_predict
            },
        )
    )
    
    # Ejecutar
    print(f"\n🔄 Procesando historias...\n")
    result_df = pipeline.run(use_cache=False)
    
    # Guardar
    print(f"\n💾 Guardando resultados...")
    result_df.to_csv(output_csv, index=False)
    print(f"   ✓ CSV: {output_csv}")
    
    print(f"\n{'='*80}")
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"{'='*80}\n")
    
    # Mostrar ejemplo
    print("📋 Ejemplo de resultado:")
    print(result_df.head())


def main():
    parser = argparse.ArgumentParser(
        description="Genera tareas de desarrollo a partir de historias de usuario",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python user_story_to_tasks.py historias.csv tareas.csv
  python user_story_to_tasks.py historias.csv tareas.csv --model mistral
  python user_story_to_tasks.py historias.csv tareas.csv --batch-size 4 --temperature 0.6

Formato del CSV de entrada:
  Debe tener una columna 'input' con JSON:
  {"project": "Mi Proyecto", "summary": "...", "description": "..."}
        """
    )
    
    parser.add_argument(
        'input_csv',
        help='Ruta al CSV con historias de usuario'
    )
    
    parser.add_argument(
        'output_csv',
        help='Ruta donde guardar las tareas generadas'
    )
    
    parser.add_argument(
        '--model',
        default='llama3.1:8b',
        help='Modelo de Ollama a usar (default: llama3.1:8b)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Historias a procesar simultáneamente (default: 2)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperatura para generación (default: 0.7)'
    )
    
    parser.add_argument(
        '--num-predict',
        type=int,
        default=1000,
        help='Tokens máximos a generar (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Validar archivos
    if not Path(args.input_csv).exists():
        print(f"❌ Error: No se encontró el archivo: {args.input_csv}")
        return 1
    
    try:
        run_pipeline(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            model_name=args.model,
            batch_size=args.batch_size,
            temperature=args.temperature,
            num_predict=args.num_predict
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
