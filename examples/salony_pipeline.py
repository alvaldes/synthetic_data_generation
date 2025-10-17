#!/usr/bin/env python3
"""
Script para generar tareas de desarrollo a partir de historias de usuario del dataset Salony.

Este pipeline toma historias de usuario del dataset salony_train.csv y las descompone
en tareas de desarrollo más pequeñas y accionables.

Uso:
    python salony_pipeline.py output.csv
    python salony_pipeline.py output.csv --model llama3.1:8b --batch-size 4
    python salony_pipeline.py output.csv --sample 10
"""

import pandas as pd
import argparse
from pathlib import Path
from typing import Dict

from simple_pipeline import SimplePipeline
from simple_pipeline.steps import LoadDataFrame, OllamaLLMStep


def create_task_generation_prompt(row: Dict) -> str:
    """
    Crea el prompt para generar tareas a partir de una historia de usuario del dataset Salony.
    
    Args:
        row: Fila del DataFrame con la columna 'input' que contiene la historia
    
    Returns:
        Prompt formateado
    """
    user_story = row['input'].strip()
    
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides a user story.

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

{user_story}


Response:"""
    
    return prompt


def run_salony_pipeline(
    output_csv: str,
    model_name: str = "llama3.1:8b",
    batch_size: int = 2,
    temperature: float = 0.3,
    num_predict: int = 1000,
    sample_size: int = None
):
    """
    Ejecuta el pipeline de generación de tareas para historias de usuario Salony.
    
    Args:
        output_csv: Ruta donde guardar el resultado
        model_name: Modelo de Ollama a usar
        batch_size: Número de historias a procesar simultáneamente
        temperature: Temperatura para generación
        num_predict: Tokens máximos a generar
        sample_size: Si se especifica, procesa solo N historias (para pruebas)
    """
    
    print(f"\n{'='*80}")
    print("🚀 SALONY USER STORIES TO TASKS PIPELINE")
    print(f"{'='*80}\n")
    
    # Cargar datos
    input_csv = Path(__file__).parent.parent / "data" / "salony_train.csv"
    print(f"📥 Cargando datos desde: {input_csv}")
    
    if not input_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {input_csv}")
    
    df = pd.read_csv(input_csv)
    
    # Eliminar la primera columna si es un índice
    if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
        df = df.iloc[:, 1:]
    
    print(f"   ✓ {len(df)} historias cargadas")
    
    # Verificar columna 'input'
    if 'input' not in df.columns:
        raise ValueError("El CSV debe tener una columna 'input' con las historias de usuario")
    
    # Aplicar sampling si se solicita
    if sample_size:
        df = df.head(sample_size)
        print(f"   ℹ️  Procesando solo {sample_size} historias (modo muestra)")
    
    # Limpiar datos
    df = df.dropna(subset=['input'])
    df['input'] = df['input'].str.strip()
    
    # Crear pipeline
    print(f"\n⚙️ Configurando pipeline:")
    print(f"   Modelo: {model_name}")
    print(f"   Batch size: {batch_size}")
    print(f"   Temperature: {temperature}")
    print(f"   Historias a procesar: {len(df)}")
    
    pipeline = SimplePipeline(
        name="salony-tasks-pipeline",
        description="Pipeline para generar tareas de desarrollo del dataset Salony"
    )
    
    pipeline.add_step(
        LoadDataFrame(name="load", df=df)
    )
    
    pipeline.add_step(
        OllamaLLMStep(
            name="generate_tasks",
            model_name=model_name,
            prompt_column="input",
            output_column="tasks",
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
    print(f"   ✓ CSV guardado: {output_csv}")
    print(f"   ✓ {len(result_df)} historias procesadas")
    
    print(f"\n{'='*80}")
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"{'='*80}\n")
    
    # Mostrar ejemplo
    print("📋 Ejemplo de resultado (primeras 3 filas):\n")
    for idx, row in result_df.head(3).iterrows():
        print(f"🔹 Historia #{idx}:")
        print(f"   Input: {row['input'][:100]}...")
        if 'tasks' in row and pd.notna(row['tasks']):
            print(f"   Tasks: {row['tasks'][:200]}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Genera tareas de desarrollo a partir de historias de usuario del dataset Salony",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python salony_pipeline.py salony_tasks.csv
  python salony_pipeline.py salony_tasks.csv --model mistral
  python salony_pipeline.py salony_tasks.csv --batch-size 4 --temperature 0.6
  python salony_pipeline.py salony_tasks.csv --sample 10  # Solo 10 historias

El script usa automáticamente el dataset: data/salony_train.csv
        """
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
        default=0.3,
        help='Temperatura para generación (default: 0.3)'
    )
    
    parser.add_argument(
        '--num-predict',
        type=int,
        default=1000,
        help='Tokens máximos a generar (default: 1000)'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Número de historias a procesar (útil para pruebas)'
    )
    
    args = parser.parse_args()
    
    try:
        run_salony_pipeline(
            output_csv=args.output_csv,
            model_name=args.model,
            batch_size=args.batch_size,
            temperature=args.temperature,
            num_predict=args.num_predict,
            sample_size=args.sample
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
