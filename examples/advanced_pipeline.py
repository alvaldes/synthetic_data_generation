# examples/advanced_pipeline.py
"""
Ejemplo avanzado que muestra:
- Carga de datos
- Filtrado de filas
- Muestreo para prototipado rápido
- Generación con múltiples pasos LLM
- Manejo robusto de errores
- Ordenamiento de resultados
"""

import pandas as pd
from typing import Dict

from simple_pipeline.pipeline import SimplePipeline
from simple_pipeline.steps import (
    LoadDataFrame,
    FilterRows,
    SampleRows,
    RobustOllamaStep,
    AddColumn,
    SortRows,
    KeepColumns
)


# --------------------------
# 1. Crear dataset de ejemplo
# --------------------------
def create_sample_data():
    """Crea un dataset de tópicos técnicos con metadatos."""
    return pd.DataFrame({
        "topic": [
            "Machine Learning", 
            "Web Development", 
            "Data Structures",
            "Algorithms",
            "Database Design",
            "DevOps",
            "Cloud Computing",
            "Cybersecurity",
            "Python",
            "JavaScript"
        ],
        "category": [
            "AI", "Web", "CS", "CS", "Database",
            "Infrastructure", "Infrastructure", "Security",
            "Programming", "Programming"
        ],
        "difficulty": [
            "advanced", "intermediate", "intermediate", "advanced", "intermediate",
            "advanced", "intermediate", "advanced", "beginner", "beginner"
        ],
        "priority": [5, 3, 4, 5, 3, 4, 4, 5, 2, 2]
    })


# --------------------------
# 2. Crear pipeline completo
# --------------------------
def create_advanced_pipeline():
    """
    Crea un pipeline avanzado con múltiples pasos de procesamiento.
    """
    
    pipeline = SimplePipeline(
        name="advanced-synthetic-data",
        description="Pipeline avanzado para generación de datos sintéticos",
        log_level="INFO"
    )
    
    # Step 1: Cargar datos
    data = create_sample_data()
    pipeline.add_step(
        LoadDataFrame(name="load_data", df=data)
    )
    
    # Step 2: Filtrar solo tópicos de dificultad intermedia o avanzada
    pipeline.add_step(
        FilterRows(
            name="filter_difficulty",
            filter_func=lambda row: row['difficulty'] in ['intermediate', 'advanced']
        )
    )
    
    # Step 3: Ordenar por prioridad (descendente)
    pipeline.add_step(
        SortRows(
            name="sort_by_priority",
            by="priority",
            ascending=False
        )
    )
    
    # Step 4: Tomar una muestra para prototipado rápido
    # (Comenta esta línea para procesar todos los datos)
    pipeline.add_step(
        SampleRows(
            name="sample_for_testing",
            n=5,
            random_state=42
        )
    )
    
    # Step 5: Generar pregunta técnica
    def question_prompt(row: Dict) -> str:
        return (
            f"Generate a {row['difficulty']} level technical question about "
            f"{row['topic']} in the context of {row['category']}. "
            f"The question should be clear, specific, and suitable for a technical interview."
        )
    
    pipeline.add_step(
        RobustOllamaStep(
            name="generate_question",
            model_name="llama3.2",
            prompt_column="topic",
            output_column="question",
            prompt_template=question_prompt,
            system_prompt=(
                "You are an expert technical interviewer. "
                "Generate clear, challenging questions that test deep understanding."
            ),
            batch_size=2,
            generation_kwargs={
                "temperature": 0.8,
                "num_predict": 150
            },
            save_failures=True,
            continue_on_error=True
        )
    )
    
    # Step 6: Generar respuesta detallada
    def answer_prompt(row: Dict) -> str:
        return (
            f"Provide a comprehensive, detailed answer to this technical question:\n\n"
            f"{row['question']}\n\n"
            f"Include explanations, examples, and best practices where appropriate."
        )
    
    pipeline.add_step(
        RobustOllamaStep(
            name="generate_answer",
            model_name="llama3.2",
            prompt_column="question",
            output_column="answer",
            prompt_template=answer_prompt,
            system_prompt=(
                "You are a knowledgeable technical educator. "
                "Provide thorough, accurate answers with examples and explanations."
            ),
            batch_size=2,
            generation_kwargs={
                "temperature": 0.6,
                "num_predict": 400
            },
            save_failures=True,
            continue_on_error=True
        )
    )
    
    # Step 7: Añadir metadatos adicionales
    pipeline.add_step(
        AddColumn(
            name="add_metadata",
            input_columns=["question", "answer"],
            output_column="word_count",
            func=lambda q, a: len(str(q).split()) + len(str(a).split())
        )
    )
    
    # Step 8: Seleccionar columnas finales
    pipeline.add_step(
        KeepColumns(
            name="select_final_columns",
            columns=[
                "topic", 
                "category", 
                "difficulty", 
                "question", 
                "answer",
                "word_count",
                "status"  # Añadido por RobustOllamaStep
            ]
        )
    )
    
    return pipeline


# --------------------------
# 3. Ejecutar y analizar
# --------------------------
def main():
    """Ejecuta el pipeline avanzado y analiza resultados."""
    
    print("="*70)
    print("🚀 ADVANCED SYNTHETIC DATA PIPELINE")
    print("="*70)
    
    # Crear y ejecutar pipeline
    pipeline = create_advanced_pipeline()
    
    try:
        result_df = pipeline.run(use_cache=True)
        
        # Análisis de resultados
        print("\n" + "="*70)
        print("📊 RESULTS ANALYSIS")
        print("="*70)
        
        print(f"\nTotal records generated: {len(result_df)}")
        
        # Analizar éxito/fallo
        if 'status' in result_df.columns:
            success_count = (result_df['status'] == 'success').sum()
            failed_count = (result_df['status'] == 'failed').sum()
            print(f"\n✅ Successful: {success_count}")
            print(f"❌ Failed: {failed_count}")
        
        # Estadísticas de palabras
        if 'word_count' in result_df.columns:
            avg_words = result_df['word_count'].mean()
            max_words = result_df['word_count'].max()
            min_words = result_df['word_count'].min()
            print(f"\n📝 Word count stats:")
            print(f"   Average: {avg_words:.0f}")
            print(f"   Max: {max_words}")
            print(f"   Min: {min_words}")
        
        # Mostrar muestra
        print("\n" + "="*70)
        print("📄 SAMPLE RECORDS")
        print("="*70)
        print(result_df[['topic', 'difficulty', 'question']].head(3).to_string())
        
        # Guardar resultados
        output_file = "data/advanced_synthetic_dataset.csv"
        result_df.to_csv(output_file, index=False)
        print(f"\n💾 Full dataset saved to: {output_file}")
        
        # Guardar también en formato Excel si hay datos válidos
        valid_data = result_df[result_df['status'] == 'success']
        if len(valid_data) > 0:
            excel_file = "data/advanced_synthetic_dataset.xlsx"
            valid_data.to_excel(excel_file, index=False, engine='openpyxl')
            print(f"💾 Valid data saved to: {excel_file}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        raise
    
    print("\n" + "="*70)
    print("✨ Pipeline completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
