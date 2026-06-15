"""
Pydantic models for DataForge configuration.

All hardcoded defaults live here as Python constants. The YAML files
override these at load time via hierarchical merge.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class LLMConfig(BaseModel):
    """Configuration for LLM generation steps."""

    default_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"
    max_retries: int = 3
    batch_size: int = 8
    temperature: float = 0.3
    num_predict: int = 1000
    num_workers: int = 1


class JudgeCriterion(BaseModel):
    """A single criterion for judge evaluation."""

    key: str
    label_es: str
    label_en: str


class JudgeConfig(BaseModel):
    """Configuration for the single-model judge (OllamaJudgeStep)."""

    model: Optional[str] = None
    approval_threshold: float = 35.0
    max_score: int = 50
    batch_size: int = 4
    temperature: float = 0.2
    num_predict: int = 800
    num_workers: int = 1
    column_prefix: str = "validacion_"
    criteria: List[JudgeCriterion] = [
        JudgeCriterion(key="coherence", label_es="Coherencia", label_en="Coherence"),
        JudgeCriterion(key="completeness", label_es="Completitud", label_en="Completeness"),
        JudgeCriterion(key="feasibility", label_es="Viabilidad", label_en="Feasibility"),
        JudgeCriterion(key="format", label_es="Formato", label_en="Format"),
        JudgeCriterion(key="granularity", label_es="Granularidad", label_en="Granularity"),
    ]


class ComparisonJudgeConfig(BaseModel):
    """Configuration for the comparison judge (ComparisonJudgeStep)."""

    model: Optional[str] = None
    batch_size: int = 1
    temperature: float = 0.2
    num_predict: int = 1000
    num_workers: int = 1
    column_prefix: str = "judge_"


class PipelineConfig(BaseModel):
    """Configuration for the pipeline orchestrator."""

    cache_dir: str = ".cache/dataforge"
    log_level: str = "INFO"
    description: str = ""


class PathsConfig(BaseModel):
    """File-system paths for data, cache, and outputs."""

    data_dir: str = "data"
    raw_dir: str = "data/raw"
    output_dir: str = "data/outputs"


class DataForgeSettings(BaseModel):
    """Top-level settings combining all sub-configs."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    comparison_judge: ComparisonJudgeConfig = Field(default_factory=ComparisonJudgeConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
