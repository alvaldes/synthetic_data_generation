# simple_pipeline/utils/logging.py

import logging

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configura un logger con formato estándar para el pipeline.
    
    Args:
        name: nombre del logger (ej. "pipeline" o "ollama_step")
        level: nivel de logging (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    return logger