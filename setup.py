from setuptools import setup, find_packages

setup(
    name="simple-pipeline",
    version="0.1.0",
    author="Tu Nombre",
    description="A simplified Distilabel-inspired pipeline for DataFrame processing with Ollama",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "ollama>=0.1.0",
        "pydantic>=2.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)