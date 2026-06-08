#!/usr/bin/env python
"""
CLI script to run a pipeline.
"""
import argparse
import importlib.util
import sys


def load_pipeline_from_file(filepath: str):
    """Load and execute pipeline from Python file."""
    spec = importlib.util.spec_from_file_location("pipeline_module", filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_module"] = module
    spec.loader.exec_module(module)
    
    if hasattr(module, 'main'):
        module.main()
    else:
        print("Error: Pipeline file must have a 'main()' function")


def main():
    parser = argparse.ArgumentParser(
        description="Run a Simple Pipeline"
    )
    parser.add_argument(
        "pipeline_file",
        help="Path to pipeline Python file"
    )
    
    args = parser.parse_args()
    load_pipeline_from_file(args.pipeline_file)


if __name__ == "__main__":
    main()