#!/usr/bin/env python
"""
CLI script to clear pipeline cache.
"""
import argparse
import shutil
from pathlib import Path


def clear_cache(cache_dir: str = ".cache/simple_pipeline"):
    """Clear all cached data."""
    cache_path = Path(cache_dir)
    
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"✓ Cleared cache directory: {cache_path}")
    else:
        print(f"No cache found at: {cache_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Clear Simple Pipeline cache"
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/simple_pipeline",
        help="Cache directory to clear"
    )
    
    args = parser.parse_args()
    clear_cache(args.cache_dir)


if __name__ == "__main__":
    main()