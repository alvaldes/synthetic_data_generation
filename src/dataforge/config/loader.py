"""
Hierarchical configuration loader.

Merge order (later overrides earlier):
  1. Pydantic model defaults (Python)
  2. Core defaults.yaml
  3. Use-case config.yaml (optional)
  4. Explicit file path (optional)

Result is cached in memory per use-case key.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from .base_config import DataForgeSettings

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_settings_cache: Dict[str, DataForgeSettings] = {}

_CONFIG_DIR = Path(__file__).parent
_USE_CASES_DIR = (
    _CONFIG_DIR.parent / "use_cases"
)
"""
Assumed layout:
  src/dataforge/config/             ← _CONFIG_DIR
  src/dataforge/use_cases/{name}/   ← sibling
"""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge *override* into *base* (mutates base)."""
    for key, val in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(val, dict)
        ):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Safely load a YAML file, returning empty dict on missing file."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _merge_yaml_into_settings(
    settings_dict: Dict[str, Any], yaml_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep-merge YAML data into a plain dict copy of settings."""
    merged = dict(settings_dict)
    _deep_merge(merged, yaml_data)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_settings(
    use_case: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> DataForgeSettings:
    """
    Build a fully-resolved :class:`DataForgeSettings` instance.

    Parameters
    ----------
    use_case :
        Short name matching ``src/dataforge/use_cases/{name}/``.
        When given, ``{use_case_dir}/config/config.yaml`` is merged on top of
        defaults.
    config_path :
        Explicit YAML path. Merged last — overrides everything else.

    Returns
    -------
    DataForgeSettings
    """
    # 1. Start from Pydantic defaults
    settings = DataForgeSettings()
    raw: Dict[str, Any] = settings.model_dump()

    # 2. Merge core defaults.yaml
    core_yaml = _load_yaml(_CONFIG_DIR / "defaults.yaml")
    raw = _merge_yaml_into_settings(raw, core_yaml)

    # 3. Merge use-case config.yaml
    if use_case is not None:
        uc_dir = _USE_CASES_DIR / use_case
        uc_config = uc_dir / "config" / "config.yaml"
        uc_yaml = _load_yaml(uc_config)
        raw = _merge_yaml_into_settings(raw, uc_yaml)

    # 4. Merge explicit file
    if config_path is not None:
        explicit_yaml = _load_yaml(config_path)
        raw = _merge_yaml_into_settings(raw, explicit_yaml)

    return DataForgeSettings(**raw)


def get_settings(
    use_case: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> DataForgeSettings:
    """
    Cached accessor — same as :func:`load_settings` but memoised per key.

    The cache key is *use_case* when provided, otherwise ``"__core__"``.
    """
    cache_key = use_case or "__core__"
    if cache_key not in _settings_cache:
        _settings_cache[cache_key] = load_settings(use_case, config_path)
    return _settings_cache[cache_key]


def clear_settings_cache() -> None:
    """Drop the in-memory settings cache (mostly useful in tests)."""
    _settings_cache.clear()
