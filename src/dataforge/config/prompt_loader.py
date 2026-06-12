"""
Prompt template loader and renderer.

Templates use ``{{variable}}`` syntax for substitution — no external
templating engine required.  Files live under ``config/prompts/`` by
default and use a ``.j2`` extension.
"""

import re
from pathlib import Path
from typing import Dict, Optional


class PromptLoader:
    """Load and render ``.j2`` prompt templates with ``{{variable}}`` syntax."""

    _prompts_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @classmethod
    def initialize(cls, prompts_dir: Path) -> None:
        """Point the loader at a custom prompts directory."""
        cls._prompts_dir = prompts_dir

    @classmethod
    def get_prompts_dir(cls) -> Path:
        """Return the active prompts directory (lazy-defaults to ``config/prompts/``)."""
        if cls._prompts_dir is None:
            cls._prompts_dir = Path(__file__).parent / "prompts"
        return cls._prompts_dir

    # ------------------------------------------------------------------
    # Load & render
    # ------------------------------------------------------------------

    @classmethod
    def render(cls, template_name: str, context: Dict[str, str]) -> str:
        """
        Load a ``.j2`` template file and substitute ``{{variable}}`` placeholders.

        Parameters
        ----------
        template_name :
            File name (with or without ``.j2`` suffix).
        context :
            Mapping of variable names → replacement values.

        Returns
        -------
        Rendered prompt string.
        """
        template_path = cls._resolve_path(template_name)

        if not template_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {template_path}"
            )

        content = template_path.read_text(encoding="utf-8")

        def _replace(match: re.Match) -> str:
            var = match.group(1).strip()
            return str(context.get(var, match.group(0)))

        return re.sub(r"\{\{(.+?)\}\}", _replace, content)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_path(cls, template_name: str) -> Path:
        path = cls.get_prompts_dir() / template_name
        if not path.suffix:
            path = path.with_suffix(".j2")
        return path
