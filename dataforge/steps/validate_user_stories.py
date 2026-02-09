"""
ValidateUserStories step for filtering user stories based on Agile format.

This step validates that user stories follow the standard format:
"As a(n) [role], I want [feature] so that [benefit]"
"""

import re
import pandas as pd
from typing import List

from dataforge.base_step import BaseStep
from dataforge.utils.logging import setup_logger


class ValidateUserStories(BaseStep):
    """
    Validates user stories against the standard Agile format.

    Filters out stories that don't match: "As a(n) [role], I want [feature] so that [benefit]"

    Args:
        name: Step name for logging and caching
        story_column: Column containing user stories to validate
        case_sensitive: Whether to enforce case-sensitive matching (default: False)
    """

    def __init__(
        self,
        name: str,
        story_column: str,
        case_sensitive: bool = False,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.story_column = story_column
        self.case_sensitive = case_sensitive
        self.logger = setup_logger(f"dataforge.steps.{name}")

        # Store regex pattern as string (pandas str.match needs string, not compiled pattern)
        self.pattern = r'^As\s+an?\s+(.+?),\s+I\s+want\s+(.+?)\s+so\s+that\s+(.+)$'

    @property
    def inputs(self) -> List[str]:
        """Required input columns."""
        return [self.story_column]

    @property
    def outputs(self) -> List[str]:
        """This step doesn't add new columns, only filters rows."""
        return []

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and filter user stories.

        Args:
            df: Input DataFrame with user stories

        Returns:
            Filtered DataFrame containing only valid user stories
        """
        total_stories = len(df)
        self.logger.info(f"Validating {total_stories} user stories...")

        # Normalize whitespace: collapse multiple spaces/newlines to single space
        normalized = df[self.story_column].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

        # Apply regex validation (case_insensitive flag for pandas str.match)
        valid_mask = normalized.str.match(self.pattern, case=self.case_sensitive, na=False)

        # Count valid and invalid stories
        valid_count = valid_mask.sum()
        invalid_count = total_stories - valid_count

        # Log statistics
        self.logger.info(f"Valid stories: {valid_count}/{total_stories} ({valid_count/total_stories*100:.1f}%)")
        self.logger.info(f"Invalid stories: {invalid_count}/{total_stories} ({invalid_count/total_stories*100:.1f}%)")

        # Log samples of invalid stories for debugging
        if invalid_count > 0:
            invalid_stories = df[~valid_mask][self.story_column].head(3).tolist()
            self.logger.warning(f"Sample invalid stories (first 3):")
            for i, story in enumerate(invalid_stories, 1):
                self.logger.warning(f"  {i}. {story}")

        # Filter to valid stories only
        result_df = df[valid_mask].copy()

        if len(result_df) == 0:
            self.logger.warning("All user stories were invalid! Returning empty DataFrame.")

        return result_df
