"""
Tests for ValidateUserStories step.
"""

import pandas as pd
import pytest
import re

from dataforge.steps.validate_user_stories import ValidateUserStories


class TestValidateUserStories:
    """Test suite for ValidateUserStories step."""

    def test_validates_correct_format(self):
        """Valid user stories should pass through unchanged."""
        df = pd.DataFrame({
            'story': [
                'As a user, I want to login so that I can access my account',
                'As an admin, I want to manage users so that I can control access',
                'As a developer, I want to write tests so that I can ensure quality'
            ]
        })

        step = ValidateUserStories(name="test", story_column="story")
        result = step.process(df)

        assert len(result) == 3
        assert list(result['story']) == list(df['story'])

    def test_filters_invalid_format(self):
        """Invalid user stories should be filtered out."""
        df = pd.DataFrame({
            'story': [
                'As a user, I want to login so that I can access my account',  # Valid
                'I want to login',  # Invalid - missing role and benefit
                'As a user',  # Invalid - incomplete
                'As an admin, I want to manage users so that I can control access',  # Valid
                'This is not a user story at all',  # Invalid
                'As a developer, I want to write tests'  # Invalid - missing "so that"
            ]
        })

        step = ValidateUserStories(name="test", story_column="story")
        result = step.process(df)

        assert len(result) == 2
        assert 'As a user, I want to login so that I can access my account' in result['story'].values
        assert 'As an admin, I want to manage users so that I can control access' in result['story'].values

    def test_case_insensitive_default(self):
        """By default, should accept various casings of 'as a'."""
        df = pd.DataFrame({
            'story': [
                'as a user, I want to login so that I can access my account',  # lowercase
                'As a user, I want to login so that I can access my account',  # normal
                'AS A user, I want to login so that I can access my account',  # uppercase
                'As A User, I Want To Login So That I Can Access My Account',  # title case
            ]
        })

        step = ValidateUserStories(name="test", story_column="story", case_sensitive=False)
        result = step.process(df)

        assert len(result) == 4

    def test_case_sensitive_mode(self):
        """With case_sensitive=True, lowercase 'as a' should fail."""
        df = pd.DataFrame({
            'story': [
                'as a user, I want to login so that I can access my account',  # lowercase - should fail
                'As a user, I want to login so that I can access my account',  # normal - should pass
            ]
        })

        step = ValidateUserStories(name="test", story_column="story", case_sensitive=True)
        result = step.process(df)

        assert len(result) == 1
        assert result['story'].iloc[0] == 'As a user, I want to login so that I can access my account'

    def test_handles_null_values(self):
        """NaN/null values should be treated as invalid."""
        df = pd.DataFrame({
            'story': [
                'As a user, I want to login so that I can access my account',
                None,
                pd.NA,
                'As an admin, I want to manage users so that I can control access',
            ]
        })

        step = ValidateUserStories(name="test", story_column="story")
        result = step.process(df)

        assert len(result) == 2
        assert pd.isna(result['story']).sum() == 0

    def test_normalizes_whitespace(self):
        """Extra spaces and newlines should be normalized before matching."""
        df = pd.DataFrame({
            'story': [
                'As a user,   I want to login   so that I can access my account',  # Extra spaces
                'As a user,\nI want to login\nso that I can access my account',  # Newlines
                'As  a  user,  I  want  to  login  so  that  I  can  access  my  account',  # Multiple spaces
            ]
        })

        step = ValidateUserStories(name="test", story_column="story")
        result = step.process(df)

        assert len(result) == 3

    def test_empty_dataframe_after_filter(self):
        """Should handle case where all stories are invalid without error."""
        df = pd.DataFrame({
            'story': [
                'Invalid story 1',
                'Invalid story 2',
                'Also not valid',
            ]
        })

        step = ValidateUserStories(name="test", story_column="story")
        result = step.process(df)

        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_logging_statistics(self, caplog):
        """Should log validation statistics correctly."""
        df = pd.DataFrame({
            'story': [
                'As a user, I want to login so that I can access my account',  # Valid
                'Invalid story',  # Invalid
                'As an admin, I want to manage users so that I can control access',  # Valid
            ]
        })

        step = ValidateUserStories(name="test", story_column="story")
        with caplog.at_level('INFO'):
            result = step.process(df)

        # Check that statistics were logged
        log_text = caplog.text
        assert 'Validating 3 user stories' in log_text
        assert 'Valid stories: 2/3' in log_text
        assert 'Invalid stories: 1/3' in log_text

    def test_handles_as_an_format(self):
        """Should accept both 'As a' and 'As an' formats."""
        df = pd.DataFrame({
            'story': [
                'As a user, I want to login so that I can access my account',  # "a"
                'As an admin, I want to manage users so that I can control access',  # "an"
                'As an engineer, I want to deploy code so that users get features',  # "an"
            ]
        })

        step = ValidateUserStories(name="test", story_column="story")
        result = step.process(df)

        assert len(result) == 3

    def test_inputs_property(self):
        """Should return the story column as required input."""
        step = ValidateUserStories(name="test", story_column="my_column")
        assert step.inputs == ["my_column"]

    def test_outputs_property(self):
        """Should return empty list as this step doesn't add columns."""
        step = ValidateUserStories(name="test", story_column="story")
        assert step.outputs == []

    def test_preserves_other_columns(self):
        """Should preserve other columns in the DataFrame."""
        df = pd.DataFrame({
            'story': [
                'As a user, I want to login so that I can access my account',
                'Invalid story',
            ],
            'id': [1, 2],
            'priority': ['high', 'low']
        })

        step = ValidateUserStories(name="test", story_column="story")
        result = step.process(df)

        assert len(result) == 1
        assert 'id' in result.columns
        assert 'priority' in result.columns
        assert result['id'].iloc[0] == 1
        assert result['priority'].iloc[0] == 'high'

    def test_logs_invalid_story_samples(self, caplog):
        """Should log samples of invalid stories for debugging."""
        df = pd.DataFrame({
            'story': [
                'As a user, I want to login so that I can access my account',  # Valid
                'Invalid story 1',  # Invalid
                'Invalid story 2',  # Invalid
                'Invalid story 3',  # Invalid
            ]
        })

        step = ValidateUserStories(name="test", story_column="story")
        with caplog.at_level('WARNING'):
            result = step.process(df)

        log_text = caplog.text
        assert 'Sample invalid stories' in log_text
        assert 'Invalid story 1' in log_text
        assert 'Invalid story 2' in log_text
        assert 'Invalid story 3' in log_text

    def test_converts_non_strings_to_strings(self):
        """Should handle non-string values by converting them to strings."""
        df = pd.DataFrame({
            'story': [
                'As a user, I want to login so that I can access my account',
                12345,  # Integer
                3.14,  # Float
            ]
        })

        step = ValidateUserStories(name="test", story_column="story")
        result = step.process(df)

        # Only the valid string should pass
        assert len(result) == 1
        assert result['story'].iloc[0] == 'As a user, I want to login so that I can access my account'
