# tests/test_new_steps.py

import pandas as pd
import pytest

from simple_pipeline.steps import (
    FilterRows,
    SortRows,
    SampleRows,
    RobustOllamaStep
)


class TestFilterRows:
    """Tests for FilterRows step."""
    
    def test_filter_with_function(self):
        """Test filtering with custom function."""
        df = pd.DataFrame({
            'value': [1, 5, 10, 15, 20],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        
        step = FilterRows(
            name="filter",
            filter_func=lambda row: row['value'] > 10
        )
        
        result = step(df)
        assert len(result) == 2
        assert result['value'].tolist() == [15, 20]
    
    def test_filter_with_condition(self):
        """Test filtering with condition string."""
        df = pd.DataFrame({
            'age': [18, 25, 30, 15, 40],
            'name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve']
        })
        
        step = FilterRows(
            name="filter",
            filter_column="age",
            condition=">= 18"
        )
        
        result = step(df)
        assert len(result) == 4
        assert 'Dave' not in result['name'].values
    
    def test_filter_category(self):
        """Test filtering categorical data."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B'],
            'value': [1, 2, 3, 4, 5]
        })
        
        step = FilterRows(
            name="filter",
            filter_func=lambda row: row['category'] in ['A', 'B']
        )
        
        result = step(df)
        assert len(result) == 4
        assert 'C' not in result['category'].values


class TestSortRows:
    """Tests for SortRows step."""
    
    def test_sort_single_column(self):
        """Test sorting by single column."""
        df = pd.DataFrame({
            'value': [3, 1, 4, 1, 5],
            'letter': ['C', 'A', 'D', 'B', 'E']
        })
        
        step = SortRows(name="sort", by="value", ascending=True)
        result = step(df)
        
        assert result['value'].tolist() == [1, 1, 3, 4, 5]
    
    def test_sort_multiple_columns(self):
        """Test sorting by multiple columns."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [3, 1, 2, 4, 1]
        })
        
        step = SortRows(
            name="sort",
            by=["category", "value"],
            ascending=[True, False]
        )
        result = step(df)
        
        # Category A should be first, then sorted by value descending
        assert result.iloc[0]['category'] == 'A'
        assert result.iloc[0]['value'] == 3
    
    def test_sort_descending(self):
        """Test descending sort."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        
        step = SortRows(name="sort", by="value", ascending=False)
        result = step(df)
        
        assert result['value'].tolist() == [5, 4, 3, 2, 1]


class TestSampleRows:
    """Tests for SampleRows step."""
    
    def test_sample_n_rows(self):
        """Test sampling n rows."""
        df = pd.DataFrame({'value': range(100)})
        
        step = SampleRows(name="sample", n=10, random_state=42)
        result = step(df)
        
        assert len(result) == 10
        assert set(result['value']).issubset(set(range(100)))
    
    def test_sample_fraction(self):
        """Test sampling by fraction."""
        df = pd.DataFrame({'value': range(100)})
        
        step = SampleRows(name="sample", frac=0.1, random_state=42)
        result = step(df)
        
        assert len(result) == 10
    
    def test_sample_reproducible(self):
        """Test that sampling is reproducible with random_state."""
        df = pd.DataFrame({'value': range(100)})
        
        step1 = SampleRows(name="sample1", n=10, random_state=42)
        step2 = SampleRows(name="sample2", n=10, random_state=42)
        
        result1 = step1(df)
        result2 = step2(df)
        
        assert result1['value'].tolist() == result2['value'].tolist()
    
    def test_sample_error_no_params(self):
        """Test that error is raised when no params provided."""
        with pytest.raises(ValueError):
            SampleRows(name="sample")
    
    def test_sample_error_both_params(self):
        """Test that error is raised when both n and frac provided."""
        with pytest.raises(ValueError):
            SampleRows(name="sample", n=10, frac=0.1)


class TestRobustOllamaStep:
    """Tests for RobustOllamaStep."""
    
    def test_initialization(self):
        """Test that RobustOllamaStep can be initialized."""
        step = RobustOllamaStep(
            name="robust",
            model_name="llama3.2",
            prompt_column="input",
            output_column="output",
            save_failures=True
        )
        
        assert step.name == "robust"
        assert step.save_failures == True
        assert step.continue_on_error == True
        assert step.success_count == 0
        assert step.failure_count == 0
    
    def test_failure_tracking(self):
        """Test that failures are tracked."""
        step = RobustOllamaStep(
            name="robust",
            model_name="llama3.2",
            prompt_column="input",
            output_column="output"
        )
        
        # Initially no failures
        assert len(step.get_failed_rows()) == 0
        
        # After processing with errors, should have failures
        # (This would need actual Ollama to test fully)
    
    def test_get_failure_summary(self):
        """Test failure summary generation."""
        step = RobustOllamaStep(
            name="robust",
            model_name="llama3.2",
            prompt_column="input",
            output_column="output"
        )
        
        summary = step.get_failure_summary()
        
        assert 'total_failures' in summary
        assert 'total_successes' in summary
        assert 'error_types' in summary
        assert 'failure_rate' in summary


class TestPipelineIntegration:
    """Integration tests combining multiple steps."""
    
    def test_filter_sort_sample(self):
        """Test pipeline with filter, sort, and sample."""
        from simple_pipeline import SimplePipeline
        from simple_pipeline.steps import LoadDataFrame
        
        df = pd.DataFrame({
            'value': range(20),
            'category': ['A', 'B'] * 10
        })
        
        pipeline = SimplePipeline(name="test")
        pipeline.add_step(LoadDataFrame(name="load", df=df))
        pipeline.add_step(FilterRows(
            name="filter",
            filter_column="value",
            condition=">= 10"
        ))
        pipeline.add_step(SortRows(
            name="sort",
            by="value",
            ascending=False
        ))
        pipeline.add_step(SampleRows(
            name="sample",
            n=3,
            random_state=42
        ))
        
        result = pipeline.run(use_cache=False)
        
        assert len(result) == 3
        assert all(result['value'] >= 10)
