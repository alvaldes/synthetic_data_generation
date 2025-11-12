#!/usr/bin/env python3
"""
Quick test script for the dual generator pipeline.

This creates a small test dataset and runs it through the pipeline to verify everything works.
"""

import pandas as pd
from pathlib import Path
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_test_data():
    """Create a small test dataset for validation."""
    test_stories = [
        "As a user, I want to be able to login to the system so that I can access my account.",
        "As an admin, I want to be able to manage user accounts so that I can control system access.",
        "As a user, I want to reset my password so that I can regain access if I forget it."
    ]

    return pd.DataFrame({"input": test_stories})

def main():
    print("🧪 Testing Dual Generator Pipeline")
    print("=" * 50)

    # Create test data
    test_df = create_test_data()
    print(f"Created test dataset with {len(test_df)} user stories")

    # Save to temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_input_path = f.name
        test_df.to_csv(test_input_path, index=False)

    # Create output path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_output_path = f.name

    try:
        # Import and run the pipeline
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from examples.salony_dual_generator_pipeline import run_dual_generator_pipeline

        print(f"Input file: {test_input_path}")
        print(f"Output file: {test_output_path}")

        # Run with minimal settings for quick test
        result_df = run_dual_generator_pipeline(
            output_csv=test_output_path,
            input_csv=test_input_path,
            model_a="llama3.1:8b",      # Adjust based on your available models
            model_b="llama3.1:8b",      # Using same model but different temps
            judge_model="llama3.1:8b",
            batch_size=1,               # Small batches for testing
            temperature_a=0.3,
            temperature_b=0.7,          # Different temperature for variation
            num_predict=500,            # Shorter responses for faster testing
            sample_size=None,           # Process all test data
            use_cache=True              # Use cache to speed up repeated runs
        )

        print("\n✅ Pipeline completed successfully!")
        print(f"Processed {len(result_df)} stories")

        # Show results
        print("\n📊 Results Summary:")
        print(result_df[['judge_winner', 'judge_score_a', 'judge_score_b']].describe())

        # Show sample
        if len(result_df) > 0:
            print("\n📝 Sample Result:")
            sample = result_df.iloc[0]
            print(f"User Story: {sample['input']}")
            print(f"Winner: Generator {sample['judge_winner']}")
            print(f"Scores: A={sample['judge_score_a']}, B={sample['judge_score_b']}")
            print(f"Reason: {sample['judge_reason']}")
            # Show the winning output based on judge_winner
            if sample['judge_winner'] == 'B':
                winning_output = sample['tasks_generator_b']
                print(f"Winning Output (Generator B): {winning_output[:200]}...")
            else:
                winning_output = sample['tasks_generator_a']
                print(f"Winning Output (Generator A): {winning_output[:200]}...")

        print(f"\n💾 Full results saved to: {test_output_path}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Clean up temporary files
        Path(test_input_path).unlink(missing_ok=True)
        # Keep output file for inspection: Path(test_output_path).unlink(missing_ok=True)

    return 0

if __name__ == "__main__":
    exit(main())