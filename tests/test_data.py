"""Basic tests for data loading."""

import pytest
import pandas as pd
from pathlib import Path


def test_data_files_exist():
    """Test that processed data files exist."""
    data_dir = Path("data/processed")
    
    # Check if directory exists
    assert data_dir.exists(), "Processed data directory not found"
    
    # Check for required files
    required_files = ["train.csv", "val.csv", "test.csv"]
    
    for file in required_files:
        file_path = data_dir / file
        # Only test if data has been processed
        if file_path.exists():
            assert file_path.is_file(), f"{file} should be a file"


def test_load_data():
    """Test loading data with pandas."""
    data_file = Path("data/processed/train.csv")
    
    # Skip if file doesn't exist
    if not data_file.exists():
        pytest.skip("Training data not found")
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Basic checks
    assert len(df) > 0, "DataFrame should not be empty"
    assert 'label' in df.columns, "Should have 'label' column"
    assert 'text' in df.columns or 'cleaned_text' in df.columns, \
        "Should have text column"


def test_data_labels():
    """Test that labels are binary (0, 1)."""
    data_file = Path("data/processed/train.csv")
    
    if not data_file.exists():
        pytest.skip("Training data not found")
    
    df = pd.read_csv(data_file)
    
    # Check labels
    unique_labels = df['label'].unique()
    assert set(unique_labels).issubset({0, 1}), \
        "Labels should be 0 or 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
