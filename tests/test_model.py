"""Tests for model components."""

import pytest
import torch


def test_model_imports():
    """Test that model modules can be imported."""
    try:
        from src.models import model, train, evaluate, inference
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import model modules: {e}")


def test_create_model():
    """Test model creation."""
    from src.models.model import create_model
    
    # Test DistilBERT creation
    model = create_model("distilbert", num_classes=2)
    assert model is not None
    assert isinstance(model, torch.nn.Module)
    
    # Test LSTM creation
    model = create_model("lstm", vocab_size=1000, num_classes=2)
    assert model is not None
    
    # Test BoW creation
    model = create_model("bow", vocab_size=1000, num_classes=2)
    assert model is not None


def test_model_forward():
    """Test model forward pass."""
    from src.models.model import SimpleBoWSentiment
    
    model = SimpleBoWSentiment(vocab_size=1000, num_classes=2)
    
    # Create dummy input
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids)
    
    # Check output shape
    assert outputs.shape == (batch_size, 2), \
        f"Expected shape (4, 2), got {outputs.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
