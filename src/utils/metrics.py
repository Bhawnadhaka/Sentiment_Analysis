"""Custom metrics utilities."""

import time
from functools import wraps
from typing import Callable
from loguru import logger


def measure_time(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} took {execution_time:.4f} seconds")
        
        return result
    
    return wrapper


def calculate_metrics(y_true, y_pred) -> dict:
    """
    Calculate common classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }


if __name__ == "__main__":
    # Test decorator
    @measure_time
    def slow_function():
        time.sleep(1)
        return "Done"
    
    result = slow_function()
    print(result)
