"""
Pydantic schemas for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""
    text: str = Field(..., description="Text to analyze for sentiment")
    metadata: Optional[Dict] = Field(default=None, description="Optional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely fantastic! I loved it.",
                "metadata": {"source": "user_review", "user_id": "12345"}
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    text: str
    sentiment: str = Field(..., description="Predicted sentiment: 'positive' or 'negative'")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    metadata: Optional[Dict] = None


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""
    texts: List[str] = Field(..., description="List of texts to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Great product! Highly recommend.",
                    "Terrible experience. Would not buy again.",
                    "It's okay, nothing special."
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
    version: Optional[str] = None
    model_loaded: Optional[bool] = None
    kafka_enabled: Optional[bool] = None
