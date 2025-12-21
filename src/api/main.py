"""
FastAPI application for sentiment analysis.
Real-time predictions with Kafka streaming support.
Includes Prometheus metrics and Redis caching.
"""

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
from pathlib import Path
from loguru import logger
import os

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse
)
from src.models.inference import SentimentPredictor
from src.api.kafka_producer import KafkaProducerClient
from src.api.metrics import (
    track_request_metrics,
    get_metrics,
    model_loaded,
    kafka_connected,
    cache_hits_total,
    cache_misses_total
)
from src.api.cache import get_cache


# Global variables
model_predictor = None
kafka_producer = None
redis_cache = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global model_predictor, kafka_producer, redis_cache
    
    # Startup
    logger.info("Starting up application...")
    
    # Initialize Redis cache
    redis_cache = get_cache()
    
    # Load model
    model_path = os.getenv("MODEL_PATH", "models/best_bow_model.pth")
    model_type = os.getenv("MODEL_TYPE", "bow")
    
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading {model_type} model from {model_path}...")
    model_predictor = SentimentPredictor(model_path, model_type)
    logger.info("✓ Model loaded successfully!")
    
    # Update Prometheus metrics
    model_loaded.labels(model_type=model_type).set(1)
    
    # Initialize Kafka producer (optional)
    if os.getenv("KAFKA_ENABLED", "false").lower() == "true":
        kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        kafka_topic = os.getenv("KAFKA_TOPIC", "sentiment-predictions")
        
        try:
            kafka_producer = KafkaProducerClient(
                bootstrap_servers=kafka_bootstrap_servers,
                topic=kafka_topic
            )
            logger.info(f"Kafka producer initialized: {kafka_bootstrap_servers}")
            kafka_connected.set(1)
        except Exception as e:
            logger.warning(f"Kafka producer initialization failed: {e}")
            kafka_producer = None
            kafka_connected.set(0)
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    if kafka_producer:
        kafka_producer.close()
    if redis_cache:
        redis_cache.close()


# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Real-time sentiment analysis with MLOps best practices",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
@track_request_metrics("root")
async def root():
    """Root endpoint."""
    return {
        "status": "healthy",
        "message": "Sentiment Analysis API is running",
        "version": "1.0.0"
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=get_metrics(), media_type="text/plain")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = model_predictor is not None
    kafka_enabled = kafka_producer is not None
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "message": "API is running",
        "model_loaded": model_loaded,
        "kafka_enabled": kafka_enabled
    }


@app.post("/predict", response_model=PredictionResponse)
@track_request_metrics("predict")
async def predict_sentiment(request: PredictionRequest):
    """
    Predict sentiment for a single text.
    
    - **text**: Input text for sentiment analysis
    - Returns: Predicted sentiment (positive/negative) with confidence score
    """
    if model_predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model_type = os.getenv("MODEL_TYPE", "bow")
        
        # Check cache first
        cached_result = redis_cache.get(request.text, model_type) if redis_cache else None
        
        if cached_result:
            cache_hits_total.inc()
            logger.debug("Cache hit")
            result = cached_result
        else:
            cache_misses_total.inc()
            # Get prediction from model
            result = model_predictor.predict(request.text)
            
            # Store in cache
            if redis_cache:
                redis_cache.set(request.text, model_type, result)
        
        # Send to Kafka (if enabled)
        if kafka_producer:
            try:
                kafka_producer.send_prediction(
                    text=request.text,
                    prediction=result['sentiment'],
                    confidence=result['confidence'],
                    metadata=request.metadata
                )
            except Exception as e:
                logger.error(f"Failed to send to Kafka: {e}")
        
        return {
            "text": request.text,
            "sentiment": result['sentiment'],
            "confidence": result['confidence'],
            "probabilities": result['probabilities'],
            "metadata": request.metadata
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_sentiment_batch(request: BatchPredictionRequest):
    """
    Predict sentiment for multiple texts.
    
    - **texts**: List of input texts
    - Returns: List of predictions
    """
    if model_predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get predictions
        results = model_predictor.predict_batch(request.texts)
        
        predictions = []
        for text, result in zip(request.texts, results):
            predictions.append({
                "text": text,
                "sentiment": result['sentiment'],
                "confidence": result['confidence'],
                "probabilities": result['probabilities']
            })
            
            # Send to Kafka (if enabled)
            if kafka_producer:
                try:
                    kafka_producer.send_prediction(
                        text=text,
                        prediction=result['sentiment'],
                        confidence=result['confidence']
                    )
                except Exception as e:
                    logger.error(f"Failed to send to Kafka: {e}")
        
        return {"predictions": predictions}
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
@track_request_metrics("model_info")
async def model_info():
    """Get model information and cache statistics."""
    if model_predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    cache_stats = redis_cache.get_stats() if redis_cache else {"enabled": False}
    
    return {
        "model_type": model_predictor.model_type,
        "device": str(model_predictor.device),
        "classes": ["negative", "positive"],
        "cache": cache_stats
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
