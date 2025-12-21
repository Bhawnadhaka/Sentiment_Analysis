"""
Prometheus metrics for FastAPI application
Tracks API performance, model inference, and system health
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from functools import wraps
import time


# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# Model metrics
prediction_requests_total = Counter(
    'prediction_requests_total',
    'Total prediction requests',
    ['model_type']
)

prediction_duration_seconds = Histogram(
    'prediction_duration_seconds',
    'Model prediction latency',
    ['model_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

predictions_by_sentiment = Counter(
    'predictions_by_sentiment_total',
    'Total predictions by sentiment',
    ['sentiment', 'model_type']
)

model_confidence = Histogram(
    'model_confidence',
    'Model prediction confidence scores',
    ['sentiment', 'model_type'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

# Cache metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits'
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses'
)

# System metrics
model_loaded = Gauge(
    'model_loaded',
    'Model loaded status (1=loaded, 0=not loaded)',
    ['model_type']
)

kafka_connected = Gauge(
    'kafka_connected',
    'Kafka connection status (1=connected, 0=disconnected)'
)


def track_request_metrics(endpoint: str):
    """Decorator to track request metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                http_requests_total.labels(
                    method="POST" if "predict" in endpoint else "GET",
                    endpoint=endpoint,
                    status=status
                ).inc()
                http_request_duration_seconds.labels(
                    method="POST" if "predict" in endpoint else "GET",
                    endpoint=endpoint
                ).observe(duration)
        
        return wrapper
    return decorator


def track_prediction_metrics(model_type: str):
    """Decorator to track prediction metrics"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Increment request counter
            prediction_requests_total.labels(model_type=model_type).inc()
            
            # Call the actual prediction
            result = func(*args, **kwargs)
            
            # Track duration
            duration = time.time() - start_time
            prediction_duration_seconds.labels(model_type=model_type).observe(duration)
            
            # Track sentiment distribution
            if isinstance(result, dict):
                sentiment = result.get('sentiment', 'unknown')
                confidence = result.get('confidence', 0.0)
                
                predictions_by_sentiment.labels(
                    sentiment=sentiment,
                    model_type=model_type
                ).inc()
                
                model_confidence.labels(
                    sentiment=sentiment,
                    model_type=model_type
                ).observe(confidence)
            
            return result
        
        return wrapper
    return decorator


def get_metrics():
    """Get current metrics in Prometheus format"""
    return generate_latest()
