"""
Kafka Producer for sending predictions to Kafka topics.

For local development with FREE Kafka:
- Use Docker: docker-compose up kafka zookeeper
- Or use Confluent Cloud Free tier: https://www.confluent.io/confluent-cloud/
"""

from kafka import KafkaProducer
import json
from datetime import datetime
from typing import Dict, Optional
from loguru import logger


class KafkaProducerClient:
    """Kafka producer for sentiment predictions."""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "sentiment-predictions"
    ):
        """
        Initialize Kafka producer.
        
        Args:
            bootstrap_servers: Kafka broker address
            topic: Kafka topic to send messages to
        """
        self.topic = topic
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            logger.info(f"Kafka producer connected to {bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def send_prediction(
        self,
        text: str,
        prediction: str,
        confidence: float,
        metadata: Optional[Dict] = None
    ):
        """
        Send prediction to Kafka topic.
        
        Args:
            text: Input text
            prediction: Predicted sentiment
            confidence: Prediction confidence
            metadata: Additional metadata
        """
        message = {
            'timestamp': datetime.utcnow().isoformat(),
            'text': text,
            'prediction': prediction,
            'confidence': confidence,
            'metadata': metadata or {}
        }
        
        try:
            # Send to Kafka
            future = self.producer.send(
                self.topic,
                value=message,
                key=str(datetime.utcnow().timestamp())
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            
            logger.debug(
                f"Message sent to {record_metadata.topic} "
                f"partition {record_metadata.partition} "
                f"offset {record_metadata.offset}"
            )
            
        except Exception as e:
            logger.error(f"Failed to send message to Kafka: {e}")
            raise
    
    def send_batch(self, messages: list):
        """Send multiple messages to Kafka."""
        for msg in messages:
            self.send_prediction(**msg)
    
    def close(self):
        """Close Kafka producer."""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")


if __name__ == "__main__":
    # Test Kafka producer
    try:
        producer = KafkaProducerClient()
        
        # Send test message
        producer.send_prediction(
            text="This is a test message",
            prediction="positive",
            confidence=0.95,
            metadata={"source": "test"}
        )
        
        print("✓ Test message sent successfully!")
        producer.close()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure Kafka is running:")
        print("  docker-compose up -d kafka zookeeper")
