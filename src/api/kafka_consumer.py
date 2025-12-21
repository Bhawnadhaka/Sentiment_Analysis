"""
Kafka Consumer for processing sentiment predictions.
Consumes messages from Kafka topics and stores them for monitoring.
"""

from kafka import KafkaConsumer
import json
from datetime import datetime
from loguru import logger
import signal
import sys
from typing import Callable, Optional
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os


Base = declarative_base()


class Prediction(Base):
    """Prediction model for database storage."""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    text = Column(Text)
    prediction = Column(String(50))
    confidence = Column(Float)
    metadata = Column(Text)  # JSON string


class KafkaConsumerClient:
    """Kafka consumer for sentiment predictions."""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "sentiment-predictions",
        group_id: str = "sentiment-consumer-group",
        db_url: str = None
    ):
        """
        Initialize Kafka consumer.
        
        Args:
            bootstrap_servers: Kafka broker address
            topic: Kafka topic to consume from
            group_id: Consumer group ID
            db_url: Database URL for storing predictions
        """
        self.topic = topic
        self.running = False
        
        # Setup database (optional)
        if db_url is None:
            db_url = os.getenv(
                "DATABASE_URL",
                "sqlite:///./sentiment_predictions.db"
            )
        
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        logger.info(f"Database initialized: {db_url}")
        
        # Setup Kafka consumer
        try:
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000
            )
            logger.info(f"Kafka consumer connected to {bootstrap_servers}")
            logger.info(f"Subscribed to topic: {topic}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise
    
    def process_message(self, message: dict):
        """
        Process a single message from Kafka.
        
        Args:
            message: Message dictionary
        """
        try:
            # Extract data
            timestamp_str = message.get('timestamp')
            text = message.get('text')
            prediction = message.get('prediction')
            confidence = message.get('confidence')
            metadata = message.get('metadata', {})
            
            # Parse timestamp
            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.utcnow()
            
            # Store in database
            pred_record = Prediction(
                timestamp=timestamp,
                text=text,
                prediction=prediction,
                confidence=confidence,
                metadata=json.dumps(metadata)
            )
            
            self.session.add(pred_record)
            self.session.commit()
            
            logger.info(
                f"Processed prediction: {prediction} "
                f"(confidence: {confidence:.4f}) - {text[:50]}..."
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.session.rollback()
    
    def consume(self, callback: Optional[Callable] = None):
        """
        Start consuming messages.
        
        Args:
            callback: Optional callback function to process messages
        """
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Starting to consume messages...")
        logger.info("Press Ctrl+C to stop")
        
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                msg_value = message.value
                
                # Process message
                if callback:
                    callback(msg_value)
                else:
                    self.process_message(msg_value)
                
        except Exception as e:
            logger.error(f"Error while consuming: {e}")
        finally:
            self.close()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("\nReceived shutdown signal. Closing consumer...")
        self.running = False
    
    def close(self):
        """Close consumer and database connection."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
        
        if self.session:
            self.session.close()
            logger.info("Database session closed")
    
    def get_recent_predictions(self, limit: int = 100):
        """Get recent predictions from database."""
        predictions = self.session.query(Prediction)\
            .order_by(Prediction.timestamp.desc())\
            .limit(limit)\
            .all()
        
        return [
            {
                'id': p.id,
                'timestamp': p.timestamp.isoformat(),
                'text': p.text,
                'prediction': p.prediction,
                'confidence': p.confidence,
                'metadata': json.loads(p.metadata) if p.metadata else {}
            }
            for p in predictions
        ]


if __name__ == "__main__":
    # Run consumer
    try:
        consumer = KafkaConsumerClient()
        
        logger.info("="*60)
        logger.info("Kafka Consumer Started")
        logger.info("="*60)
        
        # Start consuming
        consumer.consume()
        
    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        print("\nMake sure Kafka is running:")
        print("  docker-compose up -d kafka zookeeper")
