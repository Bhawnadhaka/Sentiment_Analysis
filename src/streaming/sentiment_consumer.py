"""
Kafka Consumer - Processes news and makes sentiment predictions
"""
import os
import json
import time
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from loguru import logger
from dotenv import load_dotenv
from datetime import datetime
from src.models.inference import SentimentPredictor
import sqlite3

load_dotenv()


class SentimentConsumer:
    """
    Consumes news articles from Kafka and performs sentiment analysis.
    Stores results in database.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = None,
        topic: str = None,
        group_id: str = None
    ):
        self.bootstrap_servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.topic = topic or os.getenv("KAFKA_TOPIC", "news-sentiment")
        self.group_id = group_id or os.getenv("KAFKA_GROUP_ID", "sentiment-consumer-group")
        
        # Load model
        model_path = os.getenv("MODEL_PATH", "models/best_bow_model.pth")
        model_type = os.getenv("MODEL_TYPE", "bow")
        
        logger.info(f"Loading model: {model_path}")
        self.predictor = SentimentPredictor(model_path, model_type)
        logger.info("✓ Model loaded")
        
        # Setup database
        self.db_path = "sentiment_predictions.db"
        self.setup_database()
        
        # Connect to Kafka
        logger.info(f"Connecting to Kafka: {self.bootstrap_servers}")
        
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                max_poll_records=10
            )
            logger.info(f"✓ Connected to topic: {self.topic}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def setup_database(self):
        """Create SQLite database for storing predictions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                text TEXT,
                source TEXT,
                url TEXT UNIQUE,
                published_at TEXT,
                processed_at TEXT,
                sentiment TEXT,
                confidence REAL,
                prob_positive REAL,
                prob_negative REAL,
                topic TEXT,
                category TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"✓ Database ready: {self.db_path}")
    
    def save_prediction(self, article: dict, prediction: dict):
        """Save prediction to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO predictions 
                (title, text, source, url, published_at, processed_at, 
                 sentiment, confidence, prob_positive, prob_negative, topic, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article.get('title'),
                article.get('text'),
                article.get('source'),
                article.get('url'),
                article.get('published_at'),
                datetime.now().isoformat(),
                prediction['sentiment'],
                prediction['confidence'],
                prediction['probabilities'].get('positive', 0),
                prediction['probabilities'].get('negative', 0),
                article.get('topic'),
                article.get('category')
            ))
            
            conn.commit()
        except sqlite3.IntegrityError:
            logger.debug(f"Duplicate article skipped: {article.get('url')}")
        finally:
            conn.close()
    
    def process_article(self, article: dict) -> dict:
        """Process article and make sentiment prediction."""
        # Make prediction
        prediction = self.predictor.predict(article['text'])
        
        # Save to database
        self.save_prediction(article, prediction)
        
        return {
            **article,
            'prediction': prediction,
            'processed_at': datetime.now().isoformat()
        }
    
    def consume(self, max_messages: int = None):
        """
        Start consuming messages from Kafka.
        
        Args:
            max_messages: Stop after N messages (None = infinite)
        """
        message_count = 0
        
        logger.info("="*60)
        logger.info("Starting Kafka Consumer")
        logger.info(f"Topic: {self.topic}")
        logger.info(f"Group: {self.group_id}")
        logger.info("="*60)
        
        try:
            for message in self.consumer:
                article = message.value
                
                # Process article
                result = self.process_article(article)
                
                message_count += 1
                
                # Log result
                sentiment_emoji = "😊" if result['prediction']['sentiment'] == 'positive' else "😟"
                logger.info(
                    f"\n[{message_count}] {sentiment_emoji} {result['title'][:60]}..."
                )
                logger.info(
                    f"    Sentiment: {result['prediction']['sentiment'].upper()} "
                    f"({result['prediction']['confidence']:.2%} confidence)"
                )
                logger.info(f"    Source: {result['source']} | Topic: {result.get('topic', 'N/A')}")
                
                if max_messages and message_count >= max_messages:
                    logger.info(f"\n✓ Processed {message_count} messages")
                    break
                    
        except KeyboardInterrupt:
            logger.info("\n\nConsumer stopped by user")
        finally:
            self.close()
            logger.info(f"\n✓ Total messages processed: {message_count}")
    
    def close(self):
        """Close consumer connection."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
    
    def get_stats(self):
        """Get statistics from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]
        
        # Sentiment distribution
        cursor.execute("SELECT sentiment, COUNT(*) FROM predictions GROUP BY sentiment")
        sentiment_dist = dict(cursor.fetchall())
        
        # By topic
        cursor.execute("SELECT topic, COUNT(*) FROM predictions GROUP BY topic ORDER BY COUNT(*) DESC LIMIT 5")
        top_topics = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_predictions': total,
            'sentiment_distribution': sentiment_dist,
            'top_topics': top_topics
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Consume news from Kafka and predict sentiment")
    parser.add_argument('--max-messages', type=int, default=None,
                        help='Maximum messages to process (None = infinite)')
    parser.add_argument('--stats', action='store_true',
                        help='Show database statistics and exit')
    
    args = parser.parse_args()
    
    try:
        consumer = SentimentConsumer()
        
        if args.stats:
            stats = consumer.get_stats()
            print("\n" + "="*60)
            print("Database Statistics")
            print("="*60)
            print(f"Total Predictions: {stats['total_predictions']}")
            print(f"\nSentiment Distribution:")
            for sentiment, count in stats['sentiment_distribution'].items():
                print(f"  {sentiment}: {count}")
            print(f"\nTop Topics:")
            for topic, count in stats['top_topics']:
                print(f"  {topic}: {count}")
        else:
            consumer.consume(max_messages=args.max_messages)
            
    except Exception as e:
        logger.error(f"Consumer error: {e}")
        import traceback
        traceback.print_exc()
