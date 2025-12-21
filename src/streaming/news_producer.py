"""
Kafka Producer - Streams news articles to Kafka
"""
import os
import json
import time
from kafka import KafkaProducer
from kafka.errors import KafkaError
from loguru import logger
from dotenv import load_dotenv
from src.data.news_stream import NewsAPIStream

load_dotenv()


class NewsKafkaProducer:
    """
    Produces news articles to Kafka topic for real-time processing.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = None,
        topic: str = None
    ):
        self.bootstrap_servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.topic = topic or os.getenv("KAFKA_TOPIC", "news-sentiment")
        
        logger.info(f"Connecting to Kafka: {self.bootstrap_servers}")
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            logger.info(f"✓ Connected to Kafka topic: {self.topic}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def send_article(self, article: dict) -> bool:
        """Send single article to Kafka."""
        try:
            # Use article URL as key for partitioning
            key = article.get('url', '').split('/')[-1]
            
            future = self.producer.send(
                self.topic,
                key=key,
                value=article
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            
            logger.debug(
                f"Sent: {article.get('title', 'No title')[:50]}... "
                f"(partition={record_metadata.partition}, offset={record_metadata.offset})"
            )
            
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to send article: {e}")
            return False
    
    def stream_news(
        self,
        topics: list = None,
        interval_seconds: int = 300,
        max_articles: int = None
    ):
        """
        Stream news articles continuously to Kafka.
        
        Args:
            topics: Topics to track (e.g., ['Tesla', 'Bitcoin', 'AI'])
            interval_seconds: Delay between API calls
            max_articles: Stop after N articles (None = infinite)
        """
        topics = topics or ['technology', 'business', 'AI']
        news_stream = NewsAPIStream()
        
        article_count = 0
        
        logger.info("="*60)
        logger.info(f"Starting News → Kafka Streaming")
        logger.info(f"Topics: {topics}")
        logger.info(f"Interval: {interval_seconds}s")
        logger.info(f"Target: {max_articles or 'unlimited'} articles")
        logger.info("="*60)
        
        try:
            for article in news_stream.stream_continuous(
                topics=topics,
                interval_seconds=interval_seconds,
                max_iterations=max_articles
            ):
                success = self.send_article(article)
                
                if success:
                    article_count += 1
                    logger.info(
                        f"[{article_count}] Streamed: {article['title'][:60]}... "
                        f"(Topic: {article.get('topic', 'N/A')})"
                    )
                
                if max_articles and article_count >= max_articles:
                    break
                
        except KeyboardInterrupt:
            logger.info("\n\nStreaming stopped by user")
        finally:
            self.close()
            logger.info(f"\n✓ Total articles streamed: {article_count}")
    
    def close(self):
        """Close producer connection."""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stream news to Kafka")
    parser.add_argument('--topics', nargs='+', default=['technology', 'AI', 'startups'],
                        help='Topics to track')
    parser.add_argument('--interval', type=int, default=300,
                        help='Seconds between API calls (min 300 for free tier)')
    parser.add_argument('--max-articles', type=int, default=None,
                        help='Maximum articles to stream (None = infinite)')
    
    args = parser.parse_args()
    
    try:
        producer = NewsKafkaProducer()
        producer.stream_news(
            topics=args.topics,
            interval_seconds=args.interval,
            max_articles=args.max_articles
        )
    except Exception as e:
        logger.error(f"Producer error: {e}")
        import traceback
        traceback.print_exc()
