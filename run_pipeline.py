"""
Complete Real-Time Sentiment Analysis Pipeline
News API → Kafka → Sentiment Prediction → Database → Dashboard
"""
import os
import sys
import time
import argparse
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.streaming.news_producer import NewsKafkaProducer
from src.streaming.sentiment_consumer import SentimentConsumer
from src.monitoring.dashboard import SentimentMonitor


def check_kafka_ready(bootstrap_servers: str = "localhost:9092", timeout: int = 30):
    """Check if Kafka is ready."""
    from kafka import KafkaAdminClient
    from kafka.errors import NoBrokersAvailable
    
    logger.info(f"Checking Kafka connection: {bootstrap_servers}")
    
    for i in range(timeout):
        try:
            admin = KafkaAdminClient(
                bootstrap_servers=bootstrap_servers,
                client_id='health_check'
            )
            admin.close()
            logger.info("✓ Kafka is ready!")
            return True
        except NoBrokersAvailable:
            if i < timeout - 1:
                logger.info(f"Waiting for Kafka... ({i+1}/{timeout})")
                time.sleep(1)
            else:
                logger.error("✗ Kafka not available")
                return False
    
    return False


def run_producer(topics: list, max_articles: int = 50):
    """Run news producer."""
    logger.info("="*60)
    logger.info("Starting News Producer")
    logger.info("="*60)
    
    try:
        producer = NewsKafkaProducer()
        producer.stream_news(
            topics=topics,
            interval_seconds=300,  # 5 minutes between API calls
            max_articles=max_articles
        )
    except KeyboardInterrupt:
        logger.info("\nProducer stopped by user")
    except Exception as e:
        logger.error(f"Producer error: {e}")
        import traceback
        traceback.print_exc()


def run_consumer(max_messages: int = None, show_stats: bool = False):
    """Run sentiment consumer."""
    logger.info("="*60)
    logger.info("Starting Sentiment Consumer")
    logger.info("="*60)
    
    try:
        consumer = SentimentConsumer()
        
        if show_stats:
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
                print(f"  {topic}: {count} articles")
        else:
            consumer.consume(max_messages=max_messages)
            
    except KeyboardInterrupt:
        logger.info("\nConsumer stopped by user")
    except Exception as e:
        logger.error(f"Consumer error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time News Sentiment Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start producer (streams news to Kafka)
  python run_pipeline.py producer --topics technology AI startups --articles 50
  
  # Start consumer (processes and predicts sentiment)
  python run_pipeline.py consumer
  
  # Show statistics
  python run_pipeline.py consumer --stats
  
  # Generate monitoring dashboard
  python run_pipeline.py dashboard
  
  # Check Kafka connection
  python run_pipeline.py check
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Producer command
    producer_parser = subparsers.add_parser('producer', help='Run news producer')
    producer_parser.add_argument('--topics', nargs='+', 
                                  default=['technology', 'AI', 'startups'],
                                  help='Topics to stream')
    producer_parser.add_argument('--articles', type=int, default=50,
                                  help='Max articles to stream')
    
    # Consumer command
    consumer_parser = subparsers.add_parser('consumer', help='Run sentiment consumer')
    consumer_parser.add_argument('--max-messages', type=int, default=None,
                                  help='Max messages to process')
    consumer_parser.add_argument('--stats', action='store_true',
                                  help='Show statistics only')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check Kafka connection')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Generate Evidently monitoring reports')
    dashboard_parser.add_argument('--db', default='sentiment_predictions.db',
                                   help='Path to SQLite database')
    dashboard_parser.add_argument('--output', default='monitoring/evidently_reports',
                                   help='Output directory for reports')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Check command
    if args.command == 'check':
        if check_kafka_ready():
            print("\n✓ Kafka is running and ready!")
            print("\nNext steps:")
            print("1. Start consumer: python run_pipeline.py consumer")
            print("2. Start producer: python run_pipeline.py producer --topics technology AI")
        else:
            print("\n✗ Kafka is not ready.")
            print("\nMake sure Kafka is running:")
            print("  cd docker")
            print("  docker-compose up -d zookeeper kafka")
        return
    
    # Producer command
    if args.command == 'producer':
        if not check_kafka_ready(timeout=5):
            logger.error("Kafka not available. Start it first!")
            return
        
        run_producer(topics=args.topics, max_articles=args.articles)
    
    # Consumer command
    elif args.command == 'consumer':
        if not args.stats and not check_kafka_ready(timeout=5):
            logger.error("Kafka not available. Start it first!")
            return
        
        run_consumer(max_messages=args.max_messages, show_stats=args.stats)
    
    # Dashboard command
    elif args.command == 'dashboard':
        monitor = SentimentMonitor(db_path=args.db, output_dir=args.output)
        monitor.generate_all_reports()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
