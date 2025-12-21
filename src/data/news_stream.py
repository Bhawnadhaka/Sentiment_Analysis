"""
Live News Stream Ingestion
Uses News API (100 requests/day free)
Real-time sentiment analysis on breaking news
"""
import os
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Generator
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class NewsAPIStream:
    """
    Real-time news streaming from News API.
    
    Free tier: 100 requests/day
    Get API key: https://newsapi.org/register
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("NEWS_API_KEY not found. Get free key from https://newsapi.org/register")
        
        self.base_url = "https://newsapi.org/v2"
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})
        
        logger.info("✓ News API client initialized")
    
    def get_top_headlines(
        self,
        country: str = "us",
        category: str = None,
        sources: str = None,
        page_size: int = 20
    ) -> List[Dict]:
        """
        Get top headlines.
        
        Args:
            country: Country code (us, gb, in, etc.)
            category: business, entertainment, health, science, sports, technology
            sources: Comma-separated source IDs (e.g., 'bbc-news,cnn')
            page_size: Number of articles (max 100)
        """
        url = f"{self.base_url}/top-headlines"
        
        params = {
            'pageSize': page_size,
        }
        
        if country and not sources:
            params['country'] = country
        if category:
            params['category'] = category
        if sources:
            params['sources'] = sources
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'text': f"{article.get('title', '')}. {article.get('description', '')}",
                    'title': article.get('title'),
                    'source': article.get('source', {}).get('name'),
                    'url': article.get('url'),
                    'published_at': article.get('publishedAt'),
                    'timestamp': datetime.now().isoformat(),
                    'category': category or 'general',
                    'country': country
                })
            
            logger.info(f"Fetched {len(articles)} articles")
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []
    
    def search_everything(
        self,
        query: str,
        from_date: str = None,
        sort_by: str = "publishedAt",
        page_size: int = 20
    ) -> List[Dict]:
        """
        Search all articles.
        
        Args:
            query: Search keywords (e.g., 'Tesla', 'Bitcoin', 'AI')
            from_date: YYYY-MM-DD format
            sort_by: relevancy, popularity, publishedAt
            page_size: Number of results (max 100)
        """
        url = f"{self.base_url}/everything"
        
        if not from_date:
            from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'from': from_date,
            'sortBy': sort_by,
            'pageSize': page_size,
            'language': 'en'
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'text': f"{article.get('title', '')}. {article.get('description', '')}",
                    'title': article.get('title'),
                    'source': article.get('source', {}).get('name'),
                    'url': article.get('url'),
                    'published_at': article.get('publishedAt'),
                    'timestamp': datetime.now().isoformat(),
                    'query': query
                })
            
            logger.info(f"Found {len(articles)} articles for '{query}'")
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []
    
    def stream_continuous(
        self,
        topics: List[str] = None,
        interval_seconds: int = 300,
        max_iterations: int = None
    ) -> Generator[Dict, None, None]:
        """
        Continuous streaming of news articles.
        
        Args:
            topics: List of topics to track (e.g., ['Tesla', 'Bitcoin', 'Apple'])
            interval_seconds: Delay between API calls (min 300 for free tier)
            max_iterations: Stop after N iterations (None = infinite)
        """
        topics = topics or ['technology', 'business', 'politics']
        iteration = 0
        seen_urls = set()
        
        logger.info(f"Starting continuous stream for topics: {topics}")
        logger.info(f"Interval: {interval_seconds}s | Max iterations: {max_iterations or 'infinite'}")
        
        while True:
            if max_iterations and iteration >= max_iterations:
                logger.info("Reached max iterations")
                break
            
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration} - {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"{'='*60}")
            
            for topic in topics:
                articles = self.search_everything(
                    query=topic,
                    sort_by='publishedAt',
                    page_size=10
                )
                
                for article in articles:
                    # Skip duplicates
                    if article['url'] in seen_urls:
                        continue
                    
                    seen_urls.add(article['url'])
                    article['topic'] = topic
                    article['iteration'] = iteration
                    
                    yield article
                
                time.sleep(2)  # Rate limiting between requests
            
            if max_iterations is None or iteration < max_iterations:
                logger.info(f"\nWaiting {interval_seconds}s until next fetch...")
                time.sleep(interval_seconds)


if __name__ == "__main__":
    print("="*60)
    print("Testing News API Stream")
    print("="*60)
    
    # Check for API key
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        print("\n❗ NEWS_API_KEY not found!")
        print("\nTo get started:")
        print("1. Go to: https://newsapi.org/register")
        print("2. Sign up for free (no credit card needed)")
        print("3. Copy your API key")
        print("4. Add to .env file:")
        print("   NEWS_API_KEY=your_api_key_here")
        print("\nFree tier: 100 requests/day")
        exit(1)
    
    try:
        stream = NewsAPIStream(api_key)
        
        print("\n📰 Fetching top headlines...")
        headlines = stream.get_top_headlines(country='us', category='technology', page_size=5)
        
        for i, article in enumerate(headlines, 1):
            print(f"\n[{i}] {article['title']}")
            print(f"    Source: {article['source']}")
            print(f"    Text: {article['text'][:150]}...")
        
        print("\n" + "="*60)
        print("✓ News API working! Ready for streaming.")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
