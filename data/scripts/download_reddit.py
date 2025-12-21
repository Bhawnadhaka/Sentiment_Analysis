"""
Download Reddit data for sentiment analysis.
Uses PRAW (Python Reddit API Wrapper) - FREE tier available.

Setup:
1. Create Reddit app at https://www.reddit.com/prefs/apps
2. Get client_id and client_secret
3. Set environment variables or update config
"""

import os
import praw
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class RedditScraper:
    """Scrape Reddit posts for sentiment analysis."""
    
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID", "your_client_id"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET", "your_client_secret"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "sentiment_analysis_bot/1.0"),
        )
        
    def scrape_subreddit(
        self, 
        subreddit_name: str, 
        limit: int = 1000,
        time_filter: str = "week"
    ) -> pd.DataFrame:
        """
        Scrape posts from a subreddit.
        
        Args:
            subreddit_name: Name of subreddit
            limit: Number of posts to fetch (max 1000 per request)
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'
            
        Returns:
            DataFrame with posts
        """
        logger.info(f"Scraping r/{subreddit_name}...")
        
        subreddit = self.reddit.subreddit(subreddit_name)
        posts_data = []
        
        # Fetch top posts
        for post in subreddit.top(time_filter=time_filter, limit=limit):
            posts_data.append({
                'post_id': post.id,
                'title': post.title,
                'text': post.selftext,
                'score': post.score,
                'num_comments': post.num_comments,
                'created_utc': datetime.fromtimestamp(post.created_utc),
                'subreddit': subreddit_name,
                'url': post.url,
                'author': str(post.author) if post.author else '[deleted]'
            })
            
        df = pd.DataFrame(posts_data)
        logger.info(f"Scraped {len(df)} posts from r/{subreddit_name}")
        return df
    
    def scrape_comments(self, post_id: str, limit: int = 100) -> pd.DataFrame:
        """Scrape comments from a specific post."""
        logger.info(f"Scraping comments from post {post_id}...")
        
        submission = self.reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)  # Remove MoreComments objects
        
        comments_data = []
        for comment in submission.comments.list()[:limit]:
            comments_data.append({
                'comment_id': comment.id,
                'post_id': post_id,
                'text': comment.body,
                'score': comment.score,
                'created_utc': datetime.fromtimestamp(comment.created_utc),
                'author': str(comment.author) if comment.author else '[deleted]'
            })
            
        df = pd.DataFrame(comments_data)
        logger.info(f"Scraped {len(df)} comments")
        return df


def main():
    """Download Reddit data from popular subreddits."""
    scraper = RedditScraper()
    
    # Subreddits for sentiment analysis (mix of positive/negative/neutral)
    subreddits = [
        'technology',
        'movies',
        'gaming',
        'worldnews',
        'AskReddit',
    ]
    
    all_posts = []
    
    for subreddit in subreddits:
        try:
            df = scraper.scrape_subreddit(
                subreddit_name=subreddit,
                limit=200,  # 200 posts per subreddit
                time_filter='week'
            )
            all_posts.append(df)
        except Exception as e:
            logger.error(f"Error scraping r/{subreddit}: {e}")
            continue
    
    # Combine all posts
    if all_posts:
        combined_df = pd.concat(all_posts, ignore_index=True)
        
        # Save to raw data folder
        output_path = Path(__file__).parent.parent / "raw" / "reddit_data.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(combined_df)} posts to {output_path}")
        logger.info(f"Data shape: {combined_df.shape}")
        logger.info(f"Columns: {combined_df.columns.tolist()}")
    else:
        logger.error("No data collected!")


if __name__ == "__main__":
    main()
