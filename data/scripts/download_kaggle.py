"""
Download datasets from Kaggle.
Uses Kaggle API - FREE (requires account).

Setup:
1. Create Kaggle account at https://www.kaggle.com
2. Go to Account settings -> Create New API Token
3. Download kaggle.json
4. Place it in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)
"""

import os
import subprocess
from pathlib import Path
from loguru import logger
import pandas as pd


class KaggleDownloader:
    """Download datasets from Kaggle."""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent / "raw"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self, dataset_id: str, unzip: bool = True):
        """
        Download a Kaggle dataset.
        
        Args:
            dataset_id: Kaggle dataset identifier (e.g., 'username/dataset-name')
            unzip: Whether to unzip the downloaded file
        """
        logger.info(f"Downloading Kaggle dataset: {dataset_id}")
        
        try:
            cmd = [
                "kaggle", "datasets", "download",
                "-d", dataset_id,
                "-p", str(self.data_dir)
            ]
            
            if unzip:
                cmd.append("--unzip")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Downloaded successfully: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading dataset: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error(
                "Kaggle CLI not found. Install it with: pip install kaggle\n"
                "Then setup API credentials: https://github.com/Kaggle/kaggle-api#api-credentials"
            )
            raise


def main():
    """Download popular sentiment analysis datasets from Kaggle."""
    downloader = KaggleDownloader()
    
    # FREE Kaggle datasets for sentiment analysis
    datasets = [
        # IMDB Movie Reviews - 50K reviews (positive/negative)
        #"lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
        
        # Twitter Sentiment Analysis - 1.6M tweets
        "kazanova/sentiment140",
        
        # Amazon Reviews - Product reviews
        # "bittlingmayer/amazonreviews",
        
        # Yelp Reviews - Restaurant reviews
        # "yelp-dataset/yelp-dataset",
    ]
    
    for dataset_id in datasets:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {dataset_id}")
            logger.info(f"{'='*60}")
            
            downloader.download_dataset(dataset_id, unzip=True)
            
            logger.info(f"✓ Successfully downloaded {dataset_id}\n")
            
        except Exception as e:
            logger.error(f"✗ Failed to download {dataset_id}: {e}\n")
            continue
    
    # List downloaded files
    logger.info("\n📁 Downloaded files:")
    for file in downloader.data_dir.glob("*"):
        size_mb = file.stat().st_size / (1024 * 1024)
        logger.info(f"  - {file.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
