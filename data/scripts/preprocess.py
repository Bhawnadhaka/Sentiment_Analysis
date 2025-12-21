"""
Data preprocessing pipeline for sentiment analysis.
Handles text cleaning, tokenization, and feature extraction.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from loguru import logger
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


class TextPreprocessor:
    """Preprocess text data for sentiment analysis."""
    
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Download required NLTK data
        self._download_nltk_data()
        
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()
    
    def _download_nltk_data(self):
        """Download required NLTK datasets."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading NLTK wordnet...")
            nltk.download('wordnet', quiet=True)
    
    def clean_text(self, text: str) -> str:
        """
        Clean text data.
        
        Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove HTML tags
        4. Remove special characters
        5. Remove extra whitespace
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove user mentions and hashtags (for Twitter data)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return word_tokenize(text)
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list."""
        if self.remove_stopwords:
            return [word for word in tokens if word not in self.stop_words]
        return tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        if self.lemmatize:
            return [self.lemmatizer.lemmatize(word) for word in tokens]
        return tokens
    
    def preprocess(self, text: str) -> str:
        """Complete preprocessing pipeline."""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Join back to string
        return ' '.join(tokens)


def load_imdb_data(data_dir: Path) -> pd.DataFrame:
    """Load Twitter Sentiment140 dataset."""
    logger.info("Loading Sentiment140 dataset...")
    
    csv_path = data_dir / "training.1600000.processed.noemoticon.csv"
    
    if not csv_path.exists():
        logger.error(f"Dataset not found at {csv_path}")
        return None
    
    # Twitter dataset has no header and uses Latin-1 encoding
    # Columns: [polarity, id, date, query, user, text]
    df = pd.read_csv(
        csv_path, 
        encoding='latin-1',
        header=None,
        names=['polarity', 'id', 'date', 'query', 'user', 'text']
    )
    
    # Convert polarity: 0=negative, 4=positive → 0=negative, 1=positive
    df['label'] = (df['polarity'] == 4).astype(int)
    
    # Keep only text and label
    df = df[['text', 'label']]
    
    logger.info(f"Loaded Twitter data: {df.shape[0]:,} tweets")
    logger.info(f"  Positive: {(df['label']==1).sum():,}, Negative: {(df['label']==0).sum():,}")
    return df


def preprocess_dataset(
    df: pd.DataFrame,
    text_column: str = 'text',
    label_column: str = 'label',
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess entire dataset.
    
    Returns:
        train_df, val_df, test_df
    """
    logger.info(f"Preprocessing dataset with {len(df)} samples...")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    # Preprocess text
    logger.info("Cleaning and preprocessing text...")
    df['cleaned_text'] = df[text_column].apply(preprocessor.preprocess)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
    
    logger.info(f"After preprocessing: {len(df)} samples")
    
    # Split data: train, validation, test
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df[label_column]
    )
    
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=train_df[label_column]
    )
    
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def main():
    """Main preprocessing pipeline."""
    # Paths
    data_dir = Path(__file__).parent.parent
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_imdb_data(raw_dir)
    
    if df is None:
        logger.error("No data loaded. Please download datasets first.")
        return
    
    # Sample for faster experimentation (remove for full dataset)
    # df = df.sample(n=10000, random_state=42)
    
    # Preprocess
    train_df, val_df, test_df = preprocess_dataset(df)
    
    # Save processed data
    logger.info("Saving processed data...")
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    
    logger.info("✓ Preprocessing complete!")
    logger.info(f"Processed data saved to: {processed_dir}")
    
    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("Dataset Statistics:")
    logger.info("="*60)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    logger.info(f"\nLabel distribution (train):")
    logger.info(train_df['label'].value_counts())


if __name__ == "__main__":
    main()
