"""
Dataset loaders for sentiment analysis.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import DistilBertTokenizer
from typing import Dict, List, Optional
from pathlib import Path


class SentimentDataset(Dataset):
    """Dataset for sentiment analysis with transformer tokenization."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class SimpleSentimentDataset(Dataset):
    """Simpler dataset for LSTM/BoW models (no transformer tokenization)."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Dict[str, int],
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs using vocabulary."""
        tokens = text.lower().split()
        token_ids = [self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in tokens]
        
        # Pad or truncate
        if len(token_ids) < self.max_length:
            token_ids += [0] * (self.max_length - len(token_ids))  # 0 is padding
        else:
            token_ids = token_ids[:self.max_length]
        
        return token_ids
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        token_ids = self.tokenize(text)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def build_vocab(texts: List[str], max_vocab_size: int = 10000) -> Dict[str, int]:
    """
    Build vocabulary from texts.
    
    Args:
        texts: List of text strings
        max_vocab_size: Maximum vocabulary size
        
    Returns:
        vocab: Dictionary mapping words to IDs
    """
    from collections import Counter
    
    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    # Get most common words
    most_common = word_counts.most_common(max_vocab_size - 2)  # Reserve for special tokens
    
    # Build vocab (0: <PAD>, 1: <UNK>, 2+: actual words)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        vocab[word] = idx
    
    return vocab


def load_data(
    data_dir: Path,
    batch_size: int = 32,
    model_type: str = "distilbert",
    max_length: int = 128,
    num_workers: int = 0
) -> tuple:
    """
    Load train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size
        model_type: 'distilbert', 'lstm', or 'bow'
        max_length: Maximum sequence length
        num_workers: Number of workers for DataLoader
        
    Returns:
        train_loader, val_loader, test_loader, (vocab if not distilbert)
    """
    # Load data
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    # Use 'cleaned_text' if available, otherwise 'text'
    text_col = 'cleaned_text' if 'cleaned_text' in train_df.columns else 'text'
    
    if model_type == "distilbert":
        # Use transformer tokenizer
        train_dataset = SentimentDataset(
            train_df[text_col].tolist(),
            train_df['label'].tolist(),
            max_length=max_length
        )
        val_dataset = SentimentDataset(
            val_df[text_col].tolist(),
            val_df['label'].tolist(),
            max_length=max_length
        )
        test_dataset = SentimentDataset(
            test_df[text_col].tolist(),
            test_df['label'].tolist(),
            max_length=max_length
        )
        vocab = None
    else:
        # Build vocabulary from training data
        vocab = build_vocab(train_df[text_col].tolist())
        
        train_dataset = SimpleSentimentDataset(
            train_df[text_col].tolist(),
            train_df['label'].tolist(),
            vocab,
            max_length=max_length
        )
        val_dataset = SimpleSentimentDataset(
            val_df[text_col].tolist(),
            val_df['label'].tolist(),
            vocab,
            max_length=max_length
        )
        test_dataset = SimpleSentimentDataset(
            test_df[text_col].tolist(),
            test_df['label'].tolist(),
            vocab,
            max_length=max_length
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    # Test data loading
    from pathlib import Path
    
    data_dir = Path("../../data/processed")
    
    if data_dir.exists():
        print("Testing data loading...")
        train_loader, val_loader, test_loader, vocab = load_data(
            data_dir,
            batch_size=8,
            model_type="distilbert"
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Get first batch
        batch = next(iter(train_loader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Labels shape: {batch['label'].shape}")
    else:
        print(f"Data directory not found: {data_dir}")
