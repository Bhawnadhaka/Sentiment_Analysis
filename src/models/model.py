"""
Sentiment Analysis Model Architecture.
Supports multiple model types: DistilBERT, LSTM, and baseline models.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from typing import Dict, Tuple


class DistilBERTSentiment(nn.Module):
    """
    DistilBERT-based sentiment classifier.
    Uses distilbert-base-uncased (lighter version of BERT).
    """
    
    def __init__(
        self,
        pretrained_model: str = "distilbert-base-uncased",
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_bert: bool = False
    ):
        super().__init__()
        
        # Load pretrained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: [batch_size, num_classes]
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class LSTMSentiment(nn.Module):
    """
    LSTM-based sentiment classifier.
    Lighter alternative to transformer models.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pretrained_embeddings: torch.Tensor = None
    ):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Use pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # Freeze embeddings
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output dimension based on bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            logits: [batch_size, num_classes]
        """
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state (for bidirectional, concatenate both directions)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Dropout and classification
        output = self.dropout(hidden)
        logits = self.fc(output)
        
        return logits


class SimpleBoWSentiment(nn.Module):
    """
    Simple Bag-of-Words baseline model.
    Uses average of word embeddings + feedforward network.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            logits: [batch_size, num_classes]
        """
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Average pooling over sequence length
        pooled = embedded.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


def create_model(model_type: str = "distilbert", **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 'distilbert', 'lstm', or 'bow'
        **kwargs: Model-specific arguments
        
    Returns:
        PyTorch model
    """
    if model_type == "distilbert":
        return DistilBERTSentiment(**kwargs)
    elif model_type == "lstm":
        return LSTMSentiment(**kwargs)
    elif model_type == "bow":
        return SimpleBoWSentiment(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("Testing model architectures...\n")
    
    # DistilBERT
    print("1. DistilBERT Model:")
    distilbert_model = create_model("distilbert", num_classes=2)
    print(f"   Parameters: {sum(p.numel() for p in distilbert_model.parameters()):,}")
    
    # LSTM
    print("\n2. LSTM Model:")
    lstm_model = create_model("lstm", vocab_size=10000, num_classes=2)
    print(f"   Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    # Bag-of-Words
    print("\n3. Bag-of-Words Model:")
    bow_model = create_model("bow", vocab_size=10000, num_classes=2)
    print(f"   Parameters: {sum(p.numel() for p in bow_model.parameters()):,}")
