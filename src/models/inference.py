"""
Model inference module.
"""

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer
from typing import Dict, List
import numpy as np
from pathlib import Path

from src.models.model import create_model


class SentimentPredictor:
    """Sentiment prediction class."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "distilbert",
        device: str = None
    ):
        self.model_type = model_type
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        self.vocab = checkpoint.get('vocab', None)
        
        # Create model
        model_config = self.config['model'].copy()
        
        # Remove 'type' key as it's not a model parameter
        model_config.pop('type', None)
        
        # Remove DistilBERT-specific params for other models
        if model_type != "distilbert":
            model_config.pop('freeze_bert', None)
        
        # Remove LSTM-specific params for BoW
        if model_type == "bow":
            model_config.pop('num_layers', None)
            model_config.pop('bidirectional', None)
        
        if self.vocab is not None and model_type != "distilbert":
            model_config['vocab_size'] = len(self.vocab)
        
        self.model = create_model(model_type, **model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Setup tokenizer
        self.max_length = self.config['training'].get('max_length', 128)
        
        if model_type == "distilbert":
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        else:
            self.tokenizer = None
        
        # Label mapping
        self.id2label = {0: "negative", 1: "positive"}
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess text for model input."""
        if self.model_type == "distilbert":
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].to(self.device),
                'attention_mask': encoding['attention_mask'].to(self.device)
            }
        else:
            # Simple tokenization for LSTM/BoW
            tokens = text.lower().split()
            token_ids = [self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in tokens]
            
            # Pad or truncate
            if len(token_ids) < self.max_length:
                token_ids += [0] * (self.max_length - len(token_ids))
            else:
                token_ids = token_ids[:self.max_length]
            
            return {
                'input_ids': torch.tensor([token_ids], dtype=torch.long).to(self.device)
            }
    
    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a single text.
        
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        inputs = self.preprocess_text(text)
        
        with torch.no_grad():
            if self.model_type == "distilbert":
                outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            else:
                outputs = self.model(inputs['input_ids'])
            
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        predicted_label = self.id2label[predicted.item()]
        confidence_score = confidence.item()
        
        probabilities = {
            self.id2label[i]: float(probs[0][i])
            for i in range(len(self.id2label))
        }
        
        return {
            'sentiment': predicted_label,
            'confidence': confidence_score,
            'probabilities': probabilities
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment for multiple texts."""
        results = []
        
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Test inference
    model_path = "models/best_distilbert_model.pth"
    
    if Path(model_path).exists():
        predictor = SentimentPredictor(model_path, model_type="distilbert")
        
        test_texts = [
            "This movie is amazing! I loved every minute of it.",
            "Terrible product. Complete waste of money.",
            "It was okay, nothing special."
        ]
        
        print("Testing sentiment predictions:\n")
        for text in test_texts:
            result = predictor.predict(text)
            print(f"Text: {text}")
            print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.4f})")
            print(f"Probabilities: {result['probabilities']}\n")
    else:
        print(f"Model not found at {model_path}")
        print("Train a model first using: python -m src.models.train")
