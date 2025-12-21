"""
Training script with MLflow experiment tracking.
Integrates with DagsHub for FREE MLOps tracking.

Setup DagsHub (FREE):
1. Create account at https://dagshub.com
2. Create new repository
3. Get credentials from Settings -> Integrations -> MLflow
4. Set environment variables or update this file
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from loguru import logger
import yaml
from datetime import datetime

from src.data.data_loader import load_data
from src.models.model import create_model
from src.models.evaluate import evaluate_model, compute_metrics


class Trainer:
    """Training pipeline with MLflow tracking."""
    
    def __init__(
        self,
        model_type: str = "distilbert",
        config_path: str = None,
        experiment_name: str = "sentiment-analysis"
    ):
        self.model_type = model_type
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup MLflow
        self.setup_mlflow(experiment_name)
        
        logger.info(f"Using device: {self.device}")
    
    def load_config(self, config_path: str = None) -> dict:
        """Load training configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'model': {
                'type': self.model_type,
                'num_classes': 2,
                'dropout': 0.3,
                'freeze_bert': False,  # For DistilBERT
                'vocab_size': 10000,    # For LSTM/BoW
                'embedding_dim': 128,   # For LSTM/BoW
                'hidden_dim': 256,      # For LSTM/BoW
            },
            'training': {
                'batch_size': 32,
                'epochs': 5,
                'learning_rate': 2e-5,
                'weight_decay': 0.01,
                'max_length': 128,
                'early_stopping_patience': 3,
            },
            'data': {
                'data_dir': 'data/processed',
            }
        }
    
    def setup_mlflow(self, experiment_name: str):
        """
        Setup MLflow tracking with DagsHub.
        
        DagsHub provides FREE:
        - Unlimited experiments
        - 10GB storage
        - Git + DVC + MLflow integration
        """
        # Option 1: Use DagsHub (recommended for students)
        dagshub_user = os.getenv("DAGSHUB_USER", "your-username")
        dagshub_repo = os.getenv("DAGSHUB_REPO", "mlops-sentiment-analysis")
        
        if dagshub_user != "your-username":
            # Connect to DagsHub (uses cached OAuth token)
            try:
                import dagshub
                dagshub.init(dagshub_user, dagshub_repo, mlflow=True)
                logger.info(f"✓ Using DagsHub MLflow tracking: {dagshub_user}/{dagshub_repo}")
            except Exception as e:
                logger.warning(f"Failed to connect to DagsHub: {e}")
                logger.info("Falling back to local MLflow tracking")
                mlflow.set_tracking_uri("file:./mlruns")
        else:
            # Option 2: Use local MLflow
            mlflow.set_tracking_uri("file:./mlruns")
            logger.info("Using local MLflow tracking")
            logger.info("To use DagsHub: Set DAGSHUB_USER and DAGSHUB_REPO env vars")
        
        mlflow.set_experiment(experiment_name)
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader,
        optimizer,
        criterion,
        epoch: int
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            if self.model_type == "distilbert":
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = model(input_ids, attention_mask)
            else:
                outputs = model(input_ids)
            
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(
        self,
        model: nn.Module,
        val_loader,
        criterion
    ) -> tuple:
        """Validate model."""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if self.model_type == "distilbert":
                    attention_mask = batch['attention_mask'].to(self.device)
                    outputs = model(input_ids, attention_mask)
                else:
                    outputs = model(input_ids)
                
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metrics = compute_metrics(all_labels, all_preds)
        
        return avg_loss, metrics
    
    def train(self):
        """Main training loop with MLflow tracking."""
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            mlflow.log_params(self.config['training'])
            mlflow.log_params(self.config['model'])
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("device", str(self.device))
            
            # Load data
            logger.info("Loading data...")
            data_dir = Path(self.config['data']['data_dir'])
            train_loader, val_loader, test_loader, vocab = load_data(
                data_dir,
                batch_size=self.config['training']['batch_size'],
                model_type=self.model_type,
                max_length=self.config['training']['max_length']
            )
            
            # Create model
            logger.info(f"Creating {self.model_type} model...")
            model_config = self.config['model'].copy()
            
            # Remove 'type' key as it's not a model parameter
            model_config.pop('type', None)
            
            # Remove DistilBERT-specific params for other models
            if self.model_type != "distilbert":
                model_config.pop('freeze_bert', None)
            
            # Remove LSTM-specific params for BoW
            if self.model_type == "bow":
                model_config.pop('num_layers', None)
                model_config.pop('bidirectional', None)
            
            if vocab is not None and self.model_type != "distilbert":
                model_config['vocab_size'] = len(vocab)
            
            model = create_model(self.model_type, **model_config)
            model = model.to(self.device)
            
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model parameters: {num_params:,}")
            mlflow.log_param("num_parameters", num_params)
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = AdamW(
                model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(1, self.config['training']['epochs'] + 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Epoch {epoch}/{self.config['training']['epochs']}")
                logger.info(f"{'='*60}")
                
                # Train
                train_loss, train_acc = self.train_epoch(
                    model, train_loader, optimizer, criterion, epoch
                )
                
                # Validate
                val_loss, val_metrics = self.validate(model, val_loader, criterion)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                
                # Log metrics to MLflow
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_metrics['accuracy'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_f1': val_metrics['f1'],
                    'learning_rate': current_lr
                }, step=epoch)
                
                # Print results
                logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
                logger.info(f"Val F1: {val_metrics['f1']:.4f} | LR: {current_lr:.2e}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save model
                    model_dir = Path("models")
                    model_dir.mkdir(exist_ok=True)
                    model_path = model_dir / f"best_{self.model_type}_model.pth"
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_metrics': val_metrics,
                        'config': self.config,
                        'vocab': vocab
                    }, model_path)
                    
                    logger.info(f"✓ Saved best model to {model_path}")
                    
                    # Log model to MLflow
                    mlflow.pytorch.log_model(model, "model")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Patience: {patience_counter}/{self.config['training']['early_stopping_patience']}")
                
                # Early stopping
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    logger.info("Early stopping triggered!")
                    break
            
            # Final evaluation on test set
            logger.info("\n" + "="*60)
            logger.info("Final Evaluation on Test Set")
            logger.info("="*60)
            
            # Load best model
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            test_loss, test_metrics = self.validate(model, test_loader, criterion)
            
            # Log final test metrics
            mlflow.log_metrics({
                'test_loss': test_loss,
                'test_accuracy': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1': test_metrics['f1']
            })
            
            logger.info(f"Test Loss: {test_loss:.4f}")
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
            logger.info(f"Test F1 Score: {test_metrics['f1']:.4f}")
            
            # Log artifacts
            mlflow.log_artifact(model_path)
            
            logger.info("\n✓ Training complete!")
            logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="distilbert", 
                       choices=["distilbert", "lstm", "bow"],
                       help="Model type to train")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    parser.add_argument("--experiment", type=str, default="sentiment-analysis",
                       help="MLflow experiment name")
    
    args = parser.parse_args()
    
    trainer = Trainer(
        model_type=args.model,
        config_path=args.config,
        experiment_name=args.experiment
    )
    
    trainer.train()
