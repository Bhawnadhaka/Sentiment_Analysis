"""
Model performance tracking and monitoring.
Tracks metrics over time and generates alerts.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import json
from collections import defaultdict


class PerformanceTracker:
    """Track model performance metrics over time."""
    
    def __init__(
        self,
        metrics_file: str = "monitoring/metrics_history.json",
        alert_thresholds: Dict[str, float] = None
    ):
        """
        Initialize performance tracker.
        
        Args:
            metrics_file: File to store metrics history
            alert_thresholds: Thresholds for alerting
        """
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'accuracy': 0.75,      # Alert if accuracy drops below 75%
            'f1': 0.70,            # Alert if F1 drops below 70%
            'confidence': 0.60,    # Alert if avg confidence drops below 60%
            'drift_score': 0.3     # Alert if drift score exceeds 0.3
        }
        
        # Load existing metrics
        self.metrics_history = self.load_metrics()
    
    def load_metrics(self) -> List[Dict]:
        """Load metrics history from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics: {e}")
                return []
        return []
    
    def save_metrics(self):
        """Save metrics history to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.debug(f"Metrics saved to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None
    ):
        """
        Log new metrics.
        
        Args:
            metrics: Dictionary of metric values
            metadata: Additional metadata (model version, dataset, etc.)
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        self.metrics_history.append(entry)
        self.save_metrics()
        
        logger.info(f"Logged metrics: {metrics}")
        
        # Check for alerts
        alerts = self.check_alerts(metrics)
        if alerts:
            logger.warning(f"⚠️ ALERTS: {alerts}")
    
    def check_alerts(self, metrics: Dict[str, float]) -> List[str]:
        """
        Check if any metrics exceed alert thresholds.
        
        Returns:
            List of alert messages
        """
        alerts = []
        
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                # Different comparison based on metric type
                if metric_name == 'drift_score':
                    # Alert if drift score is too high
                    if value > threshold:
                        alerts.append(
                            f"{metric_name} ({value:.4f}) exceeds threshold ({threshold})"
                        )
                else:
                    # Alert if metric is too low
                    if value < threshold:
                        alerts.append(
                            f"{metric_name} ({value:.4f}) below threshold ({threshold})"
                        )
        
        return alerts
    
    def get_recent_metrics(
        self,
        hours: int = 24,
        metric_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get recent metrics as DataFrame.
        
        Args:
            hours: Number of hours to look back
            metric_names: Specific metrics to retrieve
            
        Returns:
            DataFrame with metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_data = []
        for entry in self.metrics_history:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            if timestamp >= cutoff_time:
                row = {'timestamp': timestamp}
                
                # Add metrics
                for name, value in entry['metrics'].items():
                    if metric_names is None or name in metric_names:
                        row[name] = value
                
                # Add metadata
                for name, value in entry.get('metadata', {}).items():
                    row[f'meta_{name}'] = value
                
                recent_data.append(row)
        
        if recent_data:
            return pd.DataFrame(recent_data)
        else:
            return pd.DataFrame()
    
    def get_summary(self, hours: int = 24) -> Dict:
        """
        Get summary statistics for recent period.
        
        Returns:
            Dictionary with summary statistics
        """
        df = self.get_recent_metrics(hours=hours)
        
        if df.empty:
            return {'message': 'No data available'}
        
        summary = {
            'period_hours': hours,
            'num_records': len(df),
            'start_time': df['timestamp'].min().isoformat(),
            'end_time': df['timestamp'].max().isoformat(),
            'metrics': {}
        }
        
        # Calculate statistics for each metric
        metric_columns = [col for col in df.columns 
                         if col not in ['timestamp'] and not col.startswith('meta_')]
        
        for col in metric_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                summary['metrics'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'current': float(df[col].iloc[-1]) if len(df) > 0 else None
                }
        
        return summary
    
    def detect_performance_degradation(
        self,
        metric_name: str = 'accuracy',
        window_hours: int = 24,
        baseline_hours: int = 168  # 1 week
    ) -> Dict:
        """
        Detect performance degradation by comparing recent vs baseline.
        
        Args:
            metric_name: Metric to analyze
            window_hours: Recent window to compare
            baseline_hours: Baseline period for comparison
            
        Returns:
            Dictionary with degradation analysis
        """
        recent_df = self.get_recent_metrics(hours=window_hours, metric_names=[metric_name])
        baseline_df = self.get_recent_metrics(hours=baseline_hours, metric_names=[metric_name])
        
        if recent_df.empty or baseline_df.empty or metric_name not in recent_df.columns:
            return {'degradation_detected': False, 'message': 'Insufficient data'}
        
        recent_mean = recent_df[metric_name].mean()
        baseline_mean = baseline_df[metric_name].mean()
        
        # Calculate percentage change
        pct_change = ((recent_mean - baseline_mean) / baseline_mean) * 100
        
        # Degradation if recent performance is significantly worse
        threshold = -5  # 5% decrease
        degradation_detected = pct_change < threshold
        
        return {
            'degradation_detected': degradation_detected,
            'metric': metric_name,
            'recent_mean': recent_mean,
            'baseline_mean': baseline_mean,
            'percent_change': pct_change,
            'window_hours': window_hours,
            'baseline_hours': baseline_hours
        }


if __name__ == "__main__":
    # Example usage
    tracker = PerformanceTracker()
    
    # Log some sample metrics
    print("Logging sample metrics...\n")
    
    for i in range(5):
        metrics = {
            'accuracy': 0.85 + np.random.normal(0, 0.02),
            'f1': 0.83 + np.random.normal(0, 0.03),
            'precision': 0.84 + np.random.normal(0, 0.02),
            'recall': 0.82 + np.random.normal(0, 0.03),
            'confidence': 0.75 + np.random.normal(0, 0.05)
        }
        
        tracker.log_metrics(
            metrics,
            metadata={'model_version': 'v1.0', 'dataset': 'test'}
        )
    
    # Get summary
    print("\nMetrics Summary:")
    print(json.dumps(tracker.get_summary(hours=24), indent=2))
    
    # Check for degradation
    print("\nPerformance Degradation Check:")
    degradation = tracker.detect_performance_degradation('accuracy')
    print(json.dumps(degradation, indent=2))
