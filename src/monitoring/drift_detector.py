"""
Evidently AI integration for model and data drift detection.

Evidently is FREE and open-source!
- No cloud service required
- Generates interactive HTML reports
- Detects data drift, model drift, and performance degradation
"""

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    TargetDriftPreset,
    DataQualityPreset,
    ClassificationPreset
)
from evidently.metrics import *
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
from typing import Optional
import json


class DriftDetector:
    """Detect data and model drift using Evidently AI."""
    
    def __init__(
        self,
        reference_data: pd.DataFrame = None,
        output_dir: str = "monitoring/evidently_reports"
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference dataset (training data)
            output_dir: Directory to save reports
        """
        self.reference_data = reference_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Column mapping for sentiment analysis
        self.column_mapping = ColumnMapping(
            target='label',
            prediction='prediction',
            numerical_features=[],
            categorical_features=[]
        )
    
    def load_reference_data(self, file_path: str, sample_size: int = 1000):
        """Load reference data from file."""
        logger.info(f"Loading reference data from {file_path}...")
        
        df = pd.read_csv(file_path)
        
        # Sample if dataset is large
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        # Use cleaned text if available
        text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
        
        self.reference_data = df[[text_col, 'label']].copy()
        self.reference_data.columns = ['text', 'label']
        
        logger.info(f"Loaded {len(self.reference_data)} reference samples")
        
        return self.reference_data
    
    def create_data_drift_report(
        self,
        current_data: pd.DataFrame,
        report_name: str = None
    ) -> str:
        """
        Create data drift report.
        
        Args:
            current_data: Current/production data
            report_name: Custom report name
            
        Returns:
            Path to generated report
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call load_reference_data() first.")
        
        logger.info("Generating data drift report...")
        
        # Create report
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        
        # Run report
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Save report
        if report_name is None:
            report_name = f"data_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        report_path = self.output_dir / f"{report_name}.html"
        report.save_html(str(report_path))
        
        logger.info(f"Report saved to: {report_path}")
        
        # Also save JSON for programmatic access
        json_path = self.output_dir / f"{report_name}.json"
        report.save_json(str(json_path))
        
        return str(report_path)
    
    def create_model_performance_report(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame = None,
        report_name: str = None
    ) -> str:
        """
        Create model performance report.
        
        Args:
            current_data: Current data with predictions
            reference_data: Reference data with predictions (optional)
            report_name: Custom report name
            
        Returns:
            Path to generated report
        """
        logger.info("Generating model performance report...")
        
        if reference_data is None:
            reference_data = self.reference_data
        
        # Create report
        report = Report(metrics=[
            ClassificationPreset(),
            TargetDriftPreset(),
        ])
        
        # Run report
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Save report
        if report_name is None:
            report_name = f"model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        report_path = self.output_dir / f"{report_name}.html"
        report.save_html(str(report_path))
        
        logger.info(f"Report saved to: {report_path}")
        
        # Save JSON
        json_path = self.output_dir / f"{report_name}.json"
        report.save_json(str(json_path))
        
        return str(report_path)
    
    def check_drift(self, current_data: pd.DataFrame) -> dict:
        """
        Quick drift check without full report.
        
        Returns:
            Dictionary with drift metrics
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        # Create lightweight report
        report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Extract drift info from JSON
        report_dict = report.as_dict()
        
        # Parse drift results
        drift_detected = False
        drifted_features = []
        
        # This is a simplified extraction - adjust based on Evidently's JSON structure
        try:
            metrics = report_dict.get('metrics', [])
            for metric in metrics:
                if metric.get('metric') == 'DatasetDriftMetric':
                    drift_detected = metric.get('result', {}).get('dataset_drift', False)
                    drifted_features = metric.get('result', {}).get('drift_by_columns', {})
                    break
        except Exception as e:
            logger.warning(f"Could not extract drift info: {e}")
        
        return {
            'drift_detected': drift_detected,
            'drifted_features': drifted_features,
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # Paths
    reference_file = Path("data/processed/train.csv")
    current_file = Path("data/processed/test.csv")
    
    if reference_file.exists() and current_file.exists():
        # Initialize detector
        detector = DriftDetector()
        
        # Load reference data (training data)
        detector.load_reference_data(str(reference_file), sample_size=500)
        
        # Load current data (test data as example)
        current_data = pd.read_csv(current_file).head(500)
        text_col = 'cleaned_text' if 'cleaned_text' in current_data.columns else 'text'
        current_data = current_data[[text_col, 'label']].copy()
        current_data.columns = ['text', 'label']
        
        # Create drift report
        report_path = detector.create_data_drift_report(current_data)
        
        print(f"\n✓ Drift report generated: {report_path}")
        print(f"Open it in your browser to view the analysis")
        
        # Quick drift check
        drift_info = detector.check_drift(current_data)
        print(f"\nDrift detected: {drift_info['drift_detected']}")
        
    else:
        print("Data files not found. Please run preprocessing first:")
        print("  python -m data.scripts.preprocess")
