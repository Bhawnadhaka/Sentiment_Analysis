"""
Evidently AI Dashboard for Sentiment Analysis Monitoring
Generates interactive HTML reports for model performance and data drift
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    ClassificationPreset
)
from evidently.metrics import *


class SentimentMonitor:
    """Generate monitoring dashboards for sentiment predictions"""
    
    def __init__(
        self,
        db_path: str = "sentiment_predictions.db",
        output_dir: str = "monitoring/evidently_reports"
    ):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Monitor initialized. Reports → {self.output_dir}")
    
    def load_predictions(self) -> pd.DataFrame:
        """Load all predictions from database"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT 
                title,
                text,
                sentiment,
                confidence,
                source,
                topic,
                processed_at as timestamp
            FROM predictions
            ORDER BY processed_at DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            logger.warning("No predictions found in database")
            return df
        
        # Convert sentiment to binary (0=negative, 1=positive)
        df['prediction'] = df['sentiment'].map({'negative': 0, 'positive': 1})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded {len(df)} predictions from database")
        return df
    
    def generate_classification_report(self, df: pd.DataFrame) -> str:
        """Generate classification performance report"""
        if df.empty:
            logger.warning("No data available for classification report")
            return None
        
        # Split into reference (first half) and current (second half) for comparison
        mid_point = len(df) // 2
        reference_data = df.iloc[:mid_point].copy()
        current_data = df.iloc[mid_point:].copy()
        
        # Define column mapping
        column_mapping = ColumnMapping(
            target='prediction',
            prediction='prediction',
            numerical_features=['confidence']
        )
        
        # Create report
        report = Report(metrics=[
            ClassificationPreset(),
        ])
        
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"classification_report_{timestamp}.html"
        report.save_html(str(report_path))
        
        logger.info(f"✓ Classification report saved: {report_path}")
        return str(report_path)
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> str:
        """Generate data quality report"""
        if df.empty:
            logger.warning("No data available for data quality report")
            return None
        
        report = Report(metrics=[
            DataQualityPreset(),
        ])
        
        report.run(
            reference_data=None,
            current_data=df
        )
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"data_quality_{timestamp}.html"
        report.save_html(str(report_path))
        
        logger.info(f"✓ Data quality report saved: {report_path}")
        return str(report_path)
    
    def generate_drift_report(self, df: pd.DataFrame) -> str:
        """Generate data drift detection report"""
        if df.empty or len(df) < 10:
            logger.warning("Insufficient data for drift detection (need at least 10 samples)")
            return None
        
        # Split into reference (older) and current (newer) data
        mid_point = len(df) // 2
        reference_data = df.iloc[:mid_point].copy()
        current_data = df.iloc[mid_point:].copy()
        
        # Define column mapping
        column_mapping = ColumnMapping(
            target='prediction',
            numerical_features=['confidence']
        )
        
        report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"drift_report_{timestamp}.html"
        report.save_html(str(report_path))
        
        logger.info(f"✓ Drift report saved: {report_path}")
        return str(report_path)
    
    def generate_custom_metrics_report(self, df: pd.DataFrame) -> str:
        """Generate custom metrics report with detailed analysis"""
        if df.empty:
            logger.warning("No data available for custom metrics report")
            return None
        
        # Split data for comparison
        mid_point = len(df) // 2
        reference_data = df.iloc[:mid_point].copy()
        current_data = df.iloc[mid_point:].copy()
        
        column_mapping = ColumnMapping(
            target='prediction',
            prediction='prediction',
            numerical_features=['confidence']
        )
        
        report = Report(metrics=[
            # Classification metrics
            ClassificationQualityMetric(),
            ClassificationConfusionMatrix(),
            ClassificationProbDistribution(),
            
            # Confidence analysis
            ColumnDriftMetric(column_name='confidence'),
            ColumnDistributionMetric(column_name='confidence'),
            ColumnQuantileMetric(column_name='confidence', quantile=0.5),
            
            # Dataset summary
            DatasetSummaryMetric(),
            DatasetMissingValuesMetric(),
        ])
        
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"detailed_metrics_{timestamp}.html"
        report.save_html(str(report_path))
        
        logger.info(f"✓ Detailed metrics report saved: {report_path}")
        return str(report_path)
    
    def generate_all_reports(self):
        """Generate all monitoring reports"""
        logger.info("="*60)
        logger.info("Generating Evidently AI Monitoring Reports")
        logger.info("="*60)
        
        # Load data
        df = self.load_predictions()
        
        if df.empty:
            logger.error("No predictions found. Run the streaming pipeline first.")
            return
        
        # Generate reports
        reports = []
        
        logger.info("\n📊 Generating reports...")
        
        # 1. Data Quality
        quality_report = self.generate_data_quality_report(df)
        if quality_report:
            reports.append(("Data Quality", quality_report))
        
        # 2. Classification Performance
        class_report = self.generate_classification_report(df)
        if class_report:
            reports.append(("Classification Performance", class_report))
        
        # 3. Data Drift
        drift_report = self.generate_drift_report(df)
        if drift_report:
            reports.append(("Data Drift Detection", drift_report))
        
        # 4. Detailed Metrics
        metrics_report = self.generate_custom_metrics_report(df)
        if metrics_report:
            reports.append(("Detailed Metrics", metrics_report))
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("✓ Report Generation Complete!")
        logger.info("="*60)
        logger.info(f"\nGenerated {len(reports)} reports:\n")
        
        for name, path in reports:
            logger.info(f"  📈 {name}")
            logger.info(f"     → {path}\n")
        
        return reports


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Evidently AI monitoring dashboards")
    parser.add_argument(
        "--db",
        default="sentiment_predictions.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--output",
        default="monitoring/evidently_reports",
        help="Output directory for reports"
    )
    
    args = parser.parse_args()
    
    # Generate reports
    monitor = SentimentMonitor(db_path=args.db, output_dir=args.output)
    monitor.generate_all_reports()


if __name__ == "__main__":
    main()
