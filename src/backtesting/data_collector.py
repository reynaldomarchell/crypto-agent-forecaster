"""
Data collection and organization for thesis research.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataCollector:
    """Handles data collection, storage, and organization for thesis research."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.raw_data_dir = self.data_dir / "raw_data"
        self.processed_data_dir = self.data_dir / "processed_data"
        self.analysis_dir = self.data_dir / "analysis"
        self.charts_dir = self.data_dir / "charts"
        
        for dir_path in [self.raw_data_dir, self.processed_data_dir, self.analysis_dir, self.charts_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_daily_data(self, date: str, crypto_symbol: str, data: Dict[str, Any]):
        """Save daily data for a specific date and cryptocurrency."""
        try:
            file_path = self.raw_data_dir / f"{crypto_symbol}_{date}_data.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved daily data for {crypto_symbol} on {date}")
        except Exception as e:
            logger.error(f"Failed to save daily data for {date}: {e}")
    
    def load_daily_data(self, date: str, crypto_symbol: str) -> Optional[Dict[str, Any]]:
        """Load daily data for a specific date and cryptocurrency."""
        try:
            file_path = self.raw_data_dir / f"{crypto_symbol}_{date}_data.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load daily data for {date}: {e}")
        return None
    
    def save_prediction_results(self, results: Dict[str, Any], crypto_symbol: str):
        """Save prediction results to processed data."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.processed_data_dir / f"{crypto_symbol}_predictions_{timestamp}.json"
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved prediction results to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save prediction results: {e}")
    
    def export_to_csv(self, results: Dict[str, Any], crypto_symbol: str) -> Path:
        """Export results to CSV format for thesis analysis."""
        try:
            records = []
            
            for date, predictions in results.get("daily_predictions", {}).items():
                base_record = {
                    "date": date,
                    "crypto_symbol": crypto_symbol
                }
                
                for method, prediction in predictions.items():
                    if prediction.get("success"):
                        record = base_record.copy()
                        record.update({
                            "method": method,
                            "predicted_direction": prediction.get("predicted_direction", "DOWN"),  # Default to DOWN instead of NEUTRAL
                            "confidence": prediction.get("confidence", "LOW"),
                            "reasoning": prediction.get("reasoning", ""),
                            "method": prediction.get("method", "unknown"),
                            "success": prediction.get("success", False),
                            "actual_movement_24h": prediction.get("actual_movement_24h"),
                            "execution_time": prediction.get("execution_time"),
                            "timestamp": datetime.now().isoformat()
                        })
                        records.append(record)
            
            if records:
                df = pd.DataFrame(records)
                csv_path = self.processed_data_dir / f"{crypto_symbol}_thesis_data.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Exported {len(records)} records to {csv_path}")
                return csv_path
            else:
                logger.warning("No valid records to export")
                return None
                
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return None
    
    def create_comparison_dataset(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create a comparison dataset for thesis analysis."""
        try:
            comparison_data = []
            
            for date, predictions in results.get("daily_predictions", {}).items():
                # Get the actual outcome for this date
                actual_movement = None
                actual_direction = None
                
                # Find any successful prediction to get actual data
                for method, prediction in predictions.items():
                    if prediction.get("success") and prediction.get("actual_movement_24h") is not None:
                        actual_movement = prediction["actual_movement_24h"]
                        if actual_movement > 1:
                            actual_direction = "UP"
                        elif actual_movement < -1:
                            actual_direction = "DOWN"
                        else:
                            # For small movements, assign to DOWN (conservative approach)
                            actual_direction = "DOWN"
                        break
                
                if actual_direction is None:
                    continue  # Skip dates without actual outcomes
                
                # Create comparison record
                record = {
                    "date": date,
                    "actual_direction": actual_direction,
                    "actual_movement": actual_movement
                }
                
                # Add predictions from each method
                for method in ["full_agentic", "image_only", "sentiment_only"]:
                    if method in predictions and predictions[method].get("success"):
                        pred = predictions[method]
                        record[f"{method}_prediction"] = pred.get("predicted_direction", "DOWN")
                        record[f"{method}_confidence"] = pred.get("confidence", "MEDIUM")
                        record[f"{method}_correct"] = (pred.get("predicted_direction") == actual_direction)
                    else:
                        record[f"{method}_prediction"] = "FAILED"
                        record[f"{method}_confidence"] = "NONE"
                        record[f"{method}_correct"] = False
                
                comparison_data.append(record)
            
            return pd.DataFrame(comparison_data)
            
        except Exception as e:
            logger.error(f"Failed to create comparison dataset: {e}")
            return pd.DataFrame()
    
    def save_analysis_metrics(self, metrics: Dict[str, Any], crypto_symbol: str):
        """Save analysis metrics for thesis."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.analysis_dir / f"{crypto_symbol}_metrics_{timestamp}.json"
            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved analysis metrics to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save analysis metrics: {e}")
    
    def get_data_summary(self, crypto_symbol: str) -> Dict[str, Any]:
        """Get summary of collected data."""
        try:
            raw_files = list(self.raw_data_dir.glob(f"{crypto_symbol}_*_data.json"))
            processed_files = list(self.processed_data_dir.glob(f"{crypto_symbol}_*.json"))
            csv_files = list(self.processed_data_dir.glob(f"{crypto_symbol}_*.csv"))
            
            return {
                "raw_data_files": len(raw_files),
                "processed_files": len(processed_files),
                "csv_files": len(csv_files),
                "data_directory": str(self.data_dir),
                "latest_raw_file": str(raw_files[-1]) if raw_files else None,
                "latest_processed_file": str(processed_files[-1]) if processed_files else None,
                "storage_size_mb": self._get_directory_size() / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Failed to get data summary: {e}")
            return {"error": str(e)}
    
    def _get_directory_size(self) -> int:
        """Get total size of data directory in bytes."""
        try:
            return sum(f.stat().st_size for f in self.data_dir.rglob('*') if f.is_file())
        except Exception:
            return 0
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data files beyond specified days."""
        try:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            deleted_files = 0
            
            for file_path in self.data_dir.rglob('*'):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_date:
                    try:
                        file_path.unlink()
                        deleted_files += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
            
            logger.info(f"Cleaned up {deleted_files} old files")
            return deleted_files
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0 