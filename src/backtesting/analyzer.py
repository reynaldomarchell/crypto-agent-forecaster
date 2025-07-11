"""
Thesis analysis module for statistical comparison of prediction methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from scipy import stats
import json

logger = logging.getLogger(__name__)


class ThesisAnalyzer:
    """Analyzes and compares prediction methods for thesis research."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.analysis_dir = self.data_dir / "analysis"
        self.charts_dir = self.data_dir / "charts"
        
        # Ensure directories exist
        self.analysis_dir.mkdir(exist_ok=True)
        self.charts_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def analyze_prediction_accuracy(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze prediction accuracy across all methods."""
        try:
            analysis = {
                "overall_stats": {},
                "method_comparison": {},
                "confidence_analysis": {},
                "temporal_analysis": {},
                "detailed_metrics": {}
            }
            
            # Create comparison dataset
            df = self._create_analysis_dataframe(results)
            
            if df.empty:
                return {"error": "No valid data for analysis"}
            
            # Overall statistics
            analysis["overall_stats"] = self._calculate_overall_stats(df)
            
            # Method comparison
            analysis["method_comparison"] = self._compare_methods(df)
            
            # Confidence analysis
            analysis["confidence_analysis"] = self._analyze_confidence_correlation(df)
            
            # Temporal analysis
            analysis["temporal_analysis"] = self._analyze_temporal_patterns(df)
            
            # Detailed metrics
            analysis["detailed_metrics"] = self._calculate_detailed_metrics(df)
            
            # Statistical significance tests
            analysis["statistical_tests"] = self._perform_statistical_tests(df)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze prediction accuracy: {e}")
            return {"error": str(e)}
    
    def _create_analysis_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create a comprehensive dataframe for analysis."""
        records = []
        
        for date, predictions in results.get("daily_predictions", {}).items():
            # Get actual outcome
            actual_movement = None
            actual_direction = None
            actual_price = None
            
            # Find any successful prediction to get actual data
            for method, prediction in predictions.items():
                if prediction.get("success") and prediction.get("actual_movement_24h") is not None:
                    actual_movement = prediction["actual_movement_24h"]
                    actual_price = prediction.get("actual_price", 0)
                    
                    # Determine actual direction based on price movement
                    if actual_movement > 1.0:  # > 1% movement threshold
                        actual_direction = "UP"
                        actual_numerical = 1
                    elif actual_movement < -1.0:  # < -1% movement threshold
                        actual_direction = "DOWN"
                        actual_numerical = -1
                    else:
                        # For small movements, assign to DOWN (conservative approach)
                        actual_direction = "DOWN"
                        actual_numerical = -1
                    break
            
            if actual_direction is None:
                continue
            
            # Create records for each method
            for method in ["full_agentic", "image_only", "sentiment_only"]:
                if method in predictions:
                    prediction = predictions[method]
                    
                    record = {
                        "date": pd.to_datetime(date),
                        "method": method,
                        "actual_direction": actual_direction,
                        "actual_movement": actual_movement,
                        "actual_price": actual_price,
                        "predicted_direction": prediction.get("predicted_direction", "FAILED"),
                        "confidence": prediction.get("confidence", "NONE"),
                        "success": prediction.get("success", False),
                        "correct_prediction": (
                            prediction.get("predicted_direction") == actual_direction 
                            if prediction.get("success") else False
                        ),
                        "execution_time": prediction.get("execution_time"),
                        "error": prediction.get("error", "")
                    }
                    
                    records.append(record)
        
        return pd.DataFrame(records)
    
    def _calculate_overall_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall statistics across all methods."""
        return {
            "total_predictions": len(df),
            "successful_predictions": df["success"].sum(),
            "overall_accuracy": df["correct_prediction"].mean() if len(df) > 0 else 0,
            "success_rate": df["success"].mean() if len(df) > 0 else 0,
            "date_range": {
                "start": df["date"].min().strftime("%Y-%m-%d") if len(df) > 0 else None,
                "end": df["date"].max().strftime("%Y-%m-%d") if len(df) > 0 else None,
                "total_days": df["date"].nunique()
            },
            "actual_direction_distribution": df["actual_direction"].value_counts().to_dict(),
            "average_actual_movement": df["actual_movement"].mean()
        }
    
    def _compare_methods(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare accuracy across different methods."""
        method_stats = {}
        
        for method in ["full_agentic", "image_only", "sentiment_only"]:
            method_df = df[df["method"] == method]
            
            if len(method_df) > 0:
                successful_df = method_df[method_df["success"] == True]
                
                method_stats[method] = {
                    "total_attempts": len(method_df),
                    "successful_predictions": len(successful_df),
                    "success_rate": len(successful_df) / len(method_df),
                    "accuracy": successful_df["correct_prediction"].mean() if len(successful_df) > 0 else 0,
                    "correct_predictions": successful_df["correct_prediction"].sum(),
                    "direction_accuracy": {
                        direction: successful_df[successful_df["actual_direction"] == direction]["correct_prediction"].mean()
                        for direction in ["UP", "DOWN"]
                        if len(successful_df[successful_df["actual_direction"] == direction]) > 0
                    },
                    "confidence_distribution": successful_df["confidence"].value_counts().to_dict(),
                    "avg_execution_time": method_df["execution_time"].mean() if method_df["execution_time"].notna().any() else None
                }
        
        # Calculate relative performance
        accuracies = {method: stats["accuracy"] for method, stats in method_stats.items()}
        best_method = max(accuracies.keys(), key=lambda k: accuracies[k]) if accuracies else None
        
        method_stats["comparison_summary"] = {
            "best_method": best_method,
            "accuracy_ranking": sorted(accuracies.items(), key=lambda x: x[1], reverse=True),
            "accuracy_differences": {
                f"{m1}_vs_{m2}": accuracies[m1] - accuracies[m2]
                for m1 in accuracies for m2 in accuracies if m1 != m2
            }
        }
        
        return method_stats
    
    def _analyze_confidence_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation between confidence levels and accuracy."""
        confidence_analysis = {}
        
        for method in ["full_agentic", "image_only", "sentiment_only"]:
            method_df = df[(df["method"] == method) & (df["success"] == True)]
            
            if len(method_df) > 0:
                confidence_stats = {}
                
                for confidence in ["HIGH", "MEDIUM", "LOW"]:
                    conf_df = method_df[method_df["confidence"] == confidence]
                    if len(conf_df) > 0:
                        confidence_stats[confidence] = {
                            "count": len(conf_df),
                            "accuracy": conf_df["correct_prediction"].mean(),
                            "correct_predictions": conf_df["correct_prediction"].sum()
                        }
                
                confidence_analysis[method] = confidence_stats
        
        return confidence_analysis
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in predictions."""
        temporal_analysis = {}
        
        if len(df) > 0:
            df["month"] = df["date"].dt.month
            df["day_of_week"] = df["date"].dt.dayofweek
            df["week_of_year"] = df["date"].dt.isocalendar().week
            
            # Monthly analysis
            monthly_accuracy = df.groupby(["method", "month"])["correct_prediction"].mean().unstack(level=0)
            temporal_analysis["monthly_accuracy"] = monthly_accuracy.to_dict() if not monthly_accuracy.empty else {}
            
            # Day of week analysis
            dow_accuracy = df.groupby(["method", "day_of_week"])["correct_prediction"].mean().unstack(level=0)
            temporal_analysis["day_of_week_accuracy"] = dow_accuracy.to_dict() if not dow_accuracy.empty else {}
            
            # Trend analysis over time
            df_sorted = df.sort_values("date")
            rolling_accuracy = df_sorted.groupby("method")["correct_prediction"].rolling(window=30, min_periods=10).mean()
            temporal_analysis["accuracy_trend"] = rolling_accuracy.to_dict() if not rolling_accuracy.empty else {}
        
        return temporal_analysis
    
    def _calculate_detailed_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed performance metrics."""
        detailed_metrics = {}
        
        for method in ["full_agentic", "image_only", "sentiment_only"]:
            method_df = df[(df["method"] == method) & (df["success"] == True)]
            
            if len(method_df) > 0:
                # Confusion matrix components
                tp = len(method_df[(method_df["predicted_direction"] == "UP") & (method_df["actual_direction"] == "UP")])
                tn = len(method_df[(method_df["predicted_direction"] == "DOWN") & (method_df["actual_direction"] == "DOWN")])
                fp = len(method_df[(method_df["predicted_direction"] == "UP") & (method_df["actual_direction"] != "UP")])
                fn = len(method_df[(method_df["predicted_direction"] == "DOWN") & (method_df["actual_direction"] != "DOWN")])
                
                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                detailed_metrics[method] = {
                    "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
                    "balanced_accuracy": (recall + (tn / (tn + fp) if (tn + fp) > 0 else 0)) / 2
                }
        
        return detailed_metrics
    
    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        statistical_tests = {}
        
        try:
            methods = ["full_agentic", "image_only", "sentiment_only"]
            
            # Chi-square test for independence
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    method1_df = df[(df["method"] == method1) & (df["success"] == True)]
                    method2_df = df[(df["method"] == method2) & (df["success"] == True)]
                    
                    if len(method1_df) > 10 and len(method2_df) > 10:
                        # Create contingency table
                        contingency = pd.crosstab(
                            pd.concat([method1_df["correct_prediction"], method2_df["correct_prediction"]]),
                            pd.concat([pd.Series([method1]*len(method1_df)), pd.Series([method2]*len(method2_df))])
                        )
                        
                        if contingency.shape == (2, 2):
                            chi2, p_value = stats.chi2_contingency(contingency)[:2]
                            
                            statistical_tests[f"{method1}_vs_{method2}"] = {
                                "test": "chi_square",
                                "chi2_statistic": chi2,
                                "p_value": p_value,
                                "significant": p_value < 0.05,
                                "contingency_table": contingency.to_dict()
                            }
            
            # McNemar's test for paired comparisons (if same dates available)
            # This would be more complex to implement and requires careful date matching
            
        except Exception as e:
            statistical_tests["error"] = str(e)
        
        return statistical_tests
    
    def generate_thesis_visualizations(self, results: Dict[str, Any], crypto_symbol: str) -> List[Path]:
        """Generate comprehensive visualizations for thesis."""
        try:
            df = self._create_analysis_dataframe(results)
            if df.empty:
                return []
            
            chart_paths = []
            
            # 1. Overall Accuracy Comparison
            chart_paths.append(self._plot_accuracy_comparison(df, crypto_symbol))
            
            # 2. Confidence vs Accuracy Analysis
            chart_paths.append(self._plot_confidence_accuracy(df, crypto_symbol))
            
            # 3. Temporal Performance Analysis
            chart_paths.append(self._plot_temporal_analysis(df, crypto_symbol))
            
            # 4. Confusion Matrix Heatmaps
            chart_paths.append(self._plot_confusion_matrices(df, crypto_symbol))
            
            # 5. Method Performance Distribution
            chart_paths.append(self._plot_performance_distribution(df, crypto_symbol))
            
            return [path for path in chart_paths if path]
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            return []
    
    def _plot_accuracy_comparison(self, df: pd.DataFrame, crypto_symbol: str) -> Optional[Path]:
        """Plot accuracy comparison between methods."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Calculate accuracy by method
            method_accuracy = df[df["success"] == True].groupby("method")["correct_prediction"].mean()
            method_counts = df[df["success"] == True].groupby("method").size()
            
            # Create subplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy bar plot
            bars = ax1.bar(method_accuracy.index, method_accuracy.values, alpha=0.8)
            ax1.set_title(f'Prediction Accuracy by Method - {crypto_symbol.upper()}', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Accuracy Rate', fontsize=12)
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, method_accuracy.values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Sample size bar plot
            bars2 = ax2.bar(method_counts.index, method_counts.values, alpha=0.8, color='orange')
            ax2.set_title(f'Sample Sizes by Method - {crypto_symbol.upper()}', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Number of Successful Predictions', fontsize=12)
            
            # Add value labels
            for bar, value in zip(bars2, method_counts.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_accuracy_comparison.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to plot accuracy comparison: {e}")
            return None
    
    def _plot_confidence_accuracy(self, df: pd.DataFrame, crypto_symbol: str) -> Optional[Path]:
        """Plot confidence vs accuracy analysis."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            methods = ["full_agentic", "image_only", "sentiment_only"]
            
            for i, method in enumerate(methods):
                method_df = df[(df["method"] == method) & (df["success"] == True)]
                
                if len(method_df) > 0:
                    conf_accuracy = method_df.groupby("confidence")["correct_prediction"].mean()
                    conf_counts = method_df.groupby("confidence").size()
                    
                    # Create grouped bar chart
                    x_pos = np.arange(len(conf_accuracy))
                    bars = axes[i].bar(x_pos, conf_accuracy.values, alpha=0.8)
                    
                    axes[i].set_title(f'{method.replace("_", " ").title()}\nConfidence vs Accuracy', 
                                    fontsize=12, fontweight='bold')
                    axes[i].set_ylabel('Accuracy Rate')
                    axes[i].set_xlabel('Confidence Level')
                    axes[i].set_xticks(x_pos)
                    axes[i].set_xticklabels(conf_accuracy.index)
                    axes[i].set_ylim(0, 1)
                    
                    # Add value and count labels
                    for j, (bar, acc, count) in enumerate(zip(bars, conf_accuracy.values, conf_counts.values)):
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                                   f'{acc:.3f}\n(n={count})', ha='center', va='bottom', fontweight='bold')
                else:
                    axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{method.replace("_", " ").title()}\nNo Data Available')
            
            plt.suptitle(f'Confidence vs Accuracy Analysis - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_confidence_accuracy.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to plot confidence accuracy: {e}")
            return None
    
    def _plot_temporal_analysis(self, df: pd.DataFrame, crypto_symbol: str) -> Optional[Path]:
        """Plot temporal performance analysis."""
        try:
            if len(df) == 0:
                return None
                
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Accuracy over time (rolling average)
            df_success = df[df["success"] == True].copy()
            df_success = df_success.sort_values("date")
            
            for method in ["full_agentic", "image_only", "sentiment_only"]:
                method_df = df_success[df_success["method"] == method]
                if len(method_df) > 10:
                    rolling_acc = method_df.set_index("date")["correct_prediction"].rolling(window=14, min_periods=5).mean()
                    axes[0,0].plot(rolling_acc.index, rolling_acc.values, label=method.replace("_", " ").title(), linewidth=2)
            
            axes[0,0].set_title('Accuracy Trend Over Time (14-day Rolling Average)', fontweight='bold')
            axes[0,0].set_ylabel('Accuracy Rate')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # 2. Monthly performance
            df_success["month"] = df_success["date"].dt.month
            monthly_acc = df_success.groupby(["method", "month"])["correct_prediction"].mean().unstack(level=0, fill_value=0)
            
            if not monthly_acc.empty:
                monthly_acc.plot(kind='bar', ax=axes[0,1], alpha=0.8)
                axes[0,1].set_title('Monthly Accuracy Comparison', fontweight='bold')
                axes[0,1].set_ylabel('Accuracy Rate')
                axes[0,1].set_xlabel('Month')
                axes[0,1].legend(title='Method')
                axes[0,1].tick_params(axis='x', rotation=0)
            
            # 3. Day of week performance
            df_success["day_of_week"] = df_success["date"].dt.dayofweek
            dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            dow_acc = df_success.groupby(["method", "day_of_week"])["correct_prediction"].mean().unstack(level=0, fill_value=0)
            
            if not dow_acc.empty:
                dow_acc.index = [dow_names[i] for i in dow_acc.index]
                dow_acc.plot(kind='bar', ax=axes[1,0], alpha=0.8)
                axes[1,0].set_title('Day of Week Accuracy', fontweight='bold')
                axes[1,0].set_ylabel('Accuracy Rate')
                axes[1,0].set_xlabel('Day of Week')
                axes[1,0].legend(title='Method')
                axes[1,0].tick_params(axis='x', rotation=45)
            
            # 4. Sample size over time
            sample_counts = df_success.groupby([df_success["date"].dt.to_period("M"), "method"]).size().unstack(level=1, fill_value=0)
            
            if not sample_counts.empty:
                sample_counts.plot(kind='line', ax=axes[1,1], marker='o')
                axes[1,1].set_title('Sample Sizes Over Time', fontweight='bold')
                axes[1,1].set_ylabel('Number of Predictions')
                axes[1,1].set_xlabel('Month')
                axes[1,1].legend(title='Method')
                axes[1,1].grid(True, alpha=0.3)
            
            plt.suptitle(f'Temporal Analysis - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_temporal_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to plot temporal analysis: {e}")
            return None
    
    def _plot_confusion_matrices(self, df: pd.DataFrame, crypto_symbol: str) -> Optional[Path]:
        """Plot confusion matrices for each method."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            methods = ["full_agentic", "image_only", "sentiment_only"]
            
            for i, method in enumerate(methods):
                method_df = df[(df["method"] == method) & (df["success"] == True)]
                
                if len(method_df) > 0:
                    # Create confusion matrix
                    conf_matrix = pd.crosstab(
                        method_df["actual_direction"], 
                        method_df["predicted_direction"],
                        normalize='index'
                    )
                    
                    # Plot heatmap
                    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', 
                              ax=axes[i], cbar_kws={'label': 'Proportion'})
                    axes[i].set_title(f'{method.replace("_", " ").title()}\nConfusion Matrix', fontweight='bold')
                    axes[i].set_ylabel('Actual Direction')
                    axes[i].set_xlabel('Predicted Direction')
                else:
                    axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{method.replace("_", " ").title()}\nNo Data Available')
            
            plt.suptitle(f'Confusion Matrices - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_confusion_matrices.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to plot confusion matrices: {e}")
            return None
    
    def _plot_performance_distribution(self, df: pd.DataFrame, crypto_symbol: str) -> Optional[Path]:
        """Plot performance distribution analysis."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            df_success = df[df["success"] == True]
            
            # 1. Success rate by method
            success_rates = df.groupby("method")["success"].mean()
            bars1 = axes[0,0].bar(success_rates.index, success_rates.values, alpha=0.8, color='lightcoral')
            axes[0,0].set_title('Success Rate by Method', fontweight='bold')
            axes[0,0].set_ylabel('Success Rate')
            axes[0,0].set_ylim(0, 1)
            
            for bar, value in zip(bars1, success_rates.values):
                axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                              f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 2. Prediction distribution by actual direction
            pred_dist = df_success.groupby(["method", "actual_direction"])["correct_prediction"].mean().unstack(level=1, fill_value=0)
            
            if not pred_dist.empty:
                pred_dist.plot(kind='bar', ax=axes[0,1], alpha=0.8)
                axes[0,1].set_title('Accuracy by Actual Market Direction', fontweight='bold')
                axes[0,1].set_ylabel('Accuracy Rate')
                axes[0,1].legend(title='Actual Direction')
                axes[0,1].tick_params(axis='x', rotation=45)
            
            # 3. Confidence distribution
            conf_dist = df_success["confidence"].value_counts()
            axes[1,0].pie(conf_dist.values, labels=conf_dist.index, autopct='%1.1f%%', startangle=90)
            axes[1,0].set_title('Overall Confidence Distribution', fontweight='bold')
            
            # 4. Method vs Direction heatmap
            method_direction = pd.crosstab(df_success["method"], df_success["predicted_direction"])
            sns.heatmap(method_direction, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1,1])
            axes[1,1].set_title('Prediction Direction by Method', fontweight='bold')
            axes[1,1].set_ylabel('Method')
            axes[1,1].set_xlabel('Predicted Direction')
            
            plt.suptitle(f'Performance Distribution Analysis - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_performance_distribution.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to plot performance distribution: {e}")
            return None
    
    def generate_thesis_report(self, results: Dict[str, Any], crypto_symbol: str) -> Path:
        """Generate comprehensive thesis report."""
        try:
            analysis = self.analyze_prediction_accuracy(results)
            chart_paths = self.generate_thesis_visualizations(results, crypto_symbol)
            
            # Create markdown report
            report_path = self.analysis_dir / f"{crypto_symbol}_thesis_report.md"
            
            with open(report_path, 'w') as f:
                f.write(self._generate_markdown_report(analysis, chart_paths, crypto_symbol))
            
            # Also save as JSON
            json_path = self.analysis_dir / f"{crypto_symbol}_thesis_analysis.json"
            with open(json_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to generate thesis report: {e}")
            return None
    
    def _generate_markdown_report(self, analysis: Dict[str, Any], chart_paths: List[Path], crypto_symbol: str) -> str:
        """Generate markdown thesis report."""
        report = f"""# Thesis Analysis Report: {crypto_symbol.upper()}

## Multi-Agent vs One-Shot LLM Prediction Comparison

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report compares the performance of three cryptocurrency price prediction approaches:
1. **Full Agentic**: Multi-agent system using technical analysis, sentiment analysis, and data fusion
2. **Image Only**: One-shot LLM analysis of technical charts only  
3. **Sentiment Only**: One-shot LLM analysis of social sentiment only

## Overall Statistics

- **Total Predictions:** {analysis.get('overall_stats', {}).get('total_predictions', 0)}
- **Overall Accuracy:** {analysis.get('overall_stats', {}).get('overall_accuracy', 0):.3f}
- **Date Range:** {analysis.get('overall_stats', {}).get('date_range', {}).get('start', 'N/A')} to {analysis.get('overall_stats', {}).get('date_range', {}).get('end', 'N/A')}

## Method Comparison

"""
        
        # Add method comparison details
        method_comp = analysis.get('method_comparison', {})
        for method, stats in method_comp.items():
            if method != 'comparison_summary' and isinstance(stats, dict):
                report += f"""### {method.replace('_', ' ').title()}

- **Success Rate:** {stats.get('success_rate', 0):.3f}
- **Accuracy:** {stats.get('accuracy', 0):.3f}
- **Total Attempts:** {stats.get('total_attempts', 0)}
- **Successful Predictions:** {stats.get('successful_predictions', 0)}

"""

        # Add statistical significance
        if 'statistical_tests' in analysis:
            report += "\n## Statistical Significance Tests\n\n"
            for test_name, test_result in analysis['statistical_tests'].items():
                if isinstance(test_result, dict) and 'p_value' in test_result:
                    significance = "**Significant**" if test_result['p_value'] < 0.05 else "Not Significant"
                    report += f"- **{test_name}:** p-value = {test_result['p_value']:.4f} ({significance})\n"

        # Add charts
        report += "\n## Visualizations\n\n"
        for chart_path in chart_paths:
            if chart_path and chart_path.exists():
                chart_name = chart_path.stem.replace(f"{crypto_symbol}_", "").replace("_", " ").title()
                report += f"### {chart_name}\n\n![{chart_name}]({chart_path.name})\n\n"

        # Add conclusions
        report += """## Conclusions

### Key Findings

1. **Method Performance**: [Add analysis of which method performed best]
2. **Confidence Correlation**: [Add analysis of confidence vs accuracy]
3. **Temporal Patterns**: [Add analysis of performance over time]

### Implications for Thesis

This analysis provides evidence for [add conclusions about multi-agent vs one-shot approaches]

### Future Research

1. Extended time periods
2. Additional cryptocurrencies  
3. Market condition analysis
4. Cost-benefit analysis

---

*This report was generated automatically by the CryptoAgentForecaster thesis analysis system.*
"""
        
        return report 