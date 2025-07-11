"""
Enhanced pandas/numpy-based analyzer for cryptocurrency prediction backtesting.
Optimized for large-scale data processing and statistical analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import json
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PandasThesisAnalyzer:
    """Enhanced analyzer using pandas/numpy for efficient data processing."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.analysis_dir = self.data_dir / "analysis"
        self.charts_dir = self.data_dir / "charts"
        self.processed_dir = self.data_dir / "processed_data"
        
        # Ensure directories exist
        for dir_path in [self.analysis_dir, self.charts_dir, self.processed_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
    
    def load_backtest_data(self, crypto_symbol: str) -> pd.DataFrame:
        """Load and process backtest data into a comprehensive DataFrame."""
        results_file = self.data_dir / f"{crypto_symbol}_backtest_results.json"
        
        if not results_file.exists():
            logger.error(f"No backtest results found for {crypto_symbol}")
            return pd.DataFrame()
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Convert to structured DataFrame
        records = []
        
        for date_str, predictions in results.get("daily_predictions", {}).items():
            base_data = {
                'date': pd.to_datetime(date_str),
                'date_str': date_str
            }
            
            # Extract actual market data (same for all methods on a given day)
            actual_data = {}
            for method, pred in predictions.items():
                if pred.get("success") and pred.get("actual_movement_24h") is not None:
                    actual_data = {
                        'actual_movement_pct': pred["actual_movement_24h"],
                        'actual_price': pred.get("actual_price", 0),
                        'next_day_price': pred.get("next_day_price", 0),
                        'sentiment_posts': pred.get("data_quality", {}).get("sentiment_posts_count", 0)
                    }
                    break
            
            # Create records for each method
            for method in ["full_agentic", "image_only", "sentiment_only"]:
                if method in predictions:
                    pred = predictions[method]
                    
                    record = {**base_data, **actual_data}
                    record.update({
                        'method': method,
                        'predicted_direction': pred.get("predicted_direction", "FAILED"),
                        'confidence': pred.get("confidence", "NONE"),
                        'success': pred.get("success", False),
                        'execution_time': pred.get("execution_time"),
                        'error': pred.get("error", "")
                    })
                    
                    # Calculate derived features
                    if actual_data:
                        actual_movement = actual_data['actual_movement_pct']
                        
                        # Determine actual direction using threshold
                        if actual_movement > 1.0:
                            record['actual_direction'] = 'UP'
                            record['actual_numerical'] = 1
                        elif actual_movement < -1.0:
                            record['actual_direction'] = 'DOWN'
                            record['actual_numerical'] = -1
                        else:
                            # For small movements, assign to DOWN (conservative approach)
                            record['actual_direction'] = 'DOWN'
                            record['actual_numerical'] = -1
                        
                        # Convert predicted direction to numerical
                        pred_dir = record['predicted_direction']
                        if pred_dir == 'UP':
                            record['predicted_numerical'] = 1
                        elif pred_dir == 'DOWN':
                            record['predicted_numerical'] = -1
                        else:
                            # Default to DOWN for any other values
                            record['predicted_numerical'] = -1
                        
                        # Calculate if prediction was correct
                        record['correct_prediction'] = (
                            record['predicted_direction'] == record['actual_direction']
                            if record['success'] and pd.notna(record['predicted_numerical'])
                            else False
                        )
                    else:
                        # No actual data available
                        record.update({
                            'actual_direction': np.nan,
                            'actual_numerical': np.nan,
                            'predicted_numerical': np.nan,
                            'correct_prediction': False
                        })
                    
                    records.append(record)
        
        df = pd.DataFrame(records)
        
        if len(df) > 0:
            # Add additional features
            df['day_of_week'] = df['date'].dt.day_name()
            df['month'] = df['date'].dt.month
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['volatility'] = abs(df['actual_movement_pct'])
            
            # Sort by date and method for easier analysis
            df = df.sort_values(['date', 'method']).reset_index(drop=True)
        
        return df
    
    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics using pandas operations."""
        
        if df.empty:
            return {"error": "No data available for analysis"}
        
        # Filter successful predictions only
        success_df = df[df['success'] == True].copy()
        
        if success_df.empty:
            return {"error": "No successful predictions found"}
        
        metrics = {}
        
        # Overall statistics
        metrics['overall'] = {
            'total_attempts': len(df),
            'successful_attempts': len(success_df),
            'overall_success_rate': len(success_df) / len(df),
            'date_range': {
                'start': df['date'].min().strftime("%Y-%m-%d"),
                'end': df['date'].max().strftime("%Y-%m-%d"),
                'total_days': df['date'].nunique()
            }
        }
        
        # Method-specific analysis
        metrics['by_method'] = {}
        
        for method in ['full_agentic', 'image_only', 'sentiment_only']:
            method_df = success_df[success_df['method'] == method].copy()
            
            if len(method_df) == 0:
                metrics['by_method'][method] = {'no_data': True}
                continue
            
            # Basic metrics
            accuracy = method_df['correct_prediction'].mean()
            correct_count = method_df['correct_prediction'].sum()
            
            # Confusion matrix components
            conf_matrix = pd.crosstab(
                method_df['actual_direction'], 
                method_df['predicted_direction'], 
                margins=True
            )
            
            # Calculate precision, recall, F1 for each direction
            directions = ['UP', 'DOWN']
            precision_scores = {}
            recall_scores = {}
            f1_scores = {}
            
            for direction in directions:
                if direction in conf_matrix.columns and direction in conf_matrix.index:
                    tp = conf_matrix.loc[direction, direction] if direction in conf_matrix.columns else 0
                    fp = conf_matrix.loc['All', direction] - tp if 'All' in conf_matrix.index else 0
                    fn = conf_matrix.loc[direction, 'All'] - tp if 'All' in conf_matrix.columns else 0
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    precision_scores[direction] = precision
                    recall_scores[direction] = recall
                    f1_scores[direction] = f1
            
            # Confidence analysis
            confidence_stats = {}
            for conf_level in ['HIGH', 'MEDIUM', 'LOW']:
                conf_data = method_df[method_df['confidence'] == conf_level]
                if len(conf_data) > 0:
                    confidence_stats[conf_level] = {
                        'count': len(conf_data),
                        'accuracy': conf_data['correct_prediction'].mean(),
                        'sample_size': len(conf_data)
                    }
            
            # Directional bias analysis
            pred_distribution = method_df['predicted_direction'].value_counts(normalize=True)
            actual_distribution = method_df['actual_direction'].value_counts(normalize=True)
            
            metrics['by_method'][method] = {
                'total_predictions': len(method_df),
                'accuracy': accuracy,
                'correct_predictions': int(correct_count),
                'confusion_matrix': conf_matrix.to_dict(),
                'precision_by_direction': precision_scores,
                'recall_by_direction': recall_scores,
                'f1_by_direction': f1_scores,
                'confidence_analysis': confidence_stats,
                'prediction_distribution': pred_distribution.to_dict(),
                'actual_distribution': actual_distribution.to_dict(),
                'avg_execution_time': method_df['execution_time'].mean() if method_df['execution_time'].notna().any() else None
            }
        
        # Statistical significance testing
        metrics['statistical_tests'] = self._calculate_statistical_significance(success_df)
        
        # Correlation analysis
        metrics['correlation_analysis'] = self._calculate_correlations(success_df)
        
        # Temporal analysis
        metrics['temporal_analysis'] = self._calculate_temporal_patterns(success_df)
        
        return metrics
    
    def _calculate_statistical_significance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical significance between methods."""
        tests = {}
        
        methods = ['full_agentic', 'image_only', 'sentiment_only']
        
        # Pairwise comparisons
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                method1_data = df[df['method'] == method1]['correct_prediction']
                method2_data = df[df['method'] == method2]['correct_prediction']
                
                if len(method1_data) > 5 and len(method2_data) > 5:
                    # Chi-square test for independence
                    contingency = pd.crosstab(
                        pd.concat([method1_data, method2_data]),
                        pd.concat([
                            pd.Series([method1] * len(method1_data)),
                            pd.Series([method2] * len(method2_data))
                        ])
                    )
                    
                    if contingency.shape == (2, 2):
                        try:
                            chi2, p_value = chi2_contingency(contingency)[:2]
                            tests[f"{method1}_vs_{method2}"] = {
                                'test_type': 'chi_square',
                                'chi2_statistic': chi2,
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                                'sample_sizes': {method1: len(method1_data), method2: len(method2_data)}
                            }
                        except ValueError as e:
                            tests[f"{method1}_vs_{method2}"] = {'error': str(e)}
        
        # Overall correlation test
        # Remove duplicates before pivot to avoid reindex issues
        df_dedupe = df.drop_duplicates(subset=['date_str', 'method']).copy()
        
        pivot_df = df_dedupe.pivot_table(
            index='date_str', 
            columns='method', 
            values='predicted_numerical', 
            aggfunc='first'
        )
        
        actual_by_date = df_dedupe.groupby('date_str')['actual_numerical'].first()
        
        # Correlation with actual outcomes
        correlations = {}
        for method in methods:
            if method in pivot_df.columns:
                method_predictions = pivot_df[method].dropna()
                aligned_actual = actual_by_date.loc[method_predictions.index].dropna()
                
                if len(method_predictions) > 3 and len(aligned_actual) > 3:
                    # Align the data
                    common_dates = method_predictions.index.intersection(aligned_actual.index)
                    if len(common_dates) > 3:
                        corr_coef, p_val = stats.pearsonr(
                            method_predictions.loc[common_dates],
                            aligned_actual.loc[common_dates]
                        )
                        correlations[method] = {
                            'correlation': corr_coef,
                            'p_value': p_val,
                            'significant': p_val < 0.05,
                            'sample_size': len(common_dates)
                        }
        
        tests['correlation_with_actual'] = correlations
        
        return tests
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various correlation analyses."""
        correlations = {}
        
        # Correlation between sentiment volume and accuracy
        sentiment_df = df[df['method'] == 'sentiment_only'].copy()
        if len(sentiment_df) > 3:
            corr_sent_posts = sentiment_df[['sentiment_posts', 'correct_prediction']].corr()
            correlations['sentiment_volume_accuracy'] = {
                'correlation': corr_sent_posts.iloc[0, 1],
                'interpretation': 'Correlation between number of sentiment posts and prediction accuracy'
            }
        
        # Volatility vs prediction accuracy
        if 'volatility' in df.columns:
            vol_corr = df[['volatility', 'correct_prediction']].corr()
            correlations['volatility_accuracy'] = {
                'correlation': vol_corr.iloc[0, 1],
                'interpretation': 'Correlation between market volatility and prediction accuracy'
            }
        
        return correlations
    
    def _calculate_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in prediction accuracy."""
        temporal = {}
        
        # Day of week analysis
        dow_accuracy = df.groupby(['method', 'day_of_week'])['correct_prediction'].agg(['mean', 'count']).round(3)
        temporal['day_of_week'] = dow_accuracy.to_dict()
        
        # Monthly analysis  
        monthly_accuracy = df.groupby(['method', 'month'])['correct_prediction'].agg(['mean', 'count']).round(3)
        temporal['monthly'] = monthly_accuracy.to_dict()
        
        # Rolling accuracy (if enough data)
        if len(df) > 14:
            df_sorted = df.sort_values('date')
            rolling_metrics = {}
            
            for method in df['method'].unique():
                method_data = df_sorted[df_sorted['method'] == method].copy()
                if len(method_data) > 7:
                    method_data['rolling_accuracy'] = method_data['correct_prediction'].rolling(
                        window=7, min_periods=3
                    ).mean()
                    rolling_metrics[method] = method_data[['date_str', 'rolling_accuracy']].to_dict('records')
            
            temporal['rolling_accuracy'] = rolling_metrics
        
        return temporal
    
    def generate_enhanced_visualizations(self, df: pd.DataFrame, crypto_symbol: str) -> List[Path]:
        """Generate comprehensive visualizations using pandas plotting capabilities."""
        
        if df.empty:
            logger.warning("No data available for visualization")
            return []
        
        chart_paths = []
        success_df = df[df['success'] == True].copy()
        
        if success_df.empty:
            logger.warning("No successful predictions for visualization")
            return []
        
        # 1. Performance Dashboard
        chart_paths.append(self._create_performance_dashboard(success_df, crypto_symbol))
        
        # 2. Correlation Matrix
        chart_paths.append(self._create_correlation_matrix(success_df, crypto_symbol))
        
        # 3. Temporal Analysis
        chart_paths.append(self._create_temporal_analysis(success_df, crypto_symbol))
        
        # 4. Distribution Analysis
        chart_paths.append(self._create_distribution_analysis(success_df, crypto_symbol))
        
        # 5. Method Comparison Heatmap
        chart_paths.append(self._create_method_comparison_heatmap(success_df, crypto_symbol))
        
        return [p for p in chart_paths if p and p.exists()]
    
    def _create_performance_dashboard(self, df: pd.DataFrame, crypto_symbol: str) -> Optional[Path]:
        """Create comprehensive performance dashboard."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Performance Dashboard - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            
            # 1. Accuracy by Method
            accuracy_data = df.groupby('method')['correct_prediction'].agg(['mean', 'count'])
            bars = axes[0, 0].bar(accuracy_data.index, accuracy_data['mean'], 
                                 color=['#2E86C1', '#E74C3C', '#F39C12'], alpha=0.8)
            axes[0, 0].set_title('Accuracy by Method', fontweight='bold')
            axes[0, 0].set_ylabel('Accuracy Rate')
            axes[0, 0].set_ylim(0, 1)
            
            # Add sample size annotations
            for i, (bar, count) in enumerate(zip(bars, accuracy_data['count'])):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{bar.get_height():.3f}\n(n={count})', 
                               ha='center', va='bottom', fontweight='bold')
            
            # 2. Confidence vs Accuracy
            conf_accuracy = df.groupby(['method', 'confidence'])['correct_prediction'].mean().unstack(level=0)
            if not conf_accuracy.empty:
                conf_accuracy.plot(kind='bar', ax=axes[0, 1], alpha=0.8)
                axes[0, 1].set_title('Confidence vs Accuracy', fontweight='bold')
                axes[0, 1].set_ylabel('Accuracy Rate')
                axes[0, 1].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. Prediction Distribution
            pred_dist = df.groupby(['method', 'predicted_direction']).size().unstack(level=0, fill_value=0)
            if not pred_dist.empty:
                pred_dist.plot(kind='bar', ax=axes[0, 2], alpha=0.8)
                axes[0, 2].set_title('Prediction Distribution', fontweight='bold')
                axes[0, 2].set_ylabel('Count')
                axes[0, 2].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[0, 2].tick_params(axis='x', rotation=45)
            
            # 4. Success Rate by Method
            success_rate = df.groupby('method')['success'].mean()
            bars2 = axes[1, 0].bar(success_rate.index, success_rate.values, 
                                  color=['#2E86C1', '#E74C3C', '#F39C12'], alpha=0.8)
            axes[1, 0].set_title('Success Rate by Method', fontweight='bold')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].set_ylim(0, 1)
            
            for bar in bars2:
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{bar.get_height():.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 5. Accuracy over Time
            if len(df) > 7:
                daily_accuracy = df.groupby(['date', 'method'])['correct_prediction'].mean().unstack(level=1)
                if not daily_accuracy.empty:
                    daily_accuracy.plot(ax=axes[1, 1], marker='o', alpha=0.7)
                    axes[1, 1].set_title('Accuracy Over Time', fontweight='bold')
                    axes[1, 1].set_ylabel('Accuracy Rate')
                    axes[1, 1].legend(title='Method')
                    axes[1, 1].tick_params(axis='x', rotation=45)
            
            # 6. Market Direction Distribution
            market_dist = df['actual_direction'].value_counts()
            if len(market_dist) > 0:
                axes[1, 2].pie(market_dist.values, labels=market_dist.index, autopct='%1.1f%%', startangle=90)
                axes[1, 2].set_title('Actual Market Direction Distribution', fontweight='bold')
            
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_performance_dashboard.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to create performance dashboard: {e}")
            return None
    
    def _create_correlation_matrix(self, df: pd.DataFrame, crypto_symbol: str) -> Optional[Path]:
        """Create correlation matrix visualization."""
        try:
            # Prepare numerical data
            numerical_cols = ['predicted_numerical', 'actual_numerical', 'actual_movement_pct', 
                            'sentiment_posts', 'volatility']
            
            # Filter available columns
            available_cols = [col for col in numerical_cols if col in df.columns]
            
            if len(available_cols) < 2:
                return None
            
            # Calculate correlation matrix for each method
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'Correlation Analysis - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            
            methods = ['full_agentic', 'image_only', 'sentiment_only']
            
            for i, method in enumerate(methods):
                method_df = df[df['method'] == method][available_cols]
                
                if len(method_df) > 3:
                    corr_matrix = method_df.corr()
                    
                    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                              ax=axes[i], cbar_kws={'label': 'Correlation'})
                    axes[i].set_title(f'{method.replace("_", " ").title()}', fontweight='bold')
                else:
                    axes[i].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center',
                               transform=axes[i].transAxes, fontsize=14)
                    axes[i].set_title(f'{method.replace("_", " ").title()}', fontweight='bold')
            
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_correlation_matrix.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to create correlation matrix: {e}")
            return None
    
    def _create_temporal_analysis(self, df: pd.DataFrame, crypto_symbol: str) -> Optional[Path]:
        """Create temporal pattern analysis."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Temporal Analysis - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            
            # 1. Day of week accuracy
            dow_accuracy = df.groupby(['method', 'day_of_week'])['correct_prediction'].mean().unstack(level=0)
            if not dow_accuracy.empty:
                dow_accuracy.plot(kind='bar', ax=axes[0, 0], alpha=0.8)
                axes[0, 0].set_title('Accuracy by Day of Week', fontweight='bold')
                axes[0, 0].set_ylabel('Accuracy Rate')
                axes[0, 0].legend(title='Method')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Success rate over time
            if len(df['date'].unique()) > 3:
                daily_success = df.groupby(['date', 'method'])['success'].mean().unstack(level=1)
                if not daily_success.empty:
                    daily_success.plot(ax=axes[0, 1], marker='o', alpha=0.7)
                    axes[0, 1].set_title('Success Rate Over Time', fontweight='bold')
                    axes[0, 1].set_ylabel('Success Rate')
                    axes[0, 1].legend(title='Method')
                    axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. Volatility vs Accuracy
            if 'volatility' in df.columns:
                for method in df['method'].unique():
                    method_data = df[df['method'] == method]
                    if len(method_data) > 3:
                        axes[1, 0].scatter(method_data['volatility'], method_data['correct_prediction'], 
                                         label=method, alpha=0.6, s=50)
                
                axes[1, 0].set_title('Volatility vs Accuracy', fontweight='bold')
                axes[1, 0].set_xlabel('Market Volatility (%)')
                axes[1, 0].set_ylabel('Correct Prediction (1/0)')
                axes[1, 0].legend(title='Method')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Sample sizes over time
            daily_counts = df.groupby(['date', 'method']).size().unstack(level=1, fill_value=0)
            if not daily_counts.empty:
                daily_counts.plot(kind='bar', ax=axes[1, 1], alpha=0.8)
                axes[1, 1].set_title('Daily Sample Sizes', fontweight='bold')
                axes[1, 1].set_ylabel('Number of Predictions')
                axes[1, 1].legend(title='Method')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_temporal_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to create temporal analysis: {e}")
            return None
    
    def _create_distribution_analysis(self, df: pd.DataFrame, crypto_symbol: str) -> Optional[Path]:
        """Create distribution analysis charts."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Distribution Analysis - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            
            # 1. Confidence distribution
            conf_counts = df.groupby(['method', 'confidence']).size().unstack(level=0, fill_value=0)
            if not conf_counts.empty:
                conf_counts.plot(kind='bar', ax=axes[0, 0], alpha=0.8)
                axes[0, 0].set_title('Confidence Distribution by Method', fontweight='bold')
                axes[0, 0].set_ylabel('Count')
                axes[0, 0].legend(title='Method')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Accuracy distribution
            method_accuracies = df.groupby('method')['correct_prediction'].apply(list)
            for i, (method, accuracies) in enumerate(method_accuracies.items()):
                axes[0, 1].hist(accuracies, alpha=0.6, label=method, bins=2)
            axes[0, 1].set_title('Accuracy Distribution', fontweight='bold')
            axes[0, 1].set_xlabel('Prediction Accuracy')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend(title='Method')
            
            # 3. Market movement distribution
            if 'actual_movement_pct' in df.columns:
                unique_movements = df.drop_duplicates('date')['actual_movement_pct'].dropna()
                axes[1, 0].hist(unique_movements, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1, 0].set_title('Market Movement Distribution', fontweight='bold')
                axes[1, 0].set_xlabel('24h Price Movement (%)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.8)
            
            # 4. Method performance comparison
            perf_data = df.groupby('method')['correct_prediction'].agg(['mean', 'std', 'count'])
            x_pos = range(len(perf_data))
            axes[1, 1].bar(x_pos, perf_data['mean'], yerr=perf_data['std'], 
                          alpha=0.8, capsize=10, color=['#2E86C1', '#E74C3C', '#F39C12'])
            axes[1, 1].set_title('Method Performance with Error Bars', fontweight='bold')
            axes[1, 1].set_xlabel('Method')
            axes[1, 1].set_ylabel('Accuracy Rate')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(perf_data.index, rotation=45)
            axes[1, 1].set_ylim(0, 1)
            
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_distribution_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to create distribution analysis: {e}")
            return None
    
    def _create_method_comparison_heatmap(self, df: pd.DataFrame, crypto_symbol: str) -> Optional[Path]:
        """Create method comparison heatmap."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'Method Comparison Analysis - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            
            # 1. Accuracy heatmap by actual direction
            accuracy_matrix = df.groupby(['method', 'actual_direction'])['correct_prediction'].mean().unstack(level=1)
            if not accuracy_matrix.empty:
                sns.heatmap(accuracy_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                           ax=axes[0], cbar_kws={'label': 'Accuracy Rate'})
                axes[0].set_title('Accuracy by Method and Market Direction', fontweight='bold')
                axes[0].set_xlabel('Actual Market Direction')
                axes[0].set_ylabel('Prediction Method')
            
            # 2. Sample size heatmap
            sample_matrix = df.groupby(['method', 'actual_direction']).size().unstack(level=1, fill_value=0)
            if not sample_matrix.empty:
                sns.heatmap(sample_matrix, annot=True, fmt='d', cmap='Blues', 
                           ax=axes[1], cbar_kws={'label': 'Sample Size'})
                axes[1].set_title('Sample Sizes by Method and Market Direction', fontweight='bold')
                axes[1].set_xlabel('Actual Market Direction')
                axes[1].set_ylabel('Prediction Method')
            
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_method_comparison_heatmap.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to create method comparison heatmap: {e}")
            return None
    
    def export_analysis_data(self, df: pd.DataFrame, metrics: Dict[str, Any], crypto_symbol: str) -> List[Path]:
        """Export analysis data in multiple formats."""
        export_paths = []
        
        try:
            # 1. Enhanced CSV export
            if not df.empty:
                csv_path = self.processed_dir / f"{crypto_symbol}_enhanced_analysis.csv"
                df.to_csv(csv_path, index=False)
                export_paths.append(csv_path)
            
            # 2. Metrics JSON export
            metrics_path = self.analysis_dir / f"{crypto_symbol}_comprehensive_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            export_paths.append(metrics_path)
            
            # 3. Summary statistics
            if not df.empty:
                summary_stats = df.describe(include='all')
                summary_path = self.analysis_dir / f"{crypto_symbol}_summary_statistics.csv"
                summary_stats.to_csv(summary_path)
                export_paths.append(summary_path)
            
            # 4. Method comparison table
            if not df.empty:
                success_df = df[df['success'] == True]
                if not success_df.empty:
                    comparison = success_df.groupby('method').agg({
                        'correct_prediction': ['mean', 'count', 'sum'],
                        'confidence': lambda x: x.value_counts().to_dict(),
                        'actual_movement_pct': ['mean', 'std']
                    }).round(4)
                    
                    comparison_path = self.analysis_dir / f"{crypto_symbol}_method_comparison.csv"
                    comparison.to_csv(comparison_path)
                    export_paths.append(comparison_path)
            
            return export_paths
            
        except Exception as e:
            logger.error(f"Failed to export analysis data: {e}")
            return []
    
    def run_complete_analysis(self, crypto_symbol: str) -> Dict[str, Any]:
        """Run complete pandas-based analysis pipeline."""
        try:
            logger.info(f"Starting comprehensive pandas analysis for {crypto_symbol}")
            
            # Load data
            df = self.load_backtest_data(crypto_symbol)
            if df.empty:
                return {"error": "No data available for analysis"}
            
            logger.info(f"Loaded {len(df)} records for analysis")
            
            # Calculate metrics
            metrics = self.calculate_performance_metrics(df)
            
            # Generate visualizations
            chart_paths = self.generate_enhanced_visualizations(df, crypto_symbol)
            
            # Export data
            export_paths = self.export_analysis_data(df, metrics, crypto_symbol)
            
            # Add file paths to results
            metrics['generated_files'] = {
                'charts': [str(p) for p in chart_paths],
                'exports': [str(p) for p in export_paths],
                'data_shape': df.shape,
                'analysis_date': datetime.now().isoformat()
            }
            
            logger.info(f"Analysis completed. Generated {len(chart_paths)} charts and {len(export_paths)} export files")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Complete analysis failed: {e}")
            return {"error": str(e)} 