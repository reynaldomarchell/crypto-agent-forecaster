#!/usr/bin/env python3
"""
Enhanced Analysis Runner for CryptoAgentForecaster

This script runs both the existing data analysis and a new one-week backtest
using the pandas/numpy-based analyzer.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.backtesting.framework import BacktestingFramework
from src.backtesting.analyzer import ThesisAnalyzer
from src.backtesting.pandas_analyzer import PandasThesisAnalyzer
from src.core.logging_config import LoggingConfig

def main():
    """Run enhanced analysis pipeline."""
    
    # Setup logging
    LoggingConfig.setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 Enhanced CryptoAgentForecaster Analysis Pipeline")
    print("=" * 60)
    
    crypto_symbol = "bitcoin"
    data_dir = Path("thesis_data")
    
    # Initialize analyzers
    pandas_analyzer = PandasThesisAnalyzer(data_dir)
    original_analyzer = ThesisAnalyzer(data_dir)
    
    print("\n📊 STEP 1: Analyzing Existing Data")
    print("-" * 40)
    
    # Check if we have existing data
    results_file = data_dir / f"{crypto_symbol}_backtest_results.json"
    
    if results_file.exists():
        print(f"✅ Found existing backtest data: {results_file}")
        
        # Run pandas analysis on existing data
        print("\n📈 Running pandas-based analysis...")
        pandas_results = pandas_analyzer.run_complete_analysis(crypto_symbol)
        
        if "error" not in pandas_results:
            print(f"✅ Pandas analysis completed successfully!")
            print(f"   📊 Processed data shape: {pandas_results['generated_files']['data_shape']}")
            print(f"   📈 Generated {len(pandas_results['generated_files']['charts'])} charts")
            print(f"   📄 Created {len(pandas_results['generated_files']['exports'])} export files")
            
            # Print key metrics
            if 'by_method' in pandas_results:
                print("\n📊 Key Performance Metrics:")
                for method, metrics in pandas_results['by_method'].items():
                    if 'accuracy' in metrics:
                        print(f"   {method}: {metrics['accuracy']:.3f} accuracy ({metrics['total_predictions']} samples)")
        else:
            print(f"❌ Pandas analysis failed: {pandas_results['error']}")
        
        # Run original analysis to fill any gaps
        print("\n🔄 Running original analysis for comparison...")
        try:
            with open(results_file, 'r') as f:
                backtest_data = json.load(f)
            
            original_results = original_analyzer.analyze_prediction_accuracy(backtest_data)
            if "error" not in original_results:
                print("✅ Original analysis completed successfully!")
                
                # Generate visualizations
                chart_paths = original_analyzer.generate_thesis_visualizations(backtest_data, crypto_symbol)
                print(f"   📈 Generated {len(chart_paths)} original charts")
                
                # Generate report
                report_path = original_analyzer.generate_thesis_report(backtest_data, crypto_symbol)
                if report_path:
                    print(f"   📝 Report generated: {report_path}")
            else:
                print(f"❌ Original analysis failed: {original_results['error']}")
                
        except Exception as e:
            print(f"❌ Original analysis error: {e}")
    
    else:
        print(f"❌ No existing backtest data found at {results_file}")
    
    print("\n🔬 STEP 2: Running New One-Week Backtest")
    print("-" * 40)
    
    # Run a fresh one-week backtest
    print("🚀 Starting new one-week backtest...")
    
    # Use recent dates that should have good data availability
    end_date = datetime(2024, 12, 28)  # End of December 2024
    start_date = end_date - timedelta(days=7)  # One week prior
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    print(f"📅 Date range: {start_date_str} to {end_date_str}")
    
    try:
        # Initialize framework
        framework = BacktestingFramework(crypto_symbol=crypto_symbol, data_dir=str(data_dir))
        
        # Run backtest
        print("⚙️  Running backtest...")
        results = framework.run_backtest(
            start_date=start_date_str,
            end_date=end_date_str,
            methods=["full_agentic", "image_only", "sentiment_only"],
            skip_existing=False  # Force fresh run
        )
        
        if results and "error" not in results:
            print("✅ New backtest completed successfully!")
            
            # Analyze new results with pandas
            print("\n📊 Analyzing new results with pandas...")
            new_pandas_results = pandas_analyzer.run_complete_analysis(crypto_symbol)
            
            if "error" not in new_pandas_results:
                print("✅ New pandas analysis completed!")
                
                # Print summary
                if 'overall' in new_pandas_results:
                    overall = new_pandas_results['overall']
                    print(f"\n📊 New Backtest Summary:")
                    print(f"   📈 Total attempts: {overall['total_attempts']}")
                    print(f"   ✅ Successful attempts: {overall['successful_attempts']}")
                    print(f"   📊 Success rate: {overall['overall_success_rate']:.3f}")
                    print(f"   📅 Date range: {overall['date_range']['start']} to {overall['date_range']['end']}")
                
                # Show method performance
                if 'by_method' in new_pandas_results:
                    print("\n🎯 Method Performance:")
                    for method, metrics in new_pandas_results['by_method'].items():
                        if 'accuracy' in metrics:
                            print(f"   {method}: {metrics['accuracy']:.3f} accuracy ({metrics['total_predictions']} predictions)")
                
                # Show statistical significance
                if 'statistical_tests' in new_pandas_results:
                    corr_tests = new_pandas_results['statistical_tests'].get('correlation_with_actual', {})
                    if corr_tests:
                        print("\n📈 Correlation with Actual Outcomes:")
                        for method, test in corr_tests.items():
                            significance = "**significant**" if test.get('significant') else "not significant"
                            print(f"   {method}: r={test['correlation']:.3f}, p={test['p_value']:.4f} ({significance})")
            
            else:
                print(f"❌ New pandas analysis failed: {new_pandas_results['error']}")
        
        else:
            print(f"❌ New backtest failed: {results.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        logger.error(f"Backtest failed: {e}")
    
    print("\n📁 STEP 3: Summary of Generated Files")
    print("-" * 40)
    
    # List all files in thesis_data
    def list_directory_contents(directory, title):
        print(f"\n📂 {title}:")
        if directory.exists():
            files = list(directory.iterdir())
            if files:
                for file_path in sorted(files):
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                        print(f"   📄 {file_path.name} ({size_str})")
                    elif file_path.is_dir():
                        print(f"   📁 {file_path.name}/")
            else:
                print("   (empty)")
        else:
            print("   (directory does not exist)")
    
    list_directory_contents(data_dir, "Main Data Directory")
    list_directory_contents(data_dir / "analysis", "Analysis Results")
    list_directory_contents(data_dir / "charts", "Generated Charts")
    list_directory_contents(data_dir / "processed_data", "Processed Data")
    
    print("\n✅ Enhanced Analysis Pipeline Complete!")
    print("=" * 60)
    print(f"📁 All results saved to: {data_dir.absolute()}")
    print("🎯 Key files to review:")
    print(f"   📊 Enhanced CSV: {data_dir}/processed_data/{crypto_symbol}_enhanced_analysis.csv")
    print(f"   📈 Performance Dashboard: {data_dir}/charts/{crypto_symbol}_performance_dashboard.png")
    print(f"   📝 Comprehensive Metrics: {data_dir}/analysis/{crypto_symbol}_comprehensive_metrics.json")
    print(f"   📋 Original Report: {data_dir}/analysis/{crypto_symbol}_thesis_report.md")

if __name__ == "__main__":
    main() 