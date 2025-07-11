"""
Backtesting package for CryptoAgentForecaster thesis research.
"""

from .framework import BacktestingFramework
from .methods import PredictionMethods
from .data_collector import DataCollector
from .analyzer import ThesisAnalyzer

__all__ = [
    "BacktestingFramework",
    "PredictionMethods", 
    "DataCollector",
    "ThesisAnalyzer",
] 