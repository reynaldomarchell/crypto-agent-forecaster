"""
Configuration for analysis modules (technical and sentiment analysis).
"""

from typing import Dict, Any, List


class AnalysisConfig:
    """Configuration for technical and sentiment analysis."""
    
    # Technical Analysis Configuration
    TA_INDICATORS: Dict[str, Any] = {
        "sma_periods": [20, 50, 100, 200],  # Extended SMA periods
        "ema_periods": [9, 12, 26, 50],     # Extended EMA periods for professional analysis
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2,
        "volume_sma_period": 20,             # Volume SMA overlay
        "professional_colors": {             # Professional color scheme matching TradingView
            "ema_9": "#00D4AA",              # Cyan
            "ema_12": "#00CED1",             # Light cyan  
            "ema_21": "#20B2AA",             # Light sea green
            "sma_20": "#FFD700",             # Gold
            "ema_26": "#FF8C00",             # Orange
            "sma_50": "#DA70D6",             # Purple
            "sma_100": "#9370DB",            # Medium purple
            "sma_200": "#8A2BE2",            # Blue violet
            "bollinger_bands": "#87CEEB",    # Sky blue
            "rsi_line": "#F59E0B",           # Amber
            "rsi_overbought": "#DC2626",     # Red
            "rsi_oversold": "#059669",       # Green
            "macd_line": "#00D4AA",          # Cyan
            "macd_signal": "#FF6B6B",        # Light red
            "macd_histogram_positive": "#10B981", # Green
            "macd_histogram_negative": "#EF4444", # Red
            "volume_bullish": "#10B981",     # Green
            "volume_bearish": "#EF4444",     # Red
            "volume_sma": "#FFD700",         # Gold
            "pattern_annotation_bg": "#2A2E39",    # Dark annotation background
            "pattern_annotation_border": "#363A45", # Annotation border
            "text_primary": "#D1D4DC",       # Primary text color
            "text_secondary": "#9CA3AF"      # Secondary text color
        }
    }
    
    # Sentiment Analysis Configuration
    SENTIMENT_CONFIG: Dict[str, Any] = {
        "fud_keywords": [
            "crash", "dump", "scam", "rug", "dead", "worthless", "bubble", 
            "ponzi", "exit", "sell", "drop", "fall", "bearish", "fear",
            "uncertainty", "doubt", "manipulation", "whale", "whales"
        ],
        "shill_keywords": [
            "moon", "mooning", "rocket", "pump", "bull", "bullish", "diamond",
            "hands", "hodl", "buy", "accumulate", "breakout", "rally", "surge",
            "explosive", "massive", "huge", "gains", "profit", "lambo"
        ],
        "neutral_keywords": [
            "analysis", "chart", "technical", "support", "resistance", "volume",
            "indicator", "pattern", "trend", "price", "market", "trading"
        ],
        "sentiment_weights": {
            "news": 0.4,      # Professional news sources
            "reddit": 0.3,    # Reddit sentiment
            "twitter": 0.2,   # Twitter sentiment  
            "fourchan": 0.1   # 4chan sentiment (less weight due to noise)
        },
        "confidence_thresholds": {
            "high": 0.8,      # High confidence threshold
            "medium": 0.6,    # Medium confidence threshold
            "low": 0.4        # Low confidence threshold
        }
    }
    
    # Chart Generation Configuration
    CHART_CONFIG: Dict[str, Any] = {
        "figure_size": (16, 12),
        "dpi": 300,
        "style": "dark_background",
        "grid": True,
        "grid_alpha": 0.3,
        "title_fontsize": 16,
        "label_fontsize": 12,
        "tick_fontsize": 10,
        "legend_fontsize": 10,
        "line_width": 1.5,
        "marker_size": 4,
        "save_format": "png",
        "save_quality": 95,
        "tight_layout": True
    }
    
    # Pattern Recognition Configuration
    PATTERN_CONFIG: Dict[str, Any] = {
        "candlestick_patterns": [
            "doji", "hammer", "shooting_star", "engulfing", "morning_star", 
            "evening_star", "three_white_soldiers", "three_black_crows",
            "hanging_man", "inverted_hammer", "spinning_top", "marubozu"
        ],
        "trend_patterns": [
            "ascending_triangle", "descending_triangle", "symmetrical_triangle",
            "head_and_shoulders", "inverse_head_and_shoulders", "double_top",
            "double_bottom", "cup_and_handle", "wedge", "flag", "pennant"
        ],
        "pattern_confidence": {
            "strong": 0.8,
            "moderate": 0.6,
            "weak": 0.4
        }
    }
    
    @classmethod
    def get_ta_indicator_config(cls, indicator: str) -> Any:
        """
        Get configuration for a specific technical indicator.
        
        Args:
            indicator: Name of the indicator
            
        Returns:
            Configuration value for the indicator
        """
        return cls.TA_INDICATORS.get(indicator)
    
    @classmethod
    def get_sentiment_keywords(cls, category: str) -> List[str]:
        """
        Get sentiment keywords for a specific category.
        
        Args:
            category: Category of keywords (fud, shill, neutral)
            
        Returns:
            List of keywords for the category
        """
        key = f"{category}_keywords"
        return cls.SENTIMENT_CONFIG.get(key, [])
    
    @classmethod
    def get_sentiment_weight(cls, source: str) -> float:
        """
        Get sentiment weight for a specific source.
        
        Args:
            source: Sentiment source (news, reddit, twitter, fourchan)
            
        Returns:
            Weight value for the source
        """
        return cls.SENTIMENT_CONFIG["sentiment_weights"].get(source, 0.1)
    
    @classmethod
    def get_confidence_threshold(cls, level: str) -> float:
        """
        Get confidence threshold for a specific level.
        
        Args:
            level: Confidence level (high, medium, low)
            
        Returns:
            Threshold value for the level
        """
        return cls.SENTIMENT_CONFIG["confidence_thresholds"].get(level, 0.5)
    
    @classmethod
    def get_chart_config(cls, parameter: str) -> Any:
        """
        Get chart configuration parameter.
        
        Args:
            parameter: Chart configuration parameter name
            
        Returns:
            Configuration value for the parameter
        """
        return cls.CHART_CONFIG.get(parameter)
    
    @classmethod
    def get_pattern_list(cls, pattern_type: str) -> List[str]:
        """
        Get list of patterns for a specific type.
        
        Args:
            pattern_type: Type of patterns (candlestick, trend)
            
        Returns:
            List of pattern names
        """
        key = f"{pattern_type}_patterns"
        return cls.PATTERN_CONFIG.get(key, []) 