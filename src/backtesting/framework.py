"""
Core backtesting framework for thesis research.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import time
import logging
import requests

from ..tools.coingecko_tool import coingecko_tool
from ..tools.warosu_tool import warosu_archive_tool
from ..agents.crew_manager import CryptoForecastingCrew
from .methods import PredictionMethods
from .data_collector import DataCollector
from ..utils import APIRateLimiter, APICache
from ..tools.technical_analysis_tool import technical_analysis_tool
from ..tools.fourchan_tool import fourchan_biz_tool

logger = logging.getLogger(__name__)


class BacktestingFramework:
    """Main framework for running backtesting experiments for thesis research."""
    
    def __init__(self, crypto_symbol: str = "bitcoin", data_dir: str = "thesis_data"):
        self.crypto_symbol = crypto_symbol
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.prediction_methods = PredictionMethods()
        self.data_collector = DataCollector(self.data_dir)
        
        # Initialize rate limiter for CoinGecko API (30 req/min for Demo plan)
        self.rate_limiter = APIRateLimiter(max_requests_per_minute=30, safety_buffer=0.1)
        
        # Initialize cache for API responses
        cache_file = self.data_dir / "coingecko_cache.json"
        self.api_cache = APICache(cache_file=str(cache_file), max_age_hours=24)
        
        # Results storage
        self.results_file = self.data_dir / f"{crypto_symbol}_backtest_results.json"
        self.results = self._load_existing_results()
        
        # Initialize the real forecasting crew for agentic testing
        self.forecasting_crew = CryptoForecastingCrew(verbose=False)
        
        logger.info(f"Initialized backtesting framework for {crypto_symbol}")
        logger.info(f"Rate limiter: 30 req/min with {self.rate_limiter.min_delay:.1f}s min delay")
        logger.info(f"Real agentic crew initialized for proper backtesting")
    
    def _load_existing_results(self) -> Dict[str, Any]:
        """Load existing results to resume backtesting."""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")
        
        return {
            "metadata": {
                "crypto_symbol": self.crypto_symbol,
                "start_date": None,
                "end_date": None,
                "total_days": 0,
                "completed_days": 0
            },
            "daily_predictions": {},
            "summary_stats": {}
        }
    
    def _save_results(self):
        """Save current results to file."""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _get_historical_price_data(self, date: str) -> Optional[Dict[str, Any]]:
        """Get historical price data for a specific date using CoinGecko's historical API with rate limiting."""
        try:
            from ..config import Config
            
            # Parse the date
            current_date = datetime.strptime(date, '%Y-%m-%d')
            next_date = current_date + timedelta(days=1)
            
            # For backtesting, we allow any dates - real API calls will handle availability
            if current_date.year < 2010:
                logger.warning(f"Date {date} is very old, data may be limited")
            
            if current_date > datetime.now():
                logger.info(f"Future date {date} - using simulated/extrapolated data for backtesting")
            
            # Check cache first
            cache_key = f"{self.crypto_symbol}_{date}"
            cached_data = self.api_cache.get(cache_key)
            if cached_data:
                logger.info(f"Using cached data for {date}")
                return cached_data
            
            # CoinGecko historical data endpoint
            date_formatted = current_date.strftime('%d-%m-%Y')
            next_date_formatted = next_date.strftime('%d-%m-%Y')
            
            session = requests.Session()
            api_key = Config.COINGECKO_API_KEY or ""
            if api_key:
                session.headers.update({"x-cg-demo-api-key": api_key})
            
            # Get data for current date with rate limiting and retry logic
            current_price_data = self._fetch_historical_data_with_retry(
                session, date_formatted, date, max_retries=3
            )
            
            if not current_price_data:
                return None
            
            current_price = current_price_data["price"]
            market_data = current_price_data["market_data"]
            
            # Get next day's data for price movement calculation
            next_day_price = None
            next_day_data = self._fetch_historical_data_with_retry(
                session, next_date_formatted, next_date.strftime('%Y-%m-%d'), max_retries=2
            )
            
            if next_day_data:
                next_day_price = next_day_data["price"]
            
            # Calculate price movement
            actual_movement = None
            if next_day_price:
                actual_movement = ((next_day_price - current_price) / current_price) * 100
            
            # Create OHLCV-like data
            ohlcv_data = {
                "timestamp": current_date.isoformat(),
                "open": current_price,
                "high": current_price,
                "low": current_price,
                "close": current_price,
                "volume": market_data.get("total_volume", {}).get("usd", 0)
            }
            
            result = {
                "date": date,
                "price": current_price,
                "ohlcv": ohlcv_data,
                "next_day_price": next_day_price,
                "actual_movement_24h": actual_movement,
                "data_available": True,
                "market_cap": market_data.get("market_cap", {}).get("usd", 0),
                "volume_24h": market_data.get("total_volume", {}).get("usd", 0)
            }
            
            # Cache the result
            self.api_cache.set(cache_key, result)
            
            logger.info(f"Retrieved historical data for {date}: ${current_price:.2f}")
            return result
        
        except Exception as e:
            logger.error(f"Error getting historical price data for {date}: {e}")
            return None
    
    def _fetch_historical_data_with_retry(self, session: requests.Session, date_formatted: str, 
                                        date_str: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Fetch historical data with rate limiting and retry logic."""
        from ..config import Config
        
        for attempt in range(max_retries):
            try:
                # Wait for rate limit before making request
                self.rate_limiter.wait_if_needed()
                
                url = f"{Config.COINGECKO_BASE_URL}/coins/{self.crypto_symbol}/history"
                params = {
                    "date": date_formatted,
                    "localization": "false"
                }
                
                logger.debug(f"Fetching data for {date_str} (attempt {attempt + 1}/{max_retries})")
                response = session.get(url, params=params, timeout=15)
                
                # Handle different response codes
                if response.status_code == 401:
                    logger.error(f"CoinGecko API authentication error for {date_str}")
                    return None
                elif response.status_code == 429:
                    # Rate limit exceeded - wait longer and retry
                    wait_time = (2 ** attempt) * 30  # Exponential backoff: 30s, 60s, 120s
                    logger.warning(f"Rate limit exceeded for {date_str}, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 404:
                    logger.error(f"Data not found for {date_str}")
                    return None
                elif "time range" in response.text.lower():
                    logger.error(f"CoinGecko API time range limitation for {date_str}")
                    return None
                
                response.raise_for_status()
                data = response.json()
                
                if "market_data" not in data:
                    logger.error(f"No market data available from CoinGecko for {date_str}")
                    return None
                
                market_data = data["market_data"]
                current_price = market_data.get("current_price", {}).get("usd")
                
                if not current_price:
                    logger.error(f"No USD price available from CoinGecko for {date_str}")
                    return None
                
                return {
                    "price": current_price,
                    "market_data": market_data
                }
                
            except requests.RequestException as e:
                logger.warning(f"Request failed for {date_str} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"All retry attempts failed for {date_str}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error for {date_str}: {e}")
                return None
        
        return None
    

    
    def _get_historical_sentiment_data(self, date: str) -> Optional[Dict[str, Any]]:
        """Get historical sentiment data from Warosu archive."""
        try:
            # Get sentiment data from the date and previous day
            current_date = datetime.strptime(date, '%Y-%m-%d')
            prev_date = current_date - timedelta(days=1)
            
            date_from = prev_date.strftime('%Y-%m-%d')
            date_to = date
            
            # Search for relevant keywords
            keywords = ['btc', 'bitcoin'] if self.crypto_symbol == 'bitcoin' else [self.crypto_symbol]
            
            result = warosu_archive_tool.func(keywords, date_from, date_to, max_posts=30)
            
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result
            
            if "error" in data:
                return {"date": date, "posts": [], "sentiment_available": False, "error": data["error"]}
            
            return {
                "date": date,
                "posts": data.get("posts", []),
                "total_posts": data.get("total_posts", 0),
                "sentiment_available": True,
                "keywords": keywords
            }
        
        except Exception as e:
            logger.error(f"Error getting historical sentiment data for {date}: {e}")
            return {"date": date, "posts": [], "sentiment_available": False, "error": str(e)}
    
    def _setup_historical_context(self, historical_date: str):
        """Setup historical context for all tools by monkey-patching their functions."""
        
        # Store original functions
        self._original_coingecko_func = coingecko_tool.func
        self._original_warosu_func = warosu_archive_tool.func
        self._original_technical_func = technical_analysis_tool.func
        self._original_fourchan_func = fourchan_biz_tool.func
        
        # Store original sentiment agent tools before replacing them
        self._original_sentiment_tools = None
        if hasattr(self.forecasting_crew, 'sentiment_agent') and hasattr(self.forecasting_crew.sentiment_agent, 'tools'):
            self._original_sentiment_tools = self.forecasting_crew.sentiment_agent.tools.copy()
        
        # Store the backtest date to avoid variable shadowing
        backtest_date = historical_date
        
        # Create historical mode wrappers that automatically inject the historical date
        def coingecko_historical_wrapper(query: str, historical_date: Optional[str] = None):
            # Always use the backtest historical date, ignore any passed parameter
            return self._original_coingecko_func(query, backtest_date)
        
        def warosu_historical_wrapper(keywords: List[str], date_from: Optional[str] = None, date_to: Optional[str] = None, max_posts: int = 50, historical_date: Optional[str] = None):
            # Always use the backtest historical date, ignore any passed parameter
            return self._original_warosu_func(keywords, date_from, date_to, max_posts, backtest_date)
        
        def technical_historical_wrapper(crypto_name: str, forecast_horizon: str = "24 hours", historical_date: str = ""):
            # Always use the backtest historical date, ignore any passed parameter
            return self._original_technical_func(crypto_name, forecast_horizon, backtest_date)
        
        def fourchan_historical_wrapper(keywords: List[str], max_threads: int = 5, max_posts_per_thread: int = 20, historical_date: Optional[str] = None):
            # Always use the backtest historical date, ignore any passed parameter
            return self._original_fourchan_func(keywords, max_threads, max_posts_per_thread, backtest_date)
        
        # Replace the functions with historical wrappers
        coingecko_tool.func = coingecko_historical_wrapper
        warosu_archive_tool.func = warosu_historical_wrapper
        technical_analysis_tool.func = technical_historical_wrapper
        fourchan_biz_tool.func = fourchan_historical_wrapper
        
        # IMPORTANT: Replace sentiment agent's fourchan tool with warosu tool for backtesting
        # Since fourchan has no historical data but warosu does
        self._replace_sentiment_tool_for_backtesting()
        
        logger.info(f"Historical context set for {backtest_date} - all tools will use historical data")
    
    def _replace_sentiment_tool_for_backtesting(self):
        """Replace sentiment agent's fourchan tool with warosu tool for historical data."""
        from ..tools import create_warosu_tool
        
        try:
            # Get the sentiment agent from the forecasting crew
            sentiment_agent = self.forecasting_crew.sentiment_agent
            
            # Create warosu tool instance
            warosu_tool = create_warosu_tool()
            
            # Replace the tools list - sentiment agent should only have one tool
            if hasattr(sentiment_agent, 'tools') and sentiment_agent.tools:
                original_tool_count = len(sentiment_agent.tools)
                sentiment_agent.tools = [warosu_tool]
                logger.info(f"Replaced {original_tool_count} sentiment tool(s) with warosu tool for backtesting")
            else:
                # If no tools, add warosu tool
                sentiment_agent.tools = [warosu_tool]
                logger.info("Added warosu tool to sentiment agent for backtesting")
                
        except Exception as e:
            logger.warning(f"Failed to replace sentiment tool with warosu: {e}")
            logger.warning("Sentiment analysis may use fourchan tool which has no historical data")
    
    def _restore_tool_context(self):
        """Restore original tool functions after historical testing."""
        
        # Restore original functions
        coingecko_tool.func = self._original_coingecko_func
        warosu_archive_tool.func = self._original_warosu_func
        technical_analysis_tool.func = self._original_technical_func
        fourchan_biz_tool.func = self._original_fourchan_func
        
        # Restore original sentiment agent tools
        if self._original_sentiment_tools is not None:
            try:
                sentiment_agent = self.forecasting_crew.sentiment_agent
                if hasattr(sentiment_agent, 'tools'):
                    sentiment_agent.tools = self._original_sentiment_tools
                    logger.info(f"Restored {len(self._original_sentiment_tools)} original sentiment tool(s)")
            except Exception as e:
                logger.warning(f"Failed to restore original sentiment tools: {e}")
        
        logger.debug("Tool context restored to normal mode")
    
    def run_daily_prediction(self, date: str, method: str = "full_agentic") -> Dict[str, Any]:
        """Run prediction for a specific date using specified method with REAL agentic approach."""
        
        logger.info(f"Running {method} prediction for {date} using REAL agentic system")
        
        try:
            if method == "full_agentic":
                # Use the REAL CryptoForecastingCrew with historical context
                return self._run_real_agentic_prediction(date)
            elif method == "image_only":
                # Use simplified prediction for comparison
                return self._run_image_only_prediction(date)
            elif method == "sentiment_only":
                # Use simplified prediction for comparison
                return self._run_sentiment_only_prediction(date)
            else:
                raise ValueError(f"Unknown prediction method: {method}")
                
        except Exception as e:
            logger.error(f"Error running {method} prediction for {date}: {e}")
            return {
                "date": date,
                "method": method,
                "success": False,
                "error": str(e)
            }
    
    def _run_real_agentic_prediction(self, date: str) -> Dict[str, Any]:
        """Run the REAL agentic prediction using CryptoForecastingCrew with historical context."""
        
        try:
            # Setup historical context for all tools
            self._setup_historical_context(date)
            
            # Run the REAL forecasting crew (this is the actual agentic system!)
            logger.info(f"Running REAL CryptoForecastingCrew for {date}")
            crew_results = self.forecasting_crew.run_forecast(
                crypto_name=self.crypto_symbol,
                forecast_horizon="24 hours"
            )
            
            # Restore normal tool context
            self._restore_tool_context()
            
            # Get actual outcome data for validation
            price_data = self._get_historical_price_data(date)
            
            # Extract prediction data from crew results
            prediction = {
                "date": date,
                "method": "full_agentic",
                "success": True,
                "predicted_direction": crew_results.get("direction", "UNKNOWN"),
                "confidence": crew_results.get("confidence", "UNKNOWN"),
                "reasoning": crew_results.get("explanation", ""),
                "crew_forecast": crew_results.get("forecast", ""),
                "execution_time": None,
                "agents_used": crew_results.get("execution_summary", {}).get("agents_used", []),
                "tools_used": crew_results.get("execution_summary", {}).get("tools_used", []),
                "charts_generated": crew_results.get("charts_generated", False),
                "price_validation": crew_results.get("price_validation", {}),
                
                # Extract trading metrics if available
                "target_price": crew_results.get("targets", {}).get("primary", "Not specified"),
                "take_profits": crew_results.get("take_profits", {}),
                "stop_loss": crew_results.get("stop_loss", "Not specified"),
                "risk_reward_ratio": crew_results.get("risk_reward_ratio", "Not specified"),
                "position_size": crew_results.get("position_size", "Not specified"),
                "time_horizon": crew_results.get("time_horizon", "24 hours"),
                "key_catalysts": crew_results.get("key_catalysts", []),
                "risk_factors": crew_results.get("risk_factors", []),
                
                # Add actual outcome data for analysis
                "actual_price": price_data.get("price") if price_data else None,
                "actual_movement_24h": price_data.get("actual_movement_24h") if price_data else None,
                "next_day_price": price_data.get("next_day_price") if price_data else None,
                "data_quality": {
                    "price_data_available": bool(price_data and price_data.get("data_available")),
                    "crew_execution_successful": not crew_results.get("error"),
                    "historical_mode": True
                },
                
                # Mark as real agentic system
                "agentic_system": "REAL",
                "methodology": "Multi-agent crew with tool autonomy"
            }
            
            # Handle crew errors
            if crew_results.get("error"):
                prediction.update({
                    "success": False,
                    "error": crew_results["error"],
                    "crew_error_details": crew_results.get("error_details", {})
                })
            
            logger.info(f"Real agentic prediction completed for {date}: {prediction['predicted_direction']} with {prediction['confidence']} confidence")
            return prediction
            
        except Exception as e:
            # Make sure to restore context even if there's an error
            try:
                self._restore_tool_context()
            except:
                pass
            
            logger.error(f"Real agentic prediction failed for {date}: {e}")
            return {
                "date": date,
                "method": "full_agentic",
                "success": False,
                "error": f"Real agentic system error: {str(e)}",
                "agentic_system": "REAL",
                "methodology": "Multi-agent crew with tool autonomy"
            }
    
    def _run_image_only_prediction(self, date: str) -> Dict[str, Any]:
        """Run image-only prediction for comparison (simplified approach)."""
        
        try:
            # Use the old approach for comparison
            price_data = self._get_historical_price_data(date)
            
            if not price_data or not price_data.get("data_available"):
                return {
                    "date": date,
                    "method": "image_only",
                    "success": False,
                    "error": "Price data not available"
                }
            
            prediction = self.prediction_methods.image_only_prediction(
                self.crypto_symbol, date, price_data
            )
            
            # Add actual outcome data
            prediction.update({
                "actual_price": price_data["price"],
                "actual_movement_24h": price_data.get("actual_movement_24h"),
                "next_day_price": price_data.get("next_day_price"),
                "data_quality": {
                    "price_data_available": True,
                    "historical_mode": True
                },
                "agentic_system": "SIMPLIFIED",
                "methodology": "Pre-collected data analysis"
            })
            
            return prediction
            
        except Exception as e:
            logger.error(f"Image-only prediction failed for {date}: {e}")
            return {
                "date": date,
                "method": "image_only",
                "success": False,
                "error": str(e),
                "agentic_system": "SIMPLIFIED"
            }
    
    def _run_sentiment_only_prediction(self, date: str) -> Dict[str, Any]:
        """Run sentiment-only prediction for comparison (simplified approach)."""
        
        try:
            # Use the old approach for comparison
            sentiment_data = self._get_historical_sentiment_data(date)
            
            prediction = self.prediction_methods.sentiment_only_prediction(
                self.crypto_symbol, date, sentiment_data
            )
            
            # Add actual outcome data
            price_data = self._get_historical_price_data(date)
            if price_data:
                prediction.update({
                    "actual_price": price_data["price"],
                    "actual_movement_24h": price_data.get("actual_movement_24h"),
                    "next_day_price": price_data.get("next_day_price"),
                })
            
            prediction.update({
                "data_quality": {
                    "sentiment_data_available": sentiment_data.get("sentiment_available", False),
                    "sentiment_posts_count": sentiment_data.get("total_posts", 0),
                    "historical_mode": True
                },
                "agentic_system": "SIMPLIFIED",
                "methodology": "Pre-collected data analysis"
            })
            
            return prediction
            
        except Exception as e:
            logger.error(f"Sentiment-only prediction failed for {date}: {e}")
            return {
                "date": date,
                "method": "sentiment_only",
                "success": False,
                "error": str(e),
                "agentic_system": "SIMPLIFIED"
            }
    
    def run_backtest(self, 
                     start_date: str, 
                     end_date: str, 
                     methods: List[str] = None,
                     skip_existing: bool = True) -> Dict[str, Any]:
        """
        Run complete backtest for date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            methods: List of prediction methods to test
            skip_existing: Skip dates that already have results
        """
        
        if methods is None:
            methods = ["full_agentic", "image_only", "sentiment_only"]
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        total_days = (end_dt - start_dt).days + 1
        
        # Update metadata
        self.results["metadata"].update({
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "methods": methods,
            "backtest_started": datetime.now().isoformat()
        })
        
        logger.info(f"Starting backtest from {start_date} to {end_date} ({total_days} days)")
        logger.info(f"Methods to test: {methods}")
        
        current_date = start_dt
        completed_count = 0
        
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Check if we should skip this date
            if skip_existing and date_str in self.results["daily_predictions"]:
                existing_methods = set(self.results["daily_predictions"][date_str].keys())
                remaining_methods = [m for m in methods if m not in existing_methods]
                if not remaining_methods:
                    logger.info(f"Skipping {date_str} - all methods already completed")
                    current_date += timedelta(days=1)
                    completed_count += 1
                    continue
            else:
                remaining_methods = methods
            
            # Initialize date entry if not exists
            if date_str not in self.results["daily_predictions"]:
                self.results["daily_predictions"][date_str] = {}
            
            # Run predictions for each method
            for method in remaining_methods:
                try:
                    # Show rate limiter status
                    rate_status = self.rate_limiter.get_status()
                    logger.info(f"[{completed_count+1}/{total_days}] Running {method} for {date_str} "
                               f"(API: {rate_status['recent_requests']}/{rate_status['max_requests_per_minute']} req/min)")
                    
                    prediction = self.run_daily_prediction(date_str, method)
                    self.results["daily_predictions"][date_str][method] = prediction
                    
                    # Small delay between methods to be extra safe
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Failed {method} prediction for {date_str}: {e}")
                    self.results["daily_predictions"][date_str][method] = {
                        "date": date_str,
                        "method": method,
                        "success": False,
                        "error": str(e)
                    }
            
            completed_count += 1
            self.results["metadata"]["completed_days"] = completed_count
            
            # Save progress every 5 days (more frequent saves to avoid data loss)
            if completed_count % 5 == 0:
                self._save_results()
                rate_status = self.rate_limiter.get_status()
                logger.info(f"Progress saved: {completed_count}/{total_days} days completed. "
                           f"API usage: {rate_status['recent_requests']}/{rate_status['max_requests_per_minute']} req/min")
            
            current_date += timedelta(days=1)
        
        # Final save
        self.results["metadata"]["backtest_completed"] = datetime.now().isoformat()
        self._save_results()
        
        logger.info(f"Backtest completed: {completed_count} days processed")
        return self.results
    
    def run_year_backtest(self, methods: List[str] = None) -> Dict[str, Any]:
        """Run backtest for the past year."""
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=365)  # One year ago
        
        return self.run_backtest(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            methods
        )
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary statistics of backtest results."""
        if not self.results["daily_predictions"]:
            return {"error": "No prediction data available"}
        
        summary = {}
        
        for method in ["full_agentic", "image_only", "sentiment_only"]:
            method_results = []
            
            for date, predictions in self.results["daily_predictions"].items():
                if method in predictions and predictions[method].get("success"):
                    method_results.append(predictions[method])
            
            if method_results:
                summary[method] = self._calculate_method_stats(method_results)
        
        return summary
    
    def _calculate_method_stats(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a prediction method including trading performance."""
        total_predictions = len(predictions)
        
        if total_predictions == 0:
            return {"total_predictions": 0}
        
        # Basic accuracy metrics
        correct_predictions = 0
        direction_predictions = []
        movement_predictions = []
        
        # Enhanced trading metrics
        trading_results = []
        portfolio_value = 10000  # Starting with $10,000 portfolio
        portfolio_history = [portfolio_value]
        
        for pred in predictions:
            if pred.get("actual_movement_24h") is not None:
                actual_movement = pred["actual_movement_24h"]
                current_price = pred.get("historical_price", 0)
                next_day_price = pred.get("next_day_price")
                
                # Determine actual direction based on 24h movement
                if actual_movement > 1.0:
                    actual_direction = "UP"
                elif actual_movement < -1.0:
                    actual_direction = "DOWN"
                else:
                    # For small movements, assign to DOWN (conservative approach)
                    actual_direction = "DOWN"
                
                predicted_direction = pred.get("predicted_direction", "DOWN")
                
                direction_predictions.append({
                    "predicted": predicted_direction,
                    "actual": actual_direction,
                    "correct": predicted_direction == actual_direction
                })
                
                if predicted_direction == actual_direction:
                    correct_predictions += 1
                
                # Calculate trading performance if we have enhanced data
                if all(key in pred for key in ["target_price", "take_profit", "stop_loss"]):
                    trade_result = self._calculate_trade_performance(
                        pred, actual_movement, current_price, next_day_price
                    )
                    trading_results.append(trade_result)
                    
                    # Update portfolio value
                    position_size_multiplier = self._get_position_size_multiplier(pred.get("position_size", "MEDIUM"))
                    trade_pnl = trade_result.get("pnl_percentage", 0)
                    portfolio_change = portfolio_value * (trade_pnl / 100) * position_size_multiplier
                    portfolio_value += portfolio_change
                    portfolio_history.append(portfolio_value)
        
        accuracy = correct_predictions / len(direction_predictions) if direction_predictions else 0
        
        # Calculate trading-specific metrics
        trading_metrics = self._calculate_trading_metrics(trading_results, portfolio_history)
        
        return {
            "total_predictions": total_predictions,
            "valid_predictions": len(direction_predictions),
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "detailed_predictions": direction_predictions[:10],  # Sample of predictions
            
            # Enhanced trading metrics
            "trading_performance": trading_metrics,
            "portfolio_final_value": portfolio_value,
            "portfolio_return": ((portfolio_value - 10000) / 10000) * 100,
            "total_trades": len(trading_results),
            "sample_trading_results": trading_results[:5]  # Sample trades
        }
    
    def _calculate_trade_performance(self, 
                                   prediction: Dict[str, Any], 
                                   actual_movement: float,
                                   current_price: float, 
                                   next_day_price: Optional[float]) -> Dict[str, Any]:
        """Calculate individual trade performance including stop loss and take profit analysis."""
        
        if not next_day_price or current_price <= 0:
            return {"error": "Insufficient price data"}
        
        predicted_direction = prediction.get("predicted_direction", "DOWN")
        target_price = prediction.get("target_price", current_price)
        take_profit = prediction.get("take_profit", current_price)
        stop_loss = prediction.get("stop_loss", current_price)
        confidence = prediction.get("confidence", "MEDIUM")
        
        # Calculate what would have happened in a real trade
        trade_result = {
            "entry_price": current_price,
            "exit_price": next_day_price,
            "predicted_direction": predicted_direction,
            "actual_movement_pct": actual_movement,
            "target_price": target_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "confidence": confidence
        }
        
        # Determine trade outcome
        if predicted_direction == "UP":
            # Long position
            pnl_percentage = ((next_day_price - current_price) / current_price) * 100
            
            # Check if stop loss or take profit was hit
            if next_day_price <= stop_loss:
                trade_result["outcome"] = "STOP_LOSS_HIT"
                trade_result["exit_price"] = stop_loss
                pnl_percentage = ((stop_loss - current_price) / current_price) * 100
            elif next_day_price >= take_profit:
                trade_result["outcome"] = "TAKE_PROFIT_HIT"
                trade_result["exit_price"] = take_profit
                pnl_percentage = ((take_profit - current_price) / current_price) * 100
            elif next_day_price > current_price:
                trade_result["outcome"] = "PROFITABLE"
            else:
                trade_result["outcome"] = "LOSS"
                
        else:  # DOWN
            # Short position (or inverse)
            pnl_percentage = ((current_price - next_day_price) / current_price) * 100
            
            # For short positions, stop loss and take profit logic is inverted
            if next_day_price >= stop_loss:  # Stop loss for short
                trade_result["outcome"] = "STOP_LOSS_HIT"
                trade_result["exit_price"] = stop_loss
                pnl_percentage = ((current_price - stop_loss) / current_price) * 100
            elif next_day_price <= take_profit:  # Take profit for short
                trade_result["outcome"] = "TAKE_PROFIT_HIT"
                trade_result["exit_price"] = take_profit
                pnl_percentage = ((current_price - take_profit) / current_price) * 100
            elif next_day_price < current_price:
                trade_result["outcome"] = "PROFITABLE"
            else:
                trade_result["outcome"] = "LOSS"
        
        trade_result["pnl_percentage"] = pnl_percentage
        trade_result["target_hit"] = abs(next_day_price - target_price) / current_price < 0.02  # Within 2%
        
        return trade_result
    
    def _calculate_trading_metrics(self, trading_results: List[Dict[str, Any]], 
                                 portfolio_history: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive trading performance metrics."""
        
        if not trading_results:
            return {"no_trading_data": True}
        
        # P&L metrics
        pnl_values = [trade.get("pnl_percentage", 0) for trade in trading_results]
        winning_trades = [pnl for pnl in pnl_values if pnl > 0]
        losing_trades = [pnl for pnl in pnl_values if pnl < 0]
        
        # Stop loss and take profit analysis
        stop_loss_hits = len([t for t in trading_results if t.get("outcome") == "STOP_LOSS_HIT"])
        take_profit_hits = len([t for t in trading_results if t.get("outcome") == "TAKE_PROFIT_HIT"])
        target_hits = len([t for t in trading_results if t.get("target_hit", False)])
        
        # Calculate Sharpe ratio (simplified)
        if len(pnl_values) > 1:
            avg_return = sum(pnl_values) / len(pnl_values)
            std_return = (sum((x - avg_return) ** 2 for x in pnl_values) / len(pnl_values)) ** 0.5
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown calculation
        max_drawdown = 0
        peak = portfolio_history[0]
        for value in portfolio_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            # Basic P&L metrics
            "total_trades": len(trading_results),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(trading_results) if trading_results else 0,
            
            # Returns
            "total_return_pct": sum(pnl_values),
            "avg_return_per_trade": sum(pnl_values) / len(pnl_values) if pnl_values else 0,
            "avg_winning_trade": sum(winning_trades) / len(winning_trades) if winning_trades else 0,
            "avg_losing_trade": sum(losing_trades) / len(losing_trades) if losing_trades else 0,
            
            # Risk metrics
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe_ratio": sharpe_ratio,
            "volatility": std_return if len(pnl_values) > 1 else 0,
            
            # Stop loss / Take profit analysis
            "stop_loss_hit_rate": stop_loss_hits / len(trading_results) if trading_results else 0,
            "take_profit_hit_rate": take_profit_hits / len(trading_results) if trading_results else 0,
            "target_hit_rate": target_hits / len(trading_results) if trading_results else 0,
            
            # Risk/Reward
            "profit_factor": (
                abs(sum(winning_trades)) / abs(sum(losing_trades)) 
                if losing_trades and sum(losing_trades) != 0 else float('inf')
            ),
            
            # Best and worst trades
            "best_trade_pct": max(pnl_values) if pnl_values else 0,
            "worst_trade_pct": min(pnl_values) if pnl_values else 0,
            
            # Confidence correlation
            "high_confidence_trades": len([t for t in trading_results if t.get("confidence") == "HIGH"]),
            "high_confidence_win_rate": (
                len([t for t in trading_results 
                     if t.get("confidence") == "HIGH" and t.get("pnl_percentage", 0) > 0]) /
                max(1, len([t for t in trading_results if t.get("confidence") == "HIGH"]))
            )
        }
    
    def _get_position_size_multiplier(self, position_size: str) -> float:
        """Convert position size to multiplier for portfolio impact."""
        multipliers = {
            "SMALL": 0.25,   # 25% of available capital
            "MEDIUM": 0.5,   # 50% of available capital  
            "LARGE": 1.0     # 100% of available capital
        }
        return multipliers.get(position_size, 0.5) 