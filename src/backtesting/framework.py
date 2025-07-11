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
        
        logger.info(f"Initialized backtesting framework for {crypto_symbol}")
        logger.info(f"Rate limiter: 30 req/min with {self.rate_limiter.min_delay:.1f}s min delay")
    
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
            
            # Verify we're within reasonable historical data range
            if current_date.year < 2020:
                logger.error(f"Date {date} is too far in the past for reliable CoinGecko data (before 2020)")
                return None
            
            if current_date > datetime(2024, 12, 31):
                logger.error(f"Date {date} is beyond reliable historical data range")
                return None
            
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
    
    def run_daily_prediction(self, date: str, method: str = "full_agentic") -> Dict[str, Any]:
        """Run prediction for a specific date using specified method."""
        
        logger.info(f"Running {method} prediction for {date}")
        
        # Get historical data for the prediction date
        price_data = self._get_historical_price_data(date)
        sentiment_data = self._get_historical_sentiment_data(date)
        
        if not price_data or not price_data.get("data_available"):
            return {
                "date": date,
                "method": method,
                "success": False,
                "error": "Price data not available"
            }
        
        try:
            # Run prediction based on method
            if method == "full_agentic":
                prediction = self.prediction_methods.full_agentic_prediction(
                    self.crypto_symbol, date, price_data, sentiment_data
                )
            elif method == "image_only":
                prediction = self.prediction_methods.image_only_prediction(
                    self.crypto_symbol, date, price_data
                )
            elif method == "sentiment_only":
                prediction = self.prediction_methods.sentiment_only_prediction(
                    self.crypto_symbol, date, sentiment_data
                )
            else:
                raise ValueError(f"Unknown prediction method: {method}")
            
            # Add actual outcome data
            prediction.update({
                "actual_price": price_data["price"],
                "actual_movement_24h": price_data.get("actual_movement_24h"),
                "next_day_price": price_data.get("next_day_price"),
                "data_quality": {
                    "price_data_available": True,
                    "sentiment_data_available": sentiment_data.get("sentiment_available", False),
                    "sentiment_posts_count": sentiment_data.get("total_posts", 0)
                }
            })
            
            return prediction
        
        except Exception as e:
            logger.error(f"Error running {method} prediction for {date}: {e}")
            return {
                "date": date,
                "method": method,
                "success": False,
                "error": str(e)
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
        """Calculate statistics for a prediction method."""
        total_predictions = len(predictions)
        
        if total_predictions == 0:
            return {"total_predictions": 0}
        
        # Calculate accuracy metrics
        correct_predictions = 0
        direction_predictions = []
        movement_predictions = []
        
        for pred in predictions:
            if pred.get("actual_movement_24h") is not None:
                actual_movement = pred["actual_movement_24h"]
                # Determine actual direction based on 24h movement
                if actual_movement > 1.0:
                    actual_direction = "UP"
                elif actual_movement < -1.0:
                    actual_direction = "DOWN"
                else:
                    # For small movements, assign to DOWN (conservative approach)
                    actual_direction = "DOWN"
                
                predicted_direction = pred.get("predicted_direction", "DOWN")  # Default to DOWN instead of NEUTRAL
                
                direction_predictions.append({
                    "predicted": predicted_direction,
                    "actual": actual_direction,
                    "correct": predicted_direction == actual_direction
                })
                
                if predicted_direction == actual_direction:
                    correct_predictions += 1
        
        accuracy = correct_predictions / len(direction_predictions) if direction_predictions else 0
        
        return {
            "total_predictions": total_predictions,
            "valid_predictions": len(direction_predictions),
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "detailed_predictions": direction_predictions[:10]  # Sample of predictions
        } 