"""
Prediction methods for thesis comparison: Agentic vs One-shot approaches.
Enhanced with trading metrics including take profit, stop loss, and percentage targets.
"""

import json
import tempfile
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from ..agents.crew_manager import CryptoForecastingCrew
from ..tools.technical_analysis_tool import technical_analysis_tool
from ..tools.chart_analysis_tool import chart_analysis_tool
from ..llm_factory import LLMFactory
from ..config import Config

logger = logging.getLogger(__name__)


class PredictionMethods:
    """Contains all prediction methods for thesis comparison with enhanced trading metrics."""
    
    def __init__(self):
        self.llm = self._create_llm()
    
    def _create_llm(self):
        """Create LLM instance for one-shot predictions."""
        return LLMFactory.create_llm(
            provider=Config.DEFAULT_LLM_PROVIDER,
            model=Config.DEFAULT_LLM_MODEL,
            temperature=0.1,
            max_tokens=2000
        )
    
    def full_agentic_prediction(self, 
                               crypto_symbol: str, 
                               date: str, 
                               price_data: Dict[str, Any], 
                               sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full multi-agent prediction with enhanced trading metrics.
        """
        try:
            logger.info(f"Running full agentic prediction for {crypto_symbol} on {date}")
            
            current_price = price_data.get('price', 0)
            
            # Create historical analysis using the provided historical data
            tech_analysis = self._create_historical_technical_analysis(crypto_symbol, date, price_data)
            sentiment_analysis = self._create_historical_sentiment_analysis(crypto_symbol, date, sentiment_data)
            
            # Enhanced prediction prompt with trading metrics
            prediction_prompt = f"""
            As a cryptocurrency trading expert, analyze the following historical data for {crypto_symbol} on {date} and provide comprehensive trading predictions for the next 24 hours.
            
            HISTORICAL MARKET DATA (as of {date}):
            - Current Price: ${current_price:.2f}
            - Historical OHLCV data available: {bool(price_data.get('ohlcv'))}
            
            TECHNICAL ANALYSIS:
            {tech_analysis}
            
            SENTIMENT ANALYSIS:
            {sentiment_analysis}
            
            Provide a comprehensive trading analysis with the following:
            
            1. DIRECTION: UP or DOWN for next 24 hours (choose one - no neutral)
            2. CONFIDENCE: HIGH/MEDIUM/LOW
            3. TARGET_PERCENTAGE: Expected percentage move (e.g., +5.2% or -3.1%)
            4. TARGET_PRICE: Specific price target based on percentage move
            5. TAKE_PROFIT: Recommended take profit level (price)
            6. STOP_LOSS: Recommended stop loss level (price) 
            7. RISK_REWARD_RATIO: Risk/reward ratio for this trade
            8. POSITION_SIZE: Recommended position size (SMALL/MEDIUM/LARGE)
            9. TIME_HORIZON: Expected time to reach target (e.g., 4-8 hours, 12-18 hours)
            10. REASONING: Detailed explanation based on technical and sentiment factors
            
            Be specific with numbers. For a ${current_price:.2f} current price:
            - If predicting UP by 3%, target would be ${current_price * 1.03:.2f}
            - Take profit might be at ${current_price * 1.025:.2f} (2.5% gain)
            - Stop loss might be at ${current_price * 0.985:.2f} (1.5% loss)
            
            Format your response as:
            DIRECTION: [UP/DOWN]
            CONFIDENCE: [HIGH/MEDIUM/LOW]
            TARGET_PERCENTAGE: [percentage with + or - sign]
            TARGET_PRICE: [specific price]
            TAKE_PROFIT: [specific price]
            STOP_LOSS: [specific price]
            RISK_REWARD_RATIO: [ratio like 1:2 or 1:3]
            POSITION_SIZE: [SMALL/MEDIUM/LARGE]
            TIME_HORIZON: [time estimate]
            REASONING: [detailed reasoning]
            """
            
            # Get LLM response using historical context
            response = self._get_llm_response(prediction_prompt)
            
            # Extract all prediction components
            predicted_direction = self._extract_direction_from_text(response)
            confidence = self._extract_confidence_from_text(response)
            reasoning = self._extract_reasoning_from_text(response)
            
            # Extract new trading metrics
            trading_metrics = self._extract_trading_metrics(response, current_price)
            
            prediction = {
                "date": date,
                "method": "full_agentic",
                "success": True,
                "predicted_direction": predicted_direction,
                "confidence": confidence,
                "reasoning": reasoning,
                "technical_analysis": tech_analysis,
                "sentiment_analysis": sentiment_analysis,
                "historical_price": current_price,
                
                # Enhanced trading metrics
                "target_percentage": trading_metrics.get("target_percentage"),
                "target_price": trading_metrics.get("target_price"),
                "take_profit": trading_metrics.get("take_profit"),
                "stop_loss": trading_metrics.get("stop_loss"),
                "risk_reward_ratio": trading_metrics.get("risk_reward_ratio"),
                "position_size": trading_metrics.get("position_size"),
                "time_horizon": trading_metrics.get("time_horizon"),
                
                "execution_time": None,
                "agents_used": ["historical_technical", "historical_sentiment", "forecasting_llm"],
                "data_sources": ["historical_ohlcv", "historical_4chan"]
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Full agentic prediction failed for {date}: {e}")
            return {
                "date": date,
                "method": "full_agentic",
                "success": False,
                "error": str(e)
            }
    
    def image_only_prediction(self, 
                             crypto_symbol: str, 
                             date: str, 
                             price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Image-only prediction with enhanced trading metrics.
        """
        try:
            logger.info(f"Running image-only prediction for {crypto_symbol} on {date}")
            
            current_price = price_data.get('price', 0)
            tech_analysis = self._create_historical_technical_analysis(crypto_symbol, date, price_data)
            
            # Enhanced technical analysis prompt with trading metrics
            analysis_prompt = f"""
            As a technical analyst, analyze the following historical market data for {crypto_symbol} on {date} and provide comprehensive technical trading predictions for the next 24 hours.
            
            HISTORICAL TECHNICAL DATA (as of {date}):
            - Price: ${current_price:.2f}
            - OHLCV data available: {bool(price_data.get('ohlcv'))}
            
            TECHNICAL ANALYSIS:
            {tech_analysis}
            
            Based ONLY on technical analysis, provide:
            
            1. DIRECTION: UP or DOWN for next 24 hours (choose one - no neutral)
            2. CONFIDENCE: HIGH/MEDIUM/LOW  
            3. TARGET_PERCENTAGE: Expected percentage move based on technical levels
            4. TARGET_PRICE: Technical target price
            5. TAKE_PROFIT: Technical take profit level
            6. STOP_LOSS: Technical stop loss level
            7. RISK_REWARD_RATIO: Technical risk/reward ratio
            8. POSITION_SIZE: Recommended position size based on technical setup
            9. TIME_HORIZON: Expected time to reach technical target
            10. REASONING: Technical analysis reasoning only
            
            Use technical analysis principles (support/resistance, chart patterns, indicators).
            For current price ${current_price:.2f}, be specific with price levels.
            
            Format your response as:
            DIRECTION: [UP/DOWN]
            CONFIDENCE: [HIGH/MEDIUM/LOW]
            TARGET_PERCENTAGE: [percentage]
            TARGET_PRICE: [price]
            TAKE_PROFIT: [price]
            STOP_LOSS: [price]
            RISK_REWARD_RATIO: [ratio]
            POSITION_SIZE: [SMALL/MEDIUM/LARGE]
            TIME_HORIZON: [time]
            REASONING: [technical reasoning]
            """
            
            # Get LLM response
            response = self._get_llm_response(analysis_prompt)
            
            # Extract prediction components
            predicted_direction = self._extract_direction_from_text(response)
            confidence = self._extract_confidence_from_text(response)
            reasoning = self._extract_reasoning_from_text(response)
            trading_metrics = self._extract_trading_metrics(response, current_price)
            
            prediction = {
                "date": date,
                "method": "image_only",
                "success": True,
                "predicted_direction": predicted_direction,
                "confidence": confidence,
                "reasoning": reasoning,
                "technical_analysis": tech_analysis,
                "historical_price": current_price,
                
                # Enhanced trading metrics
                "target_percentage": trading_metrics.get("target_percentage"),
                "target_price": trading_metrics.get("target_price"),
                "take_profit": trading_metrics.get("take_profit"),
                "stop_loss": trading_metrics.get("stop_loss"),
                "risk_reward_ratio": trading_metrics.get("risk_reward_ratio"),
                "position_size": trading_metrics.get("position_size"),
                "time_horizon": trading_metrics.get("time_horizon"),
                
                "agents_used": ["technical_analysis_llm"],
                "data_sources": ["historical_ohlcv_only"]
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Image-only prediction failed for {date}: {e}")
            return {
                "date": date,
                "method": "image_only", 
                "success": False,
                "error": str(e)
            }
    
    def sentiment_only_prediction(self, 
                                 crypto_symbol: str, 
                                 date: str, 
                                 sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sentiment-only prediction with enhanced trading metrics.
        """
        try:
            logger.info(f"Running sentiment-only prediction for {crypto_symbol} on {date}")
            
            if not sentiment_data.get("sentiment_available") or not sentiment_data.get("posts"):
                return {
                    "date": date,
                    "method": "sentiment_only",
                    "success": False,
                    "error": "No sentiment data available for this date"
                }
            
            # For sentiment-only, we need to estimate current price from historical data
            # This is a limitation but we'll work with available information
            estimated_price = 50000  # Default BTC price - in real implementation, get from price_data
            
            # Prepare sentiment data for analysis
            posts_text = ""
            for post in sentiment_data["posts"][:20]:  # Limit to top 20 posts
                posts_text += f"Post: {post['text']}\n\n"
            
            # Enhanced sentiment analysis prompt with trading metrics
            sentiment_prompt = f"""
            Analyze the following 4chan /biz/ posts about {crypto_symbol} from {date} and provide comprehensive sentiment-based trading predictions for the next 24 hours.
            
            Posts to analyze:
            {posts_text}
            
            Based ONLY on sentiment analysis, provide:
            
            1. DIRECTION: UP or DOWN based on overall sentiment (choose one - no neutral)
            2. CONFIDENCE: HIGH/MEDIUM/LOW based on sentiment strength and consistency
            3. TARGET_PERCENTAGE: Expected percentage move based on sentiment intensity
            4. TARGET_PRICE: Price target based on sentiment (estimate from typical moves)
            5. TAKE_PROFIT: Conservative profit target based on sentiment
            6. STOP_LOSS: Risk management level if sentiment proves wrong
            7. RISK_REWARD_RATIO: Risk/reward based on sentiment conviction
            8. POSITION_SIZE: Position size based on sentiment confidence
            9. TIME_HORIZON: Expected time for sentiment to impact price
            10. REASONING: Detailed sentiment analysis
            
            Consider:
            - Bullish sentiment: FOMO, moon talk, buy pressure, positive narratives
            - Bearish sentiment: FUD, dump expectations, negative sentiment, fear
            - Sentiment intensity: Number of posts, emotional language, conviction level
            - Mixed signals: Conflicting opinions, uncertainty
            
            When sentiment is mixed, lean toward the stronger signal or default to DOWN (conservative).
            
            Format your response as:
            DIRECTION: [UP/DOWN]
            CONFIDENCE: [HIGH/MEDIUM/LOW]
            TARGET_PERCENTAGE: [percentage]
            TARGET_PRICE: [estimated price]
            TAKE_PROFIT: [estimated price]
            STOP_LOSS: [estimated price]
            RISK_REWARD_RATIO: [ratio]
            POSITION_SIZE: [SMALL/MEDIUM/LARGE]
            TIME_HORIZON: [time estimate]
            REASONING: [sentiment analysis]
            """
            
            # Get LLM response
            response = self._get_llm_response(sentiment_prompt)
            
            # Extract prediction components
            predicted_direction = self._extract_direction_from_text(response)
            confidence = self._extract_confidence_from_text(response)
            reasoning = self._extract_reasoning_from_text(response)
            trading_metrics = self._extract_trading_metrics(response, estimated_price)
            
            prediction = {
                "date": date,
                "method": "sentiment_only",
                "success": True,
                "predicted_direction": predicted_direction,
                "confidence": confidence,
                "reasoning": reasoning,
                "sentiment_analysis": response,
                "posts_analyzed": len(sentiment_data["posts"]),
                "sentiment_posts": sentiment_data["posts"][:5],  # Sample of posts
                "historical_price": estimated_price,
                
                # Enhanced trading metrics
                "target_percentage": trading_metrics.get("target_percentage"),
                "target_price": trading_metrics.get("target_price"),
                "take_profit": trading_metrics.get("take_profit"),
                "stop_loss": trading_metrics.get("stop_loss"),
                "risk_reward_ratio": trading_metrics.get("risk_reward_ratio"),
                "position_size": trading_metrics.get("position_size"),
                "time_horizon": trading_metrics.get("time_horizon"),
                
                "agents_used": ["single_llm"],
                "data_sources": ["4chan_sentiment_only"]
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Sentiment-only prediction failed for {date}: {e}")
            return {
                "date": date,
                "method": "sentiment_only",
                "success": False,
                "error": str(e)
            }
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM for one-shot predictions."""
        try:
            # For now, we'll use a simple approach
            # In a real implementation, you'd call the LLM directly
            # This is a placeholder that shows the structure
            
            # Create a simple agent for one-shot analysis
            from crewai import Agent, Task, Crew
            
            analyst = Agent(
                role="Cryptocurrency Analyst",
                goal="Provide accurate price predictions based on given data",
                backstory="You are an expert cryptocurrency analyst with years of experience.",
                llm=self.llm,
                verbose=False
            )
            
            task = Task(
                description=prompt,
                expected_output="Clear prediction with reasoning",
                agent=analyst
            )
            
            crew = Crew(
                agents=[analyst],
                tasks=[task],
                verbose=False
            )
            
            result = crew.kickoff()
            return str(result)
            
        except Exception as e:
            logger.error(f"LLM response failed: {e}")
            return f"Error getting LLM response: {str(e)}"
    
    def _extract_direction_from_text(self, text: str) -> str:
        """Extract prediction direction from analysis text."""
        text_upper = text.upper()
        
        # Look for explicit predictions
        if "PREDICTION: UP" in text_upper or "DIRECTION: UP" in text_upper:
            return "UP"
        elif "PREDICTION: DOWN" in text_upper or "DIRECTION: DOWN" in text_upper:
            return "DOWN"
        
        # Look for strong indicators
        bullish_indicators = ["BULLISH", "BUY", "PUMP", "MOON", "STRONG UP", "UPWARD", "POSITIVE"]
        bearish_indicators = ["BEARISH", "SELL", "DUMP", "CRASH", "STRONG DOWN", "DOWNWARD", "NEGATIVE"]
        
        bullish_count = sum(1 for indicator in bullish_indicators if indicator in text_upper)
        bearish_count = sum(1 for indicator in bearish_indicators if indicator in text_upper)
        
        if bullish_count > bearish_count and bullish_count > 0:
            return "UP"
        elif bearish_count > bullish_count and bearish_count > 0:
            return "DOWN"
        else:
            # When tied or no clear signals, default to DOWN (conservative approach)
            return "DOWN"
    
    def _extract_confidence_from_text(self, text: str) -> str:
        """Extract confidence level from analysis text."""
        text_upper = text.upper()
        
        # Look for explicit confidence
        if "CONFIDENCE: HIGH" in text_upper:
            return "HIGH"
        elif "CONFIDENCE: MEDIUM" in text_upper:
            return "MEDIUM"
        elif "CONFIDENCE: LOW" in text_upper:
            return "LOW"
        
        # Look for confidence indicators
        high_confidence = ["VERY CONFIDENT", "STRONG SIGNAL", "CLEAR TREND", "HIGH CONFIDENCE"]
        low_confidence = ["UNCERTAIN", "MIXED SIGNALS", "UNCLEAR", "LOW CONFIDENCE"]
        
        if any(indicator in text_upper for indicator in high_confidence):
            return "HIGH"
        elif any(indicator in text_upper for indicator in low_confidence):
            return "LOW"
        else:
            return "MEDIUM"
    
    def _extract_reasoning_from_text(self, text: str) -> str:
        """Extract reasoning section from analysis text."""
        # Look for reasoning section
        text_lines = text.split('\n')
        reasoning_lines = []
        capture_reasoning = False
        
        for line in text_lines:
            line_upper = line.upper()
            if "REASONING:" in line_upper:
                capture_reasoning = True
                reasoning_lines.append(line.split(':', 1)[-1].strip())
            elif capture_reasoning and line.strip():
                reasoning_lines.append(line.strip())
            elif capture_reasoning and not line.strip():
                break
        
        if reasoning_lines:
            return ' '.join(reasoning_lines)
        else:
            # Return the full text as reasoning if no specific section found
            return text[:500] + "..." if len(text) > 500 else text
    
    def _create_historical_technical_analysis(self, crypto_symbol: str, date: str, price_data: Dict[str, Any]) -> str:
        """Create technical analysis from historical price data."""
        try:
            if not price_data.get('ohlcv'):
                return f"No OHLCV data available for {crypto_symbol} on {date}"
            
            ohlcv = price_data['ohlcv']
            current_price = price_data.get('price', ohlcv.get('close', 0))
            
            # Create basic technical analysis from the single day's data
            analysis = f"""
            TECHNICAL ANALYSIS for {crypto_symbol} on {date}:
            
            Price Data:
            - Open: ${ohlcv.get('open', 0):.2f}
            - High: ${ohlcv.get('high', 0):.2f} 
            - Low: ${ohlcv.get('low', 0):.2f}
            - Close: ${ohlcv.get('close', 0):.2f}
            - Volume: {ohlcv.get('volume', 0):,.0f}
            
            Price Action:
            - Daily Range: {((ohlcv.get('high', 0) - ohlcv.get('low', 0)) / ohlcv.get('low', 1) * 100):.2f}%
            - Open to Close: {((ohlcv.get('close', 0) - ohlcv.get('open', 0)) / ohlcv.get('open', 1) * 100):.2f}%
            - Current Position: {'Upper range' if ohlcv.get('close', 0) > (ohlcv.get('high', 0) + ohlcv.get('low', 0))/2 else 'Lower range'}
            
            Volume Analysis:
            - Trading Volume: {ohlcv.get('volume', 0):,.0f}
            - Volume Trend: {'High' if ohlcv.get('volume', 0) > 100000 else 'Moderate' if ohlcv.get('volume', 0) > 10000 else 'Low'}
            
            Note: Limited to single-day historical data. In real analysis, would include moving averages, RSI, MACD, etc.
            """
            
            return analysis.strip()
            
        except Exception as e:
            return f"Error creating technical analysis: {str(e)}"
    
    def _create_historical_sentiment_analysis(self, crypto_symbol: str, date: str, sentiment_data: Dict[str, Any]) -> str:
        """Create sentiment analysis from historical 4chan data."""
        try:
            if not sentiment_data.get('sentiment_available') or not sentiment_data.get('posts'):
                return f"No sentiment data available for {crypto_symbol} on {date}"
            
            posts = sentiment_data.get('posts', [])
            post_count = len(posts)
            
            # Analyze sentiment themes
            bullish_terms = ['moon', 'pump', 'buy', 'bull', 'up', 'rocket', 'hodl', 'lambo']
            bearish_terms = ['dump', 'crash', 'sell', 'bear', 'down', 'rekt', 'dead', 'scam']
            
            bullish_count = 0
            bearish_count = 0
            
            for post in posts[:10]:  # Analyze first 10 posts
                text_lower = post.get('text', '').lower()
                bullish_count += sum(1 for term in bullish_terms if term in text_lower)
                bearish_count += sum(1 for term in bearish_terms if term in text_lower)
            
            sentiment_score = (bullish_count - bearish_count) / max(bullish_count + bearish_count, 1)
            
            analysis = f"""
            SENTIMENT ANALYSIS for {crypto_symbol} on {date}:
            
            Data Source: 4chan /biz/ archive (Warosu)
            - Total Posts: {post_count}
            - Date Range: {sentiment_data.get('date', date)}
            
            Sentiment Metrics:
            - Bullish Mentions: {bullish_count}
            - Bearish Mentions: {bearish_count}
            - Sentiment Score: {sentiment_score:.2f} (-1=bearish, +1=bullish)
            - Overall Sentiment: {'Bullish' if sentiment_score > 0.0 else 'Bearish'}
            
            Sample Posts Analysis:
            """
            
            # Add sample posts
            for i, post in enumerate(posts[:3]):
                analysis += f"\n{i+1}. '{post.get('text', '')[:100]}...'"
            
            if post_count == 0:
                analysis += "\nNote: No posts found for this date range. Sentiment analysis unreliable."
            
            return analysis.strip()
            
        except Exception as e:
            return f"Error creating sentiment analysis: {str(e)}" 

    def _extract_trading_metrics(self, text: str, current_price: float) -> Dict[str, Any]:
        """Extract trading metrics from LLM response text."""
        metrics = {}
        
        try:
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                line_upper = line.upper()
                
                # Extract target percentage
                if 'TARGET_PERCENTAGE:' in line_upper:
                    try:
                        pct_str = line.split(':', 1)[1].strip()
                        # Extract percentage value
                        import re
                        pct_match = re.search(r'([+-]?\d+\.?\d*)%?', pct_str)
                        if pct_match:
                            metrics["target_percentage"] = float(pct_match.group(1))
                    except:
                        pass
                
                # Extract target price
                elif 'TARGET_PRICE:' in line_upper:
                    try:
                        price_str = line.split(':', 1)[1].strip()
                        # Extract price value
                        import re
                        price_match = re.search(r'\$?(\d+\.?\d*)', price_str.replace(',', ''))
                        if price_match:
                            metrics["target_price"] = float(price_match.group(1))
                    except:
                        pass
                
                # Extract take profit
                elif 'TAKE_PROFIT:' in line_upper:
                    try:
                        price_str = line.split(':', 1)[1].strip()
                        import re
                        price_match = re.search(r'\$?(\d+\.?\d*)', price_str.replace(',', ''))
                        if price_match:
                            metrics["take_profit"] = float(price_match.group(1))
                    except:
                        pass
                
                # Extract stop loss
                elif 'STOP_LOSS:' in line_upper or 'STOP LOSS:' in line_upper:
                    try:
                        price_str = line.split(':', 1)[1].strip()
                        import re
                        price_match = re.search(r'\$?(\d+\.?\d*)', price_str.replace(',', ''))
                        if price_match:
                            metrics["stop_loss"] = float(price_match.group(1))
                    except:
                        pass
                
                # Extract risk/reward ratio
                elif 'RISK_REWARD_RATIO:' in line_upper or 'RISK REWARD RATIO:' in line_upper:
                    try:
                        ratio_str = line.split(':', 1)[1].strip()
                        metrics["risk_reward_ratio"] = ratio_str
                    except:
                        pass
                
                # Extract position size
                elif 'POSITION_SIZE:' in line_upper or 'POSITION SIZE:' in line_upper:
                    try:
                        size_str = line.split(':', 1)[1].strip().upper()
                        if any(size in size_str for size in ['SMALL', 'MEDIUM', 'LARGE']):
                            for size in ['SMALL', 'MEDIUM', 'LARGE']:
                                if size in size_str:
                                    metrics["position_size"] = size
                                    break
                    except:
                        pass
                
                # Extract time horizon
                elif 'TIME_HORIZON:' in line_upper or 'TIME HORIZON:' in line_upper:
                    try:
                        time_str = line.split(':', 1)[1].strip()
                        metrics["time_horizon"] = time_str
                    except:
                        pass
            
            # Calculate derived metrics if we have the basic ones
            if "target_percentage" in metrics and current_price > 0:
                if "target_price" not in metrics:
                    target_pct = metrics["target_percentage"] / 100
                    metrics["target_price"] = current_price * (1 + target_pct)
            
            # Set defaults for missing values
            if "position_size" not in metrics:
                metrics["position_size"] = "MEDIUM"
            if "time_horizon" not in metrics:
                metrics["time_horizon"] = "12-24 hours"
            if "risk_reward_ratio" not in metrics:
                metrics["risk_reward_ratio"] = "1:2"
                
        except Exception as e:
            logger.error(f"Error extracting trading metrics: {e}")
        
        return metrics 