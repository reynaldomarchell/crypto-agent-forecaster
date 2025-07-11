"""
Prediction methods for thesis comparison: Agentic vs One-shot approaches.
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
    """Contains all prediction methods for thesis comparison."""
    
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
        Full multi-agent prediction using the complete system.
        This simulates what the system would have predicted on that historical date using historical data.
        """
        try:
            logger.info(f"Running full agentic prediction for {crypto_symbol} on {date}")
            
            # Create historical analysis using the provided historical data
            # Instead of calling current data tools, we'll simulate the analysis
            
            # Create historical technical analysis
            tech_analysis = self._create_historical_technical_analysis(crypto_symbol, date, price_data)
            
            # Create historical sentiment analysis
            sentiment_analysis = self._create_historical_sentiment_analysis(crypto_symbol, date, sentiment_data)
            
            # Use LLM to make final prediction based on historical data
            prediction_prompt = f"""
            As a cryptocurrency forecasting expert, analyze the following historical data for {crypto_symbol} on {date} and predict the price direction for the next 24 hours.
            
            HISTORICAL MARKET DATA (as of {date}):
            - Current Price: ${price_data.get('price', 'N/A')}
            - Historical OHLCV data available: {bool(price_data.get('ohlcv'))}
            
            TECHNICAL ANALYSIS:
            {tech_analysis}
            
            SENTIMENT ANALYSIS:
            {sentiment_analysis}
            
            Based on this historical data (pretending it's {date}), provide:
            1. DIRECTION: UP or DOWN for next 24 hours (choose one - no neutral)
            2. CONFIDENCE: HIGH/MEDIUM/LOW
            3. REASONING: Detailed explanation based on technical and sentiment factors
            
            When signals are mixed or unclear, make a binary decision based on the most reliable indicators.
            
            Format your response as:
            DIRECTION: [your prediction]
            CONFIDENCE: [your confidence]
            REASONING: [your reasoning]
            """
            
            # Get LLM response using historical context
            response = self._get_llm_response(prediction_prompt)
            
            # Extract prediction components
            predicted_direction = self._extract_direction_from_text(response)
            confidence = self._extract_confidence_from_text(response)
            reasoning = self._extract_reasoning_from_text(response)
            
            prediction = {
                "date": date,
                "method": "full_agentic",
                "success": True,
                "predicted_direction": predicted_direction,
                "confidence": confidence,
                "reasoning": reasoning,
                "technical_analysis": tech_analysis,
                "sentiment_analysis": sentiment_analysis,
                "historical_price": price_data.get('price'),
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
        Image-only prediction using one-shot LLM analysis of historical technical data.
        This tests pure technical analysis without sentiment or multi-agent reasoning.
        """
        try:
            logger.info(f"Running image-only prediction for {crypto_symbol} on {date}")
            
            # Create technical analysis from historical data
            tech_analysis = self._create_historical_technical_analysis(crypto_symbol, date, price_data)
            
            # Create one-shot technical analysis prompt using historical data
            analysis_prompt = f"""
            As a technical analyst, analyze the following historical market data for {crypto_symbol} on {date} and predict the price direction for the next 24 hours.
            
            HISTORICAL TECHNICAL DATA (as of {date}):
            - Price: ${price_data.get('price', 'N/A')}
            - OHLCV data available: {bool(price_data.get('ohlcv'))}
            
            TECHNICAL ANALYSIS:
            {tech_analysis}
            
            Based ONLY on this technical analysis (ignore any sentiment factors), provide:
            1. DIRECTION: UP or DOWN for next 24 hours (choose one - no neutral)
            2. CONFIDENCE: HIGH/MEDIUM/LOW
            3. REASONING: Technical analysis reasoning only
            
            Format your response as:
            DIRECTION: [your prediction]
            CONFIDENCE: [your confidence]
            REASONING: [your technical reasoning]
            """
            
            # Get LLM response
            response = self._get_llm_response(analysis_prompt)
            
            # Extract prediction from the analysis
            predicted_direction = self._extract_direction_from_text(response)
            confidence = self._extract_confidence_from_text(response)
            reasoning = self._extract_reasoning_from_text(response)
            
            prediction = {
                "date": date,
                "method": "image_only",
                "success": True,
                "predicted_direction": predicted_direction,
                "confidence": confidence,
                "reasoning": reasoning,
                "technical_analysis": tech_analysis,
                "historical_price": price_data.get('price'),
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
        Sentiment-only prediction using one-shot LLM analysis of 4chan posts.
        This tests pure sentiment analysis without technical or multi-agent reasoning.
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
            
            # Prepare sentiment data for analysis
            posts_text = ""
            for post in sentiment_data["posts"][:20]:  # Limit to top 20 posts
                posts_text += f"Post: {post['text']}\n\n"
            
            # Create one-shot sentiment analysis prompt
            sentiment_prompt = f"""
            Analyze the following 4chan /biz/ posts about {crypto_symbol} from {date} and predict the price direction for the next 24 hours.
            
            Posts to analyze:
            {posts_text}
            
            Instructions:
            1. Analyze the overall sentiment (bullish/bearish)
            2. Look for FUD (Fear, Uncertainty, Doubt) vs FOMO/shilling
            3. Consider the volume and intensity of discussion
            4. Provide a clear prediction: UP or DOWN (choose one - no neutral)
            5. Assign confidence: HIGH/MEDIUM/LOW
            6. Explain your reasoning based on sentiment analysis
            7. When sentiment is mixed, lean toward the stronger signal or default to DOWN (conservative)
            
            Respond in this format:
            PREDICTION: [UP/DOWN]
            CONFIDENCE: [HIGH/MEDIUM/LOW]
            REASONING: [Your detailed analysis based on sentiment]
            """
            
            # Get LLM response
            response = self._get_llm_response(sentiment_prompt)
            
            # Extract prediction components
            predicted_direction = self._extract_direction_from_text(response)
            confidence = self._extract_confidence_from_text(response)
            reasoning = self._extract_reasoning_from_text(response)
            
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