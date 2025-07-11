"""
Constants for the CLI application.
"""

from typing import List, Tuple, Dict, Any

# Application metadata
APP_NAME = "crypto-agent-forecaster"
APP_DESCRIPTION = """CryptoAgentForecaster - Advanced AI-Driven Cryptocurrency Forecasting

A sophisticated multi-agent system that combines market data analysis, sentiment analysis, 
technical analysis, and LLM-powered forecasting to predict cryptocurrency price movements.

KEY FEATURES:
• Multi-agent AI analysis with 4 specialized agents
• Real-time market data from CoinGecko API  
• Social sentiment analysis from 4chan /biz/
• Advanced technical analysis with chart generation
• Multiple LLM provider support (OpenAI, Anthropic, Google)
• Interactive verbose mode for detailed execution tracking
• Automatic result saving with charts and logs
• Professional forecast reports in Markdown format

AGENTS & TOOLS:
• Market Data Agent → CoinGecko API integration
• Sentiment Agent → 4chan /biz/ social sentiment analysis  
• Technical Agent → TA indicators + chart generation
• Forecasting Agent → Multi-modal data fusion & prediction

OUTPUTS:
• Price direction prediction (UP/DOWN/NEUTRAL)
• Confidence scoring (HIGH/MEDIUM/LOW)
• Technical analysis charts (PNG format)
• Detailed reasoning and explanation
• Complete execution logs and metadata
• Professional markdown reports with embedded charts

Use 'crypto-agent-forecaster COMMAND --help' for detailed command information.
"""

# Banner text
BANNER_TEXT = """
CryptoAgentForecaster
Multimodal AI-Driven Cryptocurrency Price Forecasting

Powered by:
• CoinGecko API for market data
• Hosted LLMs (OpenAI, Anthropic, Google)
• Multi-agent analysis system
• Technical analysis & sentiment fusion
• 4chan /biz/ sentiment analysis
"""

# Popular cryptocurrencies
POPULAR_CRYPTOS: List[Tuple[str, str, str]] = [
    ("bitcoin", "Bitcoin", "BTC"),
    ("ethereum", "Ethereum", "ETH"), 
    ("solana", "Solana", "SOL"),
    ("cardano", "Cardano", "ADA"),
    ("polkadot", "Polkadot", "DOT"),
    ("chainlink", "Chainlink", "LINK"),
    ("polygon", "Polygon", "MATIC"),
    ("avalanche-2", "Avalanche", "AVAX"),
    ("dogecoin", "Dogecoin", "DOGE"),
    ("shiba-inu", "Shiba Inu", "SHIB")
]

# API key URLs
API_KEY_URLS: Dict[str, str] = {
    "OpenAI": "https://platform.openai.com/api-keys",
    "Anthropic": "https://console.anthropic.com/",
    "Google": "https://aistudio.google.com/app/apikey",
    "CoinGecko": "https://www.coingecko.com/en/api/pricing"
}

# Setup instructions
SETUP_INSTRUCTIONS: List[str] = [
    "Copy 'env_example' to '.env'",
    "Add your API keys to the .env file",
    "You need at least one LLM provider (OpenAI, Anthropic, or Google)",
    "CoinGecko API key is optional but recommended"
]

# Common troubleshooting tips
TROUBLESHOOTING_TIPS: List[str] = [
    "'env already loaded' → Environment loading multiple times (should be fixed)",
    "'LLM Failed' → Check API keys and internet connectivity",
    "Missing models → Verify model names match provider specifications",
    "Rate limiting → Wait a few minutes and reduce request frequency"
]

# Next steps for users
NEXT_STEPS: List[str] = [
    "Run: crypto-agent-forecaster config",
    "Run: crypto-agent-forecaster test --quick",
    "Run: crypto-agent-forecaster forecast bitcoin --verbose",
    "Check the 'results/' folder for outputs"
]

# Default values
DEFAULT_CRYPTO = "bitcoin"
DEFAULT_HORIZON = "24 hours"
DEFAULT_QUICK_TEST_DAYS = 7
DEFAULT_FULL_TEST_DAYS = 30

# Error messages
ERROR_MESSAGES: Dict[str, str] = {
    "no_config": "No LLM providers configured! Please set up API keys.",
    "provider_unavailable": "Provider '{}' not available or not configured",
    "test_failed": "Test failed: {}",
    "forecast_failed": "Forecast failed: {}",
    "debug_failed": "Debug failed: {}",
    "config_create_hint": "Create a .env file with your API keys. See env_example for template."
}

# Success messages
SUCCESS_MESSAGES: Dict[str, str] = {
    "forecast_completed": "Forecast completed successfully!",
    "all_tests_passed": "All tests passed!",
    "config_validated": "Configuration validated successfully!"
}

# Table headers and styles
TABLE_STYLES: Dict[str, str] = {
    "provider_column": "cyan",
    "status_column": "white", 
    "model_column": "yellow",
    "green_style": "green",
    "red_style": "red",
    "value_column": "yellow",
    "name_column": "white"
}

# File patterns and extensions
FILE_EXTENSIONS: Dict[str, str] = {
    "env_example": "env_example",
    "env_file": ".env",
    "results_folder": "results/"
}

# Agent types for debugging
AGENT_TYPES: List[str] = ["market_data", "sentiment", "technical", "forecasting"] 