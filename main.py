#!/usr/bin/env python3
"""
CryptoAgentForecaster - CLI Application

A multimodal cryptocurrency forecasting system using hosted LLMs and agent-based architecture.

Features:
- Clean, truncated logging (no verbose JSON or base64 spam)
- Automatic result saving to organized folders for each run
- Technical analysis charts saved as PNG files
- Complete run logs and forecast results in structured directories
"""

import typer
import sys
from typing import Optional

from src.cli.constants import APP_NAME, APP_DESCRIPTION
from src.cli.commands import CommandHandler

# Initialize CLI app
app = typer.Typer(
    name=APP_NAME,
    help=APP_DESCRIPTION,
    rich_markup_mode="rich"
)

# Initialize command handler
handler = CommandHandler()


@app.command()
def forecast(
    crypto: str = typer.Argument(
        ..., 
        help="Cryptocurrency name (use CoinGecko ID). Examples: bitcoin, ethereum, solana, cardano",
        metavar="CRYPTO_NAME"
    ),
    horizon: str = typer.Option(
        "24 hours", 
        "--horizon", "-h", 
        help="Forecast time horizon. Examples: '24 hours', '3 days', '1 week'",
        metavar="TIME_PERIOD"
    ),
    provider: Optional[str] = typer.Option(
        None, 
        "--provider", "-p", 
        help="LLM provider to use. Options: 'openai', 'anthropic', 'google'",
        metavar="PROVIDER"
    ),
    model: Optional[str] = typer.Option(
        None, 
        "--model", "-m", 
        help="Specific model to use. Examples: 'gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-1.5-pro'",
        metavar="MODEL_NAME"
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", "-v", 
        help="Enable verbose output with detailed execution tracking, agent interactions, and real-time progress"
    ),
    yes: bool = typer.Option(
        False, 
        "--yes", "-y", 
        help="Skip confirmation prompt and proceed automatically"
    )
):
    """
    Generate a comprehensive cryptocurrency price forecast using AI agents.
    
    This command orchestrates a sophisticated forecasting workflow using 4 specialized AI agents:
    
    WORKFLOW PROCESS:
    1. Market Data Agent collects 30 days of OHLCV data + current market stats
    2. Sentiment Agent analyzes social sentiment from 4chan /biz/ discussions  
    3. Technical Agent performs TA analysis and generates interactive charts
    4. Forecasting Agent synthesizes all data into final prediction
    
    OUTPUTS GENERATED:
    • Forecast direction (UP/DOWN/NEUTRAL) with confidence score
    • Technical analysis charts saved as PNG files
    • Complete execution logs and agent interactions
    • Professional markdown report with embedded charts
    • Structured JSON data for programmatic access
    
    RESULTS LOCATION:
    All results are automatically saved to 'results/CRYPTO_TIMESTAMP/' containing:
    • forecast_results.json - Complete forecast data
    • charts/ - Technical analysis charts (PNG format)
    • run_logs.txt - Detailed execution logs  
    • README.md - Professional report with embedded charts
    
    VERBOSE MODE:
    Use --verbose flag for interactive execution tracking including:
    • Real-time agent task progress
    • Tool usage monitoring (CoinGecko, 4chan, TA tools)
    • Detailed workflow visualization
    • Agent communication logs
    • Enhanced error diagnostics
    
    EXAMPLES:
        # Basic forecast with default settings
        crypto-agent-forecaster forecast bitcoin
        
        # Verbose mode with detailed tracking
        crypto-agent-forecaster forecast ethereum --verbose
        
        # Custom time horizon
        crypto-agent-forecaster forecast solana --horizon "3 days"
        
        # Specify LLM provider and model
        crypto-agent-forecaster forecast cardano --provider anthropic --model claude-3-5-sonnet-20241022
        
        # Full customization with verbose output
        crypto-agent-forecaster forecast bitcoin --horizon "1 week" --provider openai --model gpt-4o --verbose
        
        # Quick analysis for altcoins
        crypto-agent-forecaster forecast chainlink --horizon "12 hours" --verbose
        
        # Automated execution without confirmation prompt
        crypto-agent-forecaster forecast bitcoin --yes
        
        # Fully automated with custom settings
        crypto-agent-forecaster forecast ethereum --horizon "3 days" --provider anthropic --verbose --yes
    
    SUPPORTED CRYPTOCURRENCIES:
    Use 'crypto-agent-forecaster list-cryptos' to see popular options, or any valid CoinGecko ID.
    Common examples: bitcoin, ethereum, solana, cardano, polkadot, chainlink, avalanche-2
    
    LLM CONFIGURATION:
    Use 'crypto-agent-forecaster models' to see available providers and recommendations.
    Default provider/model can be set in .env file or overridden with --provider/--model flags.
    
    TROUBLESHOOTING:
    • Use 'crypto-agent-forecaster config' to check API key configuration
    • Use 'crypto-agent-forecaster test' to verify system components
    • Enable --verbose for detailed error diagnostics
    • Check results/ folder for saved outputs even on partial failures
    """
    success = handler.handle_forecast(crypto, horizon, provider, model, verbose, yes)
    if not success:
        raise typer.Exit(1)


@app.command()
def config():
    """
    Display current configuration and setup information.
    
    This command shows the current system configuration including:
    • LLM provider API key status (OpenAI, Anthropic, Google)
    • Available models and their specifications
    • CoinGecko API configuration
    • Default provider and model settings
    • Setup instructions for missing configurations
    
    CONFIGURATION FILES:
    • .env - Main configuration file with API keys
    • .env.example - Template file with configuration examples
    
    REQUIRED API KEYS:
    At minimum, you need ONE LLM provider configured:
    • OPENAI_API_KEY - For GPT models (gpt-4o, gpt-4o-mini)
    • ANTHROPIC_API_KEY - For Claude models (claude-3-5-sonnet)  
    • GOOGLE_API_KEY - For Gemini models (gemini-1.5-pro, gemini-2.0-flash)
    
    OPTIONAL API KEYS:
    • COINGECKO_API_KEY - For higher rate limits and pro features
    
    SETUP PROCESS:
    1. Copy '.env.example' to '.env'
    2. Add your API keys to the .env file
    3. Run 'crypto-agent-forecaster config' to verify
    4. Use 'crypto-agent-forecaster test' to validate functionality
    
    COST OPTIMIZATION:
    Use 'crypto-agent-forecaster models' to see cost comparisons and select
    the most cost-effective model for your use case.
    """
    success = handler.handle_config()
    if not success:
        raise typer.Exit(1)


@app.command()
def debug():
    """
    Debug environment and configuration issues.
    
    This command provides detailed diagnostic information to help troubleshoot
    common issues with environment setup, API key configuration, and LLM connectivity.
    
    DIAGNOSTIC INFORMATION:
    • Environment variable loading status
    • API key configuration validation
    • LLM provider connectivity tests
    • Agent-specific configuration validation
    • Common configuration problems
    
    TROUBLESHOOTING MODES:
    • Environment debugging for "env already loaded" errors
    • LLM configuration validation for each agent type
    • API key presence and format validation
    • CrewAI integration diagnostics
    
    COMMON ISSUES DIAGNOSED:
    • "LLM Failed" errors
    • "env already loaded" environment conflicts
    • Missing or invalid API keys
    • Model availability and configuration mismatches
    • Rate limiting and connectivity issues
    
    WHEN TO USE:
    • After getting LLM failure errors
    • When setting up the system for the first time
    • Before running forecasts to validate configuration
    • When switching between different LLM providers
    
    OUTPUT INCLUDES:
    • Detailed environment status
    • Per-agent LLM configuration validation
    • Suggested fixes for common problems
    • Environment variable presence checks
    """
    success = handler.handle_debug()
    if not success:
        raise typer.Exit(1)


@app.command()
def test(
    crypto: str = typer.Option(
        "bitcoin", 
        help="Cryptocurrency to test with (use CoinGecko ID)",
        metavar="CRYPTO_NAME"
    ),
    quick: bool = typer.Option(
        False, 
        "--quick", 
        help="Quick test mode (fewer data points, faster execution)"
    )
):
    """
    Test system components and validate configuration.
    
    This command performs comprehensive system testing to ensure all components
    are working correctly before running forecasts.
    
    TESTS PERFORMED:
    1. CoinGecko API connectivity and data retrieval
    2. 4chan /biz/ API access and sentiment data collection  
    3. Technical analysis tool functionality and chart generation
    4. Data processing and integration workflows
    
    TEST MODES:
    • Standard Mode: Full testing with 30 days of data
    • Quick Mode (--quick): Faster testing with 7 days of data
    
    TROUBLESHOOTING:
    If tests fail, check:
    • Internet connectivity
    • API key configuration (use 'crypto-agent-forecaster config')
    • Rate limiting (wait a few minutes and retry)
    • Firewall settings blocking API access
    
    USE CASES:
    • Initial setup validation
    • Debugging forecast issues  
    • Testing new cryptocurrencies
    • Verifying system health after configuration changes
    
    EXAMPLES:
        # Test with default Bitcoin
        crypto-agent-forecaster test
        
        # Quick test for faster validation
        crypto-agent-forecaster test --quick
        
        # Test specific cryptocurrency
        crypto-agent-forecaster test --crypto ethereum
        
        # Quick test with custom crypto
        crypto-agent-forecaster test --crypto solana --quick
    """
    success = handler.handle_test(crypto, quick)
    if not success:
        raise typer.Exit(1)


@app.command(name="list-cryptos")
def list_cryptos():
    """
    List popular cryptocurrencies available for analysis.
    
    This command displays a curated list of popular cryptocurrencies that work
    well with the forecasting system, along with their CoinGecko IDs and symbols.
    
    CRYPTOCURRENCY SELECTION:
    The listed cryptocurrencies are selected based on:
    • High trading volume and market liquidity
    • Active social media presence (for sentiment analysis)
    • Strong technical analysis signal availability
    • Reliable historical data from CoinGecko
    
    USAGE TIPS:
    • Use the 'ID' column values as the crypto parameter for forecasts
    • All major cryptocurrencies beyond this list are also supported
    • Use any valid CoinGecko ID (check coinapi.com for full list)
    • Case-insensitive: 'Bitcoin', 'bitcoin', and 'BITCOIN' all work
    
    FINDING OTHER CRYPTOCURRENCIES:
    For cryptocurrencies not listed here:
    1. Visit coinapi.com or api.coingecko.com
    2. Search for your desired cryptocurrency
    3. Use the 'id' field from the API response
    4. Test with 'crypto-agent-forecaster test --crypto YOUR_CRYPTO_ID'
    
    ANALYSIS QUALITY:
    Popular cryptocurrencies typically provide:
    • More social sentiment data from 4chan /biz/
    • Better technical analysis patterns
    • More reliable price predictions
    • Higher quality chart generation
    
    EXAMPLES:
        # Forecast popular cryptocurrencies
        crypto-agent-forecaster forecast bitcoin
        crypto-agent-forecaster forecast ethereum --verbose
        crypto-agent-forecaster forecast solana --horizon "3 days"
    """
    success = handler.handle_list_cryptos()
    if not success:
        raise typer.Exit(1)


@app.command()
def models():
    """
    Display available LLM models, specifications, and recommendations.
    
    This command provides comprehensive information about available Language Models
    including costs, capabilities, and task-specific recommendations.
    
    SUPPORTED PROVIDERS:
    • OpenAI: GPT-4o, GPT-4o-mini (balanced performance and cost)
    • Anthropic: Claude-3.5-Sonnet (excellent reasoning and analysis)
    • Google: Gemini-1.5-Pro, Gemini-2.0-Flash (large context windows)
    
    COST INFORMATION:
    • Input costs: Price per 1,000 input tokens
    • Output costs: Price per 1,000 output tokens  
    • Typical forecast costs: $0.01 - $0.10 depending on model choice
    • Use cheaper models for testing, premium models for production
    
    TASK-SPECIFIC RECOMMENDATIONS:
    • Sentiment Analysis: Claude models excel at nuanced sentiment understanding
    • Technical Analysis: GPT models provide structured technical insights
    • Data Fusion: All models perform well, choose based on cost preference
    • Chart Analysis: Models with vision capabilities (GPT-4o) for future multimodal features
    
    CONFIGURATION:
    • Set default provider/model in .env file:
      DEFAULT_LLM_PROVIDER=openai
      DEFAULT_LLM_MODEL=gpt-4o-mini
    • Override per-forecast with --provider and --model flags
    • Use 'crypto-agent-forecaster config' to check current settings
    
    PERFORMANCE TIPS:
    • Start with gpt-4o-mini for cost-effective testing
    • Upgrade to claude-3-5-sonnet for production forecasts
    • Use gemini models for experimental features and large context needs
    • Enable --verbose to see model performance in real-time
    
    MODEL COMPARISON:
    • Speed: Gemini-2.0-Flash > GPT-4o-mini > Claude-3.5-Sonnet > GPT-4o
    • Quality: Claude-3.5-Sonnet ≥ GPT-4o > GPT-4o-mini > Gemini models
    • Cost: GPT-4o-mini < Gemini < Claude < GPT-4o
    • Context: Gemini (2M tokens) > Claude (200K) > GPT (128K)
    
    EXAMPLES:
        # Use specific model for forecast
        crypto-agent-forecaster forecast bitcoin --provider openai --model gpt-4o
        
        # Cost-effective option
        crypto-agent-forecaster forecast ethereum --provider openai --model gpt-4o-mini
        
        # Premium analysis
        crypto-agent-forecaster forecast solana --provider anthropic --model claude-3-5-sonnet-20241022
    """
    success = handler.handle_models()
    if not success:
        raise typer.Exit(1)


@app.command()
def help():
    """
    Quick help and usage examples for common tasks.
    
    This command provides a quick reference guide for the most common use cases
    and workflow examples to get you started quickly.
    
    QUICK START:
    1. Check configuration: crypto-agent-forecaster config
    2. Test system: crypto-agent-forecaster test --quick  
    3. Run first forecast: crypto-agent-forecaster forecast bitcoin --verbose
    4. Check results in the 'results/' folder
    
    COMMON WORKFLOWS:
    
    BASIC FORECASTING:
        # Simple Bitcoin forecast
        crypto-agent-forecaster forecast bitcoin
        
        # Ethereum with custom timeframe
        crypto-agent-forecaster forecast ethereum --horizon "3 days"
        
        # Verbose mode for learning/debugging
        crypto-agent-forecaster forecast solana --verbose
    
    CONFIGURATION & TESTING:
        # Check API keys and setup
        crypto-agent-forecaster config
        
        # Quick system validation
        crypto-agent-forecaster test --quick
        
        # Full system test with specific crypto
        crypto-agent-forecaster test --crypto cardano
    
    MODEL SELECTION:
        # Use cost-effective model
        crypto-agent-forecaster forecast bitcoin --provider openai --model gpt-4o-mini
        
        # Premium quality analysis
        crypto-agent-forecaster forecast ethereum --provider anthropic
        
        # See all available models
        crypto-agent-forecaster models
    
    CRYPTOCURRENCY DISCOVERY:
        # See popular cryptocurrencies
        crypto-agent-forecaster list-cryptos
        
        # Try different altcoins
        crypto-agent-forecaster forecast chainlink --verbose
        crypto-agent-forecaster forecast avalanche-2 --horizon "1 week"
    
    PRODUCTION WORKFLOWS:
        # Daily BTC analysis with high-quality model
        crypto-agent-forecaster forecast bitcoin --provider anthropic --horizon "24 hours" --verbose --yes
        
        # Weekly portfolio review
        crypto-agent-forecaster forecast ethereum --horizon "1 week" --provider openai --model gpt-4o --yes
        
        # Quick market sentiment check
        crypto-agent-forecaster forecast bitcoin --horizon "4 hours" --provider openai --model gpt-4o-mini --yes
    
    OUTPUT LOCATIONS:
        # All results saved to: results/CRYPTO_TIMESTAMP/
        # ├── forecast_results.json    # Complete data
        # ├── charts/                  # Technical analysis charts (PNG)
        # ├── run_logs.txt            # Execution logs
        # └── README.md               # Professional report with charts
    
    TROUBLESHOOTING:
        # Debug configuration issues
        crypto-agent-forecaster config
        
        # Test specific components
        crypto-agent-forecaster test --crypto bitcoin --quick
        
        # Get detailed error info
        crypto-agent-forecaster forecast bitcoin --verbose
        
        # Check model availability
        crypto-agent-forecaster models
    
    PRO TIPS:
    • Use --verbose flag to understand what's happening
    • Start with gpt-4o-mini for cost-effective testing
    • Check 'results/' folder for all saved outputs
    • Popular cryptos (BTC, ETH) have better sentiment data
    • Set default provider in .env file to avoid typing --provider every time
    • Use shorter time horizons (4-24 hours) for higher accuracy
    
    FULL DOCUMENTATION:
    Use 'crypto-agent-forecaster COMMAND --help' for detailed command documentation.
    """
    success = handler.handle_help()
    if not success:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
