# CryptoAgentForecaster

**Multimodal AI-Driven Cryptocurrency Price Forecasting System**

An advanced cryptocurrency forecasting tool that leverages hosted Large Language Models (LLMs), multi-agent architecture, and novel data sources including 4chan's /biz/ board for sentiment analysis.

## Features

- **Multi-Agent Architecture**: Specialized AI agents for data collection, sentiment analysis, technical analysis, and forecasting
- **Comprehensive Data Sources**:
  - CoinGecko API for market data (OHLCV, volume, market cap)
  - 4chan /biz/ board for raw sentiment analysis
  - Technical indicators and candlestick patterns
- **Hosted LLM Integration**: Support for OpenAI GPT, Anthropic Claude, and Google Gemini
- **Advanced Analysis**:
  - FUD (Fear, Uncertainty, Doubt) detection
  - Shill detection and manipulation analysis
  - Technical pattern recognition
  - Multimodal signal fusion
- **User-Friendly CLI**: Rich terminal interface with beautiful output formatting
- **Automatic Result Management**:
  - Auto-saves all results to organized folders
  - Technical analysis charts saved as PNG files
  - Complete run logs with sanitized output
  - Clean logging (no verbose JSON or base64 spam)

## Quick Start Guide for Newcomers

### 1. Prerequisites

- **Python 3.13+** (required)
- **uv** package manager (recommended) or pip
- At least one LLM API key (OpenAI, Anthropic, or Google)

### 2. Installation

#### Option A: Using uv (Recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yoshikazuuu/crypto-agent-forecaster.git
cd crypto-agent-forecaster

# Install all dependencies and create virtual environment
uv sync
```

#### Option B: Using pip

```bash
# Clone the repository
git clone https://github.com/yoshikazuuu/crypto-agent-forecaster.git
cd crypto-agent-forecaster

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 3. Setup API Keys

Copy the environment template and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` file with your API keys:

```bash
# At least one LLM provider is required
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here  
GOOGLE_API_KEY=your_google_api_key_here

# Optional but recommended for higher rate limits
COINGECKO_API_KEY=your_coingecko_api_key_here

# LLM Configuration (optional - has defaults)
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4o-mini
```

### 4. Get API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Google**: https://aistudio.google.com/app/apikey
- **CoinGecko** (optional): https://www.coingecko.com/en/api/pricing

### 5. Verify Setup

```bash
# Check configuration
python main.py config

# Run quick test (recommended first step)
python main.py test --quick

# Optional: Run full test
python main.py test
```

### 6. Your First Forecast

```bash
# Basic forecast
python main.py forecast bitcoin

# With detailed progress tracking
python main.py forecast bitcoin --verbose
```

## Available Commands

### Core Commands

#### `forecast` - Generate AI Forecasts

Generate comprehensive cryptocurrency price forecasts using a 4-agent system:

```bash
python main.py forecast CRYPTO_NAME [OPTIONS]
```

**Workflow Process:**
1. **Market Data Agent** collects 30 days of OHLCV data + current market stats
2. **Sentiment Agent** analyzes social sentiment from 4chan /biz/ discussions  
3. **Technical Agent** performs TA analysis and generates interactive charts
4. **Forecasting Agent** synthesizes all data into final prediction

**Options:**
- `--horizon, -h`: Forecast time horizon (default: "24 hours")
  - Examples: "24 hours", "3 days", "1 week"
- `--provider, -p`: LLM provider ('openai', 'anthropic', 'google')
- `--model, -m`: Specific model to use
- `--verbose, -v`: Enable detailed execution tracking
- `--yes, -y`: Skip confirmation prompt

**Examples:**
```bash
# Basic forecast
python main.py forecast bitcoin

# Verbose mode with detailed tracking
python main.py forecast ethereum --verbose

# Custom time horizon
python main.py forecast solana --horizon "3 days"

# Specify LLM provider and model
python main.py forecast cardano --provider anthropic --model claude-3-5-sonnet-20241022

# Fully automated
python main.py forecast bitcoin --horizon "1 week" --provider openai --verbose --yes
```

**Outputs Generated:**
- Forecast direction (UP or DOWN) with confidence score
- Technical analysis charts saved as PNG files
- Complete execution logs and agent interactions
- Professional markdown report with embedded charts
- Structured JSON data for programmatic access

#### `backtest` - Research & Analysis

Run comprehensive backtesting experiments for thesis research:

```bash
python main.py backtest [CRYPTO] [OPTIONS]
```

**Prediction Methods Compared:**
1. **Full Agentic**: Complete multi-agent system
2. **Image Only**: One-shot LLM analysis of technical charts only
3. **Sentiment Only**: One-shot LLM analysis of 4chan /biz/ sentiment only

**Options:**
- `--start-date, -s`: Start date (YYYY-MM-DD format, default: 1 year ago)
- `--end-date, -e`: End date (YYYY-MM-DD format, default: yesterday)
- `--methods, -m`: Methods to test ('all', 'agentic', 'image', 'sentiment', or comma-separated)
- `--data-dir, -d`: Directory for results (default: thesis_data)
- `--resume/--no-resume`: Resume existing backtest (default: resume)
- `--quick-test, -q`: Run quick test with only 7 days of data

**Examples:**
```bash
# Full year backtest for Bitcoin
python main.py backtest bitcoin

# Quick 7-day test
python main.py backtest bitcoin --quick-test

# Specific date range
python main.py backtest ethereum --start-date 2024-01-01 --end-date 2024-06-30

# Test specific methods only
python main.py backtest solana --methods "agentic,image"
```

### Utility Commands

#### `config` - Check Configuration

Display current configuration and setup status:

```bash
python main.py config
```

Shows:
- LLM provider status (configured/not configured)
- Available models for each provider
- CoinGecko API status
- Setup instructions

#### `test` - Validate System

Test system components and validate configuration:

```bash
python main.py test [OPTIONS]
```

**Options:**
- `--crypto`: Cryptocurrency to test with (default: bitcoin)
- `--quick`: Quick test mode (fewer data points, faster execution)

**Tests Performed:**
1. CoinGecko API connectivity and data retrieval
2. 4chan /biz/ API access and sentiment data collection
3. Technical analysis tool functionality and chart generation
4. Data processing and integration workflows

**Examples:**
```bash
# Standard test
python main.py test

# Quick test for faster validation
python main.py test --quick

# Test with specific cryptocurrency
python main.py test --crypto ethereum --quick
```

#### `list-cryptos` - Available Cryptocurrencies

List popular cryptocurrencies available for analysis:

```bash
python main.py list-cryptos
```

Shows a table with:
- CoinGecko ID (use this for commands)
- Full name
- Symbol

Popular options include: bitcoin, ethereum, solana, cardano, polkadot, chainlink, avalanche-2

#### `models` - LLM Information

Display available LLM models, specifications, and recommendations:

```bash
python main.py models
```

Shows:
- Model specifications for each provider
- Cost per 1K tokens (input/output)
- Task-specific recommendations
- Usage tips

#### `debug` - Troubleshooting

Debug environment and configuration issues:

```bash
python main.py debug
```

#### `help` - Quick Reference

Show quick usage examples and next steps:

```bash
python main.py help
```

## Output Format

### Forecast Results

The system provides structured forecasts including:

- **Direction**: UP or DOWN
- **Confidence**: HIGH/MEDIUM/LOW  
- **Detailed Explanation**: Reasoning and key factors
- **Technical Analysis**: Indicators and patterns
- **Sentiment Analysis**: Market mood and narratives
- **Risk Considerations**: Caveats and uncertainties

Example output:
```
Forecast Results for BITCOIN
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
║ Metric           ║ Value                         ║
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Direction        │ UP (Bullish)                  │
│ Confidence       │ MEDIUM                        │
│ Forecast Horizon │ 24 hours                      │
│ Timestamp        │ 2024-01-15T10:30:00           │
│ Charts Generated │ 1                             │
└──────────────────┴───────────────────────────────┘

Analysis & Reasoning
Technical analysis shows bullish momentum with RSI at 45 
and MACD crossing above signal line. 4chan sentiment 
reveals moderate optimism with limited FUD detection...
```

### Results Management

Every forecast run automatically creates a dedicated folder in `results/` with:

#### Folder Structure
```
results/
└── bitcoin_20241215_143052/
    ├── README.md                # Summary with key metrics
    ├── forecast_results.json    # Complete structured data
    ├── run_logs.txt            # Sanitized execution logs
    └── charts/
        └── technical_analysis_chart.png  # Generated charts
```

#### Features
- **Clean Logging**: No verbose JSON or base64 spam in console
- **Chart Generation**: Technical analysis charts saved as PNG
- **Structured Data**: Complete results in JSON format
- **Run Logs**: Full execution history with timestamps
- **Summary**: Markdown summary with embedded charts

## LLM Configuration

### Available Providers

1. **OpenAI**
   - Models: gpt-4o-mini, gpt-4o
   - Best for: General analysis, cost-effective
   
2. **Anthropic**  
   - Models: claude-3-5-sonnet-20241022
   - Best for: Nuanced sentiment analysis
   
3. **Google**
   - Models: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-lite
   - Best for: Large context windows

### Model Selection

Use the `models` command to see current recommendations:

```bash
python main.py models
```

### Override Defaults

```bash
# Use Anthropic Claude
python main.py forecast bitcoin --provider anthropic --model claude-3-5-sonnet-20241022

# Use Google Gemini
python main.py forecast ethereum --provider google --model gemini-1.5-pro

# Use OpenAI GPT-4
python main.py forecast solana --provider openai --model gpt-4o
```

## Development

### Project Structure

```
crypto-agent-forecaster/
├── src/
│   ├── agents/          # CrewAI agent definitions
│   ├── tools/           # LangChain tools for data fetching
│   ├── prompts/         # LLM prompt templates
│   ├── cli/             # CLI commands and interface
│   ├── core/            # Configuration and error handling
│   ├── backtesting/     # Backtesting framework
│   └── utils.py         # Utility functions
├── main.py              # CLI application entry point
├── pyproject.toml       # Dependencies and project config
├── .env.example         # Environment template
└── README.md           # This file
```

### Dependencies

Key dependencies (see `pyproject.toml` for complete list):
- `crewai>=0.121.1` - Multi-agent framework
- `typer>=0.16.0` - CLI framework
- `rich>=13.9.4` - Terminal formatting
- `pandas>=2.2.3` - Data manipulation
- `matplotlib>=3.7.0` - Chart generation
- `ta>=0.11.0` - Technical analysis

### Running with uv

If you installed with `uv`, you can run commands directly:

```bash
# Run in virtual environment
uv run python main.py forecast bitcoin

# Or activate environment first
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python main.py forecast bitcoin
```

## Troubleshooting

### Common Issues

1. **API Key Issues**
   ```bash
   python main.py config  # Check configuration
   ```

2. **Component Testing**
   ```bash
   python main.py test --quick  # Validate system
   ```

3. **Verbose Mode for Debugging**
   ```bash
   python main.py forecast bitcoin --verbose
   ```

4. **Rate Limiting**
   - Wait a few minutes between requests
   - Use CoinGecko API key for higher limits

### Error Messages

- **"❌ Not configured"**: Add API key to `.env` file
- **Rate limit errors**: Wait or add CoinGecko API key
- **Network errors**: Check internet connection and firewall

## Important Considerations

### Risks & Limitations

- **Not Financial Advice**: This tool is for research and educational purposes only
- **Market Volatility**: Cryptocurrency markets are highly volatile and unpredictable  
- **Data Quality**: 4chan data is noisy and may contain manipulation attempts
- **LLM Limitations**: AI models can hallucinate or misinterpret data
- **API Dependencies**: System relies on external APIs that may have downtime

### Ethical Use

- Respect API rate limits and terms of service
- Use 4chan data responsibly and ethically
- Do not use for market manipulation
- Always conduct additional research before making financial decisions

## Related Background

This project builds upon and extends concepts from recent academic research in multi-agent systems for cryptocurrency investment:

- **"LLM-Powered Multi-Agent System for Automated Crypto Portfolio Management"** (Luo et al., 2024) [[arXiv:2501.00826]](https://arxiv.org/pdf/2501.00826)
  - Demonstrates the effectiveness of multi-agent architectures for cryptocurrency portfolio management
  - Validates the use of specialized agents for different aspects of analysis (market data, sentiment, technical analysis)
  - Shows how intrateam and interteam collaboration mechanisms enhance prediction accuracy
  - Provides empirical evidence that multi-agent systems outperform single-agent models

Our implementation extends these concepts by:

- Integrating 4chan /biz/ as a unique sentiment data source
- Providing a user-friendly CLI interface for real-time forecasting
- Implementing multimodal chart analysis using AI vision capabilities
- Offering support for multiple LLM providers (OpenAI, Anthropic, Google)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable  
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CoinGecko API for market data
- 4chan for providing public API access
- OpenAI, Anthropic, and Google for LLM APIs
- CrewAI and LangChain communities
- The broader cryptocurrency and AI research communities

---

**Disclaimer**: This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Always do your own research and never invest more than you can afford to lose.
