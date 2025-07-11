"""
Command handlers for the CLI application.
"""

import logging
import traceback
from typing import Optional

from .constants import ERROR_MESSAGES, SUCCESS_MESSAGES, DEFAULT_HORIZON
from .output import OutputManager
from .validation import Validator
from ..config import Config
from ..llm_factory import LLMFactory
from ..agents import CryptoForecastingCrew

logger = logging.getLogger(__name__)


class CommandHandler:
    """Handles all CLI command operations."""
    
    def __init__(self):
        self.output = OutputManager()
        self.validator = Validator(self.output)
    
    def handle_forecast(self, crypto: str, horizon: str = DEFAULT_HORIZON, 
                       provider: Optional[str] = None, model: Optional[str] = None,
                       verbose: bool = False, yes: bool = False) -> bool:
        """
        Handle forecast command.
        
        Args:
            crypto: Cryptocurrency name
            horizon: Forecast time horizon
            provider: LLM provider (optional)
            model: Model name (optional)
            verbose: Enable verbose output
            yes: Skip confirmation prompt
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.output.display_banner()
        
        # Validate configuration
        if not self.validator.validate_configuration():
            return False
        
        # Validate inputs
        if not self.validator.validate_forecast_inputs(crypto, horizon, provider, model):
            return False
        
        # Validate provider if specified
        if not self.validator.validate_provider(provider):
            return False
        
        # Update configuration if provider/model specified
        if provider:
            Config.DEFAULT_LLM_PROVIDER = provider
        if model:
            Config.DEFAULT_LLM_MODEL = model
        
        # Display configuration
        self.output.display_forecast_configuration(
            crypto, horizon, Config.DEFAULT_LLM_PROVIDER, Config.DEFAULT_LLM_MODEL
        )
        
        # Confirm before proceeding
        if not yes and not self.validator.get_user_confirmation("\nProceed with forecast?", default=True):
            self.output.print("Forecast cancelled.")
            return False
        
        try:
            # Initialize the forecasting crew
            crew = CryptoForecastingCrew(verbose=verbose)
            
            # Run the forecast
            results = crew.run_forecast(crypto, horizon)
            
            # Check results
            if "error" not in results:
                self.output.display_success(SUCCESS_MESSAGES["forecast_completed"])
                return True
            else:
                self.output.display_error(f"{ERROR_MESSAGES['forecast_failed'].format(results['error'])}")
                return False
                
        except Exception as e:
            self.output.display_error(f"Unexpected error: {str(e)}")
            if verbose:
                self.output.print(traceback.format_exc())
            logger.error(f"Forecast command failed: {str(e)}")
            return False
    
    def handle_config(self) -> bool:
        """
        Handle config command.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.output.display_banner()
        
        try:
            is_valid = self.output.display_configuration_status()
            self.output.display_setup_instructions()
            
            if is_valid:
                logger.info("Configuration command completed successfully")
            else:
                logger.warning("Configuration issues detected")
            
            return is_valid
            
        except Exception as e:
            self.output.display_error(f"Configuration check failed: {str(e)}")
            logger.error(f"Config command failed: {str(e)}")
            return False
    
    def handle_debug(self) -> bool:
        """
        Handle debug command.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.output.display_banner()
        
        try:
            # Get debug information
            debug_info = self.validator.validate_debug_environment()
            
            # Display debug information
            self.output.display_debug_info(debug_info)
            
            # Display agent configuration status
            self.output.display_agent_configuration_status()
            
            # Display available providers
            self.output.display_available_providers()
            
            # Display troubleshooting tips
            self.output.display_troubleshooting_tips()
            
            logger.info("Debug command completed")
            return True
            
        except Exception as e:
            self.output.display_error(ERROR_MESSAGES["debug_failed"].format(str(e)))
            self.output.print(traceback.format_exc())
            logger.error(f"Debug command failed: {str(e)}")
            return False
    
    def handle_test(self, crypto: str = "bitcoin", quick: bool = False) -> bool:
        """
        Handle test command.
        
        Args:
            crypto: Cryptocurrency to test with
            quick: Whether to run quick test
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.output.display_banner()
        
        if not self.validator.validate_configuration():
            return False
        
        # Validate and normalize test parameters
        crypto, days = self.validator.validate_test_parameters(crypto, quick)
        
        self.output.print(f"\nTesting system with {crypto.upper()}...")
        
        try:
            # Import tools for testing
            from ..tools.coingecko_tool import CoinGeckoTool
            from ..tools.fourchan_tool import FourChanBizTool
            from ..tools.technical_analysis_tool import TechnicalAnalysisTool
            
            # Test CoinGecko tool
            self.output.print("1. Testing CoinGecko API...")
            coingecko_tool = CoinGeckoTool()
            market_data = coingecko_tool._run(query=f"{crypto} ohlcv {days} days")
            self.output.print("   ✅ CoinGecko API working")
            
            # Test 4chan tool (if not quick mode)
            if not quick:
                self.output.print("2. Testing 4chan /biz/ API...")
                fourchan_tool = FourChanBizTool()
                biz_data = fourchan_tool._run(keywords=[crypto, "crypto"], max_threads=2, max_posts_per_thread=5)
                self.output.print("   ✅ 4chan API working")
            else:
                self.output.print("2. Skipping 4chan test (quick mode)")
            
            # Test technical analysis
            self.output.print("3. Testing technical analysis...")
            tech_tool = TechnicalAnalysisTool()
            forecast_horizon = "7 days" if quick else "30 days"
            tech_analysis = tech_tool._run(crypto_name=crypto, forecast_horizon=forecast_horizon)
            self.output.print("   ✅ Technical analysis working")
            
            self.output.display_success(SUCCESS_MESSAGES["all_tests_passed"])
            logger.info(f"Test command completed successfully for {crypto}")
            return True
            
        except Exception as e:
            self.output.display_error(ERROR_MESSAGES["test_failed"].format(str(e)))
            logger.error(f"Test command failed: {str(e)}")
            return False
    
    def handle_list_cryptos(self) -> bool:
        """
        Handle list-cryptos command.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.output.display_crypto_list()
            logger.info("List cryptos command completed")
            return True
            
        except Exception as e:
            self.output.display_error(f"Failed to display crypto list: {str(e)}")
            logger.error(f"List cryptos command failed: {str(e)}")
            return False
    
    def handle_models(self) -> bool:
        """
        Handle models command.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.output.display_banner()
        
        if not self.validator.validate_configuration():
            return False
        
        try:
            self.output.display_models_info()
            logger.info("Models command completed")
            return True
            
        except Exception as e:
            self.output.display_error(f"Failed to display models info: {str(e)}")
            logger.error(f"Models command failed: {str(e)}")
            return False
    
    def handle_help(self) -> bool:
        """
        Handle help command.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.output.display_banner()
            self.output.display_help_overview()
            logger.info("Help command completed")
            return True
            
        except Exception as e:
            self.output.display_error(f"Failed to display help: {str(e)}")
            logger.error(f"Help command failed: {str(e)}")
            return False 