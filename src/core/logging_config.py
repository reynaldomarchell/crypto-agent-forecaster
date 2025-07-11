"""
Logging configuration for CryptoAgentForecaster.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class LoggingConfig:
    """Centralized logging configuration."""
    
    # Default logging configuration
    DEFAULT_CONFIG = {
        "level": logging.INFO,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "file_level": logging.DEBUG,
        "console_level": logging.INFO,
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5
    }
    
    @classmethod
    def setup_logging(cls, 
                     log_level: str = "INFO",
                     log_to_file: bool = True,
                     log_dir: Optional[Path] = None,
                     verbose: bool = False) -> logging.Logger:
        """
        Set up application-wide logging configuration.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to write logs to file
            log_dir: Directory for log files
            verbose: Enable verbose logging
            
        Returns:
            Configured root logger
        """
        # Convert string level to logging constant
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Create custom formatter
        formatter = cls._create_formatter(verbose)
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add console handler
        console_handler = cls._create_console_handler(numeric_level, formatter, verbose)
        root_logger.addHandler(console_handler)
        
        # Add file handler if requested
        if log_to_file:
            file_handler = cls._create_file_handler(log_dir, formatter)
            if file_handler:
                root_logger.addHandler(file_handler)
        
        # Configure third-party loggers
        cls._configure_third_party_loggers()
        
        return root_logger
    
    @classmethod
    def _create_formatter(cls, verbose: bool = False) -> logging.Formatter:
        """Create logging formatter."""
        if verbose:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
            )
        else:
            format_string = cls.DEFAULT_CONFIG["format"]
        
        return logging.Formatter(
            format_string,
            datefmt=cls.DEFAULT_CONFIG["date_format"]
        )
    
    @classmethod
    def _create_console_handler(cls, level: int, formatter: logging.Formatter, 
                               verbose: bool = False) -> logging.StreamHandler:
        """Create console handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        
        # Add color formatting if available
        try:
            import colorlog
            if not verbose:  # Only use colors for non-verbose mode
                color_formatter = colorlog.ColoredFormatter(
                    "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt=cls.DEFAULT_CONFIG["date_format"],
                    log_colors={
                        'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'red,bg_white',
                    }
                )
                handler.setFormatter(color_formatter)
        except ImportError:
            pass  # colorlog not available, use default formatter
        
        return handler
    
    @classmethod
    def _create_file_handler(cls, log_dir: Optional[Path], 
                            formatter: logging.Formatter) -> Optional[logging.Handler]:
        """Create file handler with rotation."""
        try:
            from logging.handlers import RotatingFileHandler
            
            # Set up log directory
            if log_dir is None:
                log_dir = Path("logs")
            
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log file path
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = log_dir / f"crypto_agent_forecaster_{timestamp}.log"
            
            # Create rotating file handler
            handler = RotatingFileHandler(
                log_file,
                maxBytes=cls.DEFAULT_CONFIG["max_file_size"],
                backupCount=cls.DEFAULT_CONFIG["backup_count"]
            )
            
            handler.setLevel(cls.DEFAULT_CONFIG["file_level"])
            handler.setFormatter(formatter)
            
            return handler
            
        except Exception as e:
            # Fallback: create simple file handler
            try:
                if log_dir is None:
                    log_dir = Path("logs")
                log_dir.mkdir(parents=True, exist_ok=True)
                
                log_file = log_dir / "crypto_agent_forecaster.log"
                handler = logging.FileHandler(log_file)
                handler.setLevel(cls.DEFAULT_CONFIG["file_level"])
                handler.setFormatter(formatter)
                
                return handler
                
            except Exception:
                print(f"Warning: Could not create file handler: {e}")
                return None
    
    @classmethod
    def _configure_third_party_loggers(cls):
        """Configure third-party library loggers to reduce noise."""
        # Reduce noise from common libraries
        noisy_loggers = [
            "urllib3.connectionpool",
            "requests.packages.urllib3.connectionpool", 
            "matplotlib",
            "PIL",
            "crewai",
            "langchain"
        ]
        
        for logger_name in noisy_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger with the given name.
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)
    
    @classmethod
    def set_level(cls, level: str):
        """
        Set logging level for all handlers.
        
        Args:
            level: Logging level string
        """
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return LoggingConfig.get_logger(name) 