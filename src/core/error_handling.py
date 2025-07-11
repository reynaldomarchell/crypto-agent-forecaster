"""
Error handling utilities and custom exceptions for CryptoAgentForecaster.
"""

import logging
import traceback
from typing import Any, Dict, Optional, Callable, TypeVar, ParamSpec
from functools import wraps
from enum import Enum

from .logging_config import get_logger

logger = get_logger(__name__)

# Type variables for decorators
P = ParamSpec('P')
T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CryptoAgentError(Exception):
    """Base exception for CryptoAgentForecaster."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.severity = severity
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "details": self.details
        }


class ConfigurationError(CryptoAgentError):
    """Raised when there are configuration issues."""
    
    def __init__(self, message: str, missing_config: Optional[str] = None):
        details = {"missing_config": missing_config} if missing_config else {}
        super().__init__(message, ErrorSeverity.HIGH, details)


class APIError(CryptoAgentError):
    """Raised when API calls fail."""
    
    def __init__(self, message: str, api_name: str, status_code: Optional[int] = None,
                 response_data: Optional[Dict[str, Any]] = None):
        details = {
            "api_name": api_name,
            "status_code": status_code,
            "response_data": response_data
        }
        super().__init__(message, ErrorSeverity.MEDIUM, details)


class DataProcessingError(CryptoAgentError):
    """Raised when data processing fails."""
    
    def __init__(self, message: str, data_type: Optional[str] = None, 
                 processing_stage: Optional[str] = None):
        details = {
            "data_type": data_type,
            "processing_stage": processing_stage
        }
        super().__init__(message, ErrorSeverity.MEDIUM, details)


class AgentError(CryptoAgentError):
    """Raised when agent operations fail."""
    
    def __init__(self, message: str, agent_type: Optional[str] = None,
                 task_name: Optional[str] = None):
        details = {
            "agent_type": agent_type,
            "task_name": task_name
        }
        super().__init__(message, ErrorSeverity.HIGH, details)


class ValidationError(CryptoAgentError):
    """Raised when validation fails."""
    
    def __init__(self, message: str, field_name: Optional[str] = None,
                 field_value: Optional[Any] = None):
        details = {
            "field_name": field_name,
            "field_value": str(field_value) if field_value is not None else None
        }
        super().__init__(message, ErrorSeverity.MEDIUM, details)


class ErrorHandler:
    """Centralized error handling utilities."""
    
    @staticmethod
    def handle_exception(exc: Exception, context: Optional[str] = None,
                        reraise: bool = True) -> Dict[str, Any]:
        """
        Handle and log exceptions in a standardized way.
        
        Args:
            exc: Exception to handle
            context: Additional context about where the error occurred
            reraise: Whether to reraise the exception after handling
            
        Returns:
            Dict containing error information
        """
        error_info = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "context": context,
            "traceback": traceback.format_exc()
        }
        
        # Handle custom exceptions
        if isinstance(exc, CryptoAgentError):
            error_info.update(exc.to_dict())
            
            # Log based on severity
            if exc.severity == ErrorSeverity.CRITICAL:
                logger.critical(f"Critical error in {context}: {exc.message}", 
                              extra={"error_details": exc.details})
            elif exc.severity == ErrorSeverity.HIGH:
                logger.error(f"Error in {context}: {exc.message}", 
                           extra={"error_details": exc.details})
            elif exc.severity == ErrorSeverity.MEDIUM:
                logger.warning(f"Warning in {context}: {exc.message}", 
                             extra={"error_details": exc.details})
            else:
                logger.info(f"Info in {context}: {exc.message}", 
                          extra={"error_details": exc.details})
        else:
            # Handle standard exceptions
            logger.error(f"Unhandled exception in {context}: {str(exc)}", 
                        exc_info=True)
        
        if reraise:
            raise exc
        
        return error_info
    
    @staticmethod
    def wrap_api_call(api_name: str):
        """
        Decorator to wrap API calls with error handling.
        
        Args:
            api_name: Name of the API for error context
        """
        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if isinstance(e, APIError):
                        raise
                    else:
                        raise APIError(
                            f"API call to {api_name} failed: {str(e)}",
                            api_name=api_name
                        ) from e
            return wrapper
        return decorator
    
    @staticmethod
    def safe_execute(func: Callable[[], T], 
                    default: Optional[T] = None,
                    context: Optional[str] = None) -> Optional[T]:
        """
        Safely execute a function with error handling.
        
        Args:
            func: Function to execute
            default: Default value to return on error
            context: Context for error logging
            
        Returns:
            Function result or default value
        """
        try:
            return func()
        except Exception as e:
            ErrorHandler.handle_exception(e, context, reraise=False)
            return default
    
    @staticmethod
    def validate_and_raise(condition: bool, error_type: type, 
                          message: str, **kwargs):
        """
        Validate a condition and raise an error if it fails.
        
        Args:
            condition: Condition to validate
            error_type: Type of error to raise
            message: Error message
            **kwargs: Additional arguments for the error
        """
        if not condition:
            raise error_type(message, **kwargs)


def retry_on_error(max_attempts: int = 3, delay: float = 1.0, 
                  backoff_factor: float = 2.0):
    """
    Decorator to retry functions on error.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between attempts (seconds)
        backoff_factor: Factor to multiply delay by for each retry
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            import time
            
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Last attempt, don't retry
                        break
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
            # If we get here, all attempts failed
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


def log_exceptions(context: Optional[str] = None):
    """
    Decorator to automatically log exceptions.
    
    Args:
        context: Additional context for logging
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = context or func.__name__
                ErrorHandler.handle_exception(e, error_context)
                
        return wrapper
    return decorator


class ErrorReporter:
    """Utility for collecting and reporting errors."""
    
    def __init__(self):
        self.errors: list = []
    
    def add_error(self, error: Exception, context: Optional[str] = None):
        """Add an error to the collection."""
        error_info = ErrorHandler.handle_exception(error, context, reraise=False)
        self.errors.append(error_info)
    
    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self.errors) > 0
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected errors."""
        if not self.errors:
            return {"total_errors": 0, "errors": []}
        
        return {
            "total_errors": len(self.errors),
            "errors": self.errors,
            "error_types": list(set(error["type"] for error in self.errors)),
            "has_critical": any(
                error.get("severity") == ErrorSeverity.CRITICAL.value 
                for error in self.errors
            )
        }
    
    def clear(self):
        """Clear all collected errors."""
        self.errors.clear() 