"""
Configuration management for CryptoAgentForecaster.

This module provides a unified interface to all configuration classes while maintaining
backward compatibility with existing code.
"""

from typing import Dict, Any, Optional

from .core.config_base import BaseConfig
from .core.agent_config import AgentConfig  
from .core.analysis_config import AnalysisConfig
from .core.validation import ConfigValidator


class Config(BaseConfig):
    """
    Unified configuration class for CryptoAgentForecaster.
    
    This class inherits from BaseConfig and provides access to all configuration
    modules while maintaining backward compatibility.
    """
    
    # Include agent configurations for backward compatibility
    LLM_AGENT_CONFIGS = AgentConfig.LLM_AGENT_CONFIGS
    CREW_SETTINGS = AgentConfig.CREW_SETTINGS
    
    # Include analysis configurations for backward compatibility  
    TA_INDICATORS = AnalysisConfig.TA_INDICATORS
    SENTIMENT_CONFIG = AnalysisConfig.SENTIMENT_CONFIG
    
    @classmethod
    def get_agent_llm_config(cls, agent_type: str) -> Dict[str, Any]:
        """Get LLM configuration for a specific agent type."""
        return AgentConfig.get_agent_llm_config(agent_type)
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        return cls.validate_environment()
    
    @classmethod
    def debug_environment(cls) -> Dict[str, Any]:
        """Debug environment configuration for troubleshooting."""
        return ConfigValidator.debug_environment()
    
    @classmethod 
    def validate_llm_config(cls, agent_type: str) -> bool:
        """Validate LLM configuration for a specific agent type."""
        return AgentConfig.validate_llm_config(agent_type) 