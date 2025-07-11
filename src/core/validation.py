"""
Configuration validation and debugging utilities.
"""

import os
import logging
from typing import Dict, Any, Optional

from .config_base import BaseConfig
from .agent_config import AgentConfig

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Handles configuration validation and debugging."""
    
    @classmethod
    def validate_complete_configuration(cls) -> bool:
        """
        Validate complete system configuration.
        
        Returns:
            bool: True if all configurations are valid
        """
        try:
            # Validate base environment
            if not BaseConfig.validate_environment():
                logger.error("Base environment validation failed")
                return False
            
            # Validate all agent configurations
            for agent_type in AgentConfig.get_all_agent_types():
                if not AgentConfig.validate_llm_config(agent_type):
                    logger.error(f"Agent {agent_type} configuration validation failed")
                    return False
            
            logger.info("Complete configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    @classmethod
    def debug_environment(cls) -> Dict[str, Any]:
        """
        Get comprehensive environment debug information.
        
        Returns:
            Dict containing debug information
        """
        try:
            debug_info = {
                "environment_loaded": True,
                "api_keys_configured": BaseConfig.get_api_key_status(),
                "env_vars_present": cls._check_env_vars_present(),
                "default_provider": BaseConfig.DEFAULT_LLM_PROVIDER,
                "default_model": BaseConfig.DEFAULT_LLM_MODEL,
                "configured_providers": BaseConfig.get_configured_providers(),
                "agent_configs": cls._get_agent_config_status(),
                "validation_results": cls._get_validation_results()
            }
            
            logger.debug("Environment debug information collected")
            return debug_info
            
        except Exception as e:
            logger.error(f"Failed to collect debug information: {str(e)}")
            return {
                "error": str(e),
                "environment_loaded": False,
                "api_keys_configured": {},
                "env_vars_present": {},
                "default_provider": None,
                "default_model": None
            }
    
    @classmethod
    def _check_env_vars_present(cls) -> Dict[str, bool]:
        """Check if environment variables are present."""
        env_vars = {
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
            "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
            "COINGECKO_API_KEY": bool(os.getenv("COINGECKO_API_KEY")),
            "DEFAULT_LLM_PROVIDER": bool(os.getenv("DEFAULT_LLM_PROVIDER")),
            "DEFAULT_LLM_MODEL": bool(os.getenv("DEFAULT_LLM_MODEL"))
        }
        return env_vars
    
    @classmethod
    def _get_agent_config_status(cls) -> Dict[str, Any]:
        """Get configuration status for all agents."""
        agent_status = {}
        
        for agent_type in AgentConfig.get_all_agent_types():
            try:
                config = AgentConfig.get_agent_llm_config(agent_type)
                is_valid = AgentConfig.validate_llm_config(agent_type)
                
                agent_status[agent_type] = {
                    "config_present": bool(config),
                    "is_valid": is_valid,
                    "preferred_provider": config.get("preferred_provider"),
                    "preferred_model": config.get("preferred_model"),
                    "temperature": config.get("temperature"),
                    "max_tokens": config.get("max_tokens")
                }
                
            except Exception as e:
                agent_status[agent_type] = {
                    "config_present": False,
                    "is_valid": False,
                    "error": str(e)
                }
        
        return agent_status
    
    @classmethod
    def _get_validation_results(cls) -> Dict[str, bool]:
        """Get validation results for different components."""
        results = {
            "base_config": BaseConfig.validate_environment(),
            "has_llm_provider": BaseConfig.has_any_llm_provider(),
            "complete_config": cls.validate_complete_configuration()
        }
        
        # Add agent-specific validation results
        for agent_type in AgentConfig.get_all_agent_types():
            results[f"agent_{agent_type}"] = AgentConfig.validate_llm_config(agent_type)
        
        return results
    
    @classmethod
    def get_configuration_issues(cls) -> list:
        """
        Get a list of configuration issues that need to be addressed.
        
        Returns:
            List of configuration issues
        """
        issues = []
        
        try:
            # Check for LLM providers
            if not BaseConfig.has_any_llm_provider():
                issues.append("No LLM providers configured. Set at least one API key.")
            
            # Check specific provider configurations
            api_status = BaseConfig.get_api_key_status()
            for provider, configured in api_status.items():
                if provider != "coingecko" and not configured:
                    issues.append(f"{provider.upper()} API key not configured")
            
            # Check agent configurations
            for agent_type in AgentConfig.get_all_agent_types():
                if not AgentConfig.validate_llm_config(agent_type):
                    issues.append(f"Agent {agent_type} has invalid configuration")
            
            # Check environment variables
            env_vars = cls._check_env_vars_present()
            if not env_vars.get("DEFAULT_LLM_PROVIDER"):
                issues.append("DEFAULT_LLM_PROVIDER not set in environment")
            
            if not env_vars.get("DEFAULT_LLM_MODEL"):
                issues.append("DEFAULT_LLM_MODEL not set in environment")
            
        except Exception as e:
            issues.append(f"Error checking configuration: {str(e)}")
        
        return issues
    
    @classmethod
    def get_configuration_summary(cls) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dict containing configuration summary
        """
        try:
            summary = {
                "providers_configured": len(BaseConfig.get_configured_providers()),
                "total_providers": 3,  # OpenAI, Anthropic, Google
                "agents_configured": len(AgentConfig.get_all_agent_types()),
                "agents_valid": sum(1 for agent in AgentConfig.get_all_agent_types() 
                                  if AgentConfig.validate_llm_config(agent)),
                "default_provider": BaseConfig.DEFAULT_LLM_PROVIDER,
                "default_model": BaseConfig.DEFAULT_LLM_MODEL,
                "coingecko_configured": bool(BaseConfig.COINGECKO_API_KEY),
                "issues_count": len(cls.get_configuration_issues()),
                "overall_status": "healthy" if cls.validate_complete_configuration() else "needs_attention"
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate configuration summary: {str(e)}")
            return {
                "error": str(e),
                "overall_status": "error"
            } 