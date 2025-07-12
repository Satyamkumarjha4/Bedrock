# src/vcare_ai/config.py
import os
from enum import Enum

class ModelProvider(str, Enum):
    CLAUDE = "anthropic"
    LLAMA = "meta"
    MISTRAL = "mistral"

class ConfigError(Exception):
    """Base exception for configuration errors"""
    pass

class ModelConfig:
    def __init__(self):
        self.model_provider = os.getenv("MODEL_PROVIDER", ModelProvider.CLAUDE)
        self.model_id = os.getenv("MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
        self.region = os.getenv("AWS_REGION", "ap-south-1")
        
        try:
            self.max_tokens = int(os.getenv("MAX_TOKENS", 2048))
        except ValueError:
            raise ValueError("MAX_TOKENS must be a valid integer")
            
        try:
            self.temperature = float(os.getenv("TEMPERATURE", 0))
        except ValueError:
            raise ValueError("TEMPERATURE must be a valid float")
        
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.max_tokens <= 0:
            raise ConfigError("MAX_TOKENS must be greater than 0")
            
        if self.temperature < 0 or self.temperature > 1:
            raise ConfigError("TEMPERATURE must be between 0 and 1")

config = ModelConfig()



# Configurations for the DataBase

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vectordb",
    "user": "postgres",
    "password": "postgres"
}
