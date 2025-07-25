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
        self.model_provider = os.getenv("MODEL_PROVIDER", ModelProvider.LLAMA)
        self.model_id = os.getenv("MODEL_ID", "meta.llama3-2-90b-instruct-v1:0")
        
        # Change this line to a supported region
        self.region = os.getenv("AWS_REGION", "us-east-1") 
        
        try:
            self.max_tokens = int(os.getenv("MAX_TOKENS", 8000))
        except ValueError:
            raise ValueError("MAX_TOKENS must be a valid integer")
            
        try:
            self.temperature = float(os.getenv("TEMP", 0))
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
