import pytest
import os
from unittest.mock import patch
from vcare_ai.config import ModelConfig, ModelProvider

class TestModelConfig:
    def test_default_config(self):
        """Test default config initialization."""
        with patch.dict(os.environ, {}, clear=True):
            config = ModelConfig()
            assert config.model_provider == ModelProvider.CLAUDE
            assert config.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
            assert config.region == "us-east-1"
            assert config.max_tokens == 2048
            assert config.temperature == 0.5

    def test_config_from_env_vars(self):
        """Test config initialization from environment variables."""
        test_env = {
            "MODEL_PROVIDER": "meta",
            "MODEL_ID": "meta.llama2-70b-chat-v1",
            "AWS_REGION": "us-west-2",
            "MAX_TOKENS": "4096",
            "TEMPERATURE": "0.8"
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            config = ModelConfig()
            assert config.model_provider == ModelProvider.LLAMA
            assert config.model_id == "meta.llama2-70b-chat-v1"
            assert config.region == "us-west-2"
            assert config.max_tokens == 4096
            assert config.temperature == 0.8

    def test_invalid_max_tokens(self):
        """Test handling of invalid MAX_TOKENS."""
        with patch.dict(os.environ, {"MAX_TOKENS": "not_a_number"}, clear=True):
            with pytest.raises(ValueError):
                ModelConfig()

    def test_invalid_temperature(self):
        """Test handling of invalid TEMPERATURE."""
        with patch.dict(os.environ, {"TEMPERATURE": "not_a_number"}, clear=True):
            with pytest.raises(ValueError):
                ModelConfig() 