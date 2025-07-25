import pytest
import json
from unittest.mock import patch, MagicMock, call
from io import BytesIO

from vcare_ai.client import BedrockClient, BedrockClientError
from vcare_ai.config import ModelProvider

class TestBedrockClient:
    def test_init_with_default_config(self):
        """Test client initialization with default config."""
        with patch("boto3.client") as mock_boto3:
            client = BedrockClient()
            assert client.config is not None
            mock_boto3.assert_called_once_with("bedrock-runtime", 
                                              region_name=client.config.region)

    def test_init_with_custom_config(self, test_config):
        """Test client initialization with custom config."""
        with patch("boto3.client") as mock_boto3:
            client = BedrockClient(custom_config=test_config)
            assert client.config == test_config
            mock_boto3.assert_called_once_with("bedrock-runtime", 
                                              region_name=test_config.region)

    def test_format_prompt_claude(self, test_config):
        """Test prompt formatting for Claude models."""
        test_config.model_provider = ModelProvider.CLAUDE
        client = BedrockClient(custom_config=test_config)
        
        prompt = "Hello, world!"
        formatted = client._format_prompt(prompt)
        
        if "claude-3" in test_config.model_id:
            # Claude 3 uses messages API and 'max_tokens'
            assert "messages" in formatted
            assert formatted["messages"][0]["content"] == prompt
            assert formatted["anthropic_version"] == "bedrock-2023-05-31"
            assert formatted["max_tokens"] == test_config.max_tokens
        else:
            # Claude 2 uses prompt key and 'max_tokens_to_sample'
            assert "prompt" in formatted
            assert prompt in formatted["prompt"]
            assert formatted["max_tokens_to_sample"] == test_config.max_tokens
        assert formatted["temperature"] == test_config.temperature

    def test_format_prompt_llama(self, test_config):
        """Test prompt formatting for Llama models."""
        test_config.model_provider = ModelProvider.LLAMA
        client = BedrockClient(custom_config=test_config)
        
        prompt = "Hello, world!"
        formatted = client._format_prompt(prompt)
        
        assert "prompt" in formatted
        assert formatted["prompt"] == "Hello, world!"
        assert formatted["max_gen_len"] == test_config.max_tokens
        assert formatted["temperature"] == test_config.temperature

    def test_format_prompt_unsupported_provider(self, test_config):
        """Test prompt formatting for unsupported provider."""
        test_config.model_provider = "unsupported"
        client = BedrockClient(custom_config=test_config)
        
        with pytest.raises(BedrockClientError):
            client._format_prompt("Hello, world!")

    def test_invoke_successful(self, boto3_bedrock_client, test_config):
        """Test successful model invocation."""
        # Setup for Claude 3
        test_config.model_provider = ModelProvider.CLAUDE
        test_config.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"  # Claude 3

        claude_response = {
            "content": [
                {"type": "text", "text": "Sample response from Claude model"}
            ]
        }
        mock_response = {
            "body": BytesIO(json.dumps(claude_response).encode())
        }
        boto3_bedrock_client.invoke_model.return_value = mock_response

        client = BedrockClient(custom_config=test_config)
        prompt = "Test prompt"

        # Act
        result = client.invoke(prompt)

        # Assert
        boto3_bedrock_client.invoke_model.assert_called_once()
        assert "text" in result
        assert result["text"] == "Sample response from Claude model"

    def test_invoke_error(self, boto3_bedrock_client):
        """Test error handling during invocation."""
        # Setup
        boto3_bedrock_client.invoke_model.side_effect = Exception("API Error")
        client = BedrockClient()
        
        # Act & Assert
        with pytest.raises(BedrockClientError):
            client.invoke("Test prompt")

    

    def test_parse_response_llama(self, test_config, llama_response):
        """Test response parsing for Llama."""
        test_config.model_provider = ModelProvider.LLAMA
        client = BedrockClient(custom_config=test_config)
        
        result = client._parse_response(llama_response)
        assert "text" in result
        assert result["text"] == llama_response["generation"]

    def test_parse_response_claude(self, test_config):
        """Test response parsing for Claude 3."""
        test_config.model_provider = ModelProvider.CLAUDE
        test_config.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"  # Claude 3
        client = BedrockClient(custom_config=test_config)

        claude3_response = {
            "content": [
                {"type": "text", "text": "Sample response from Claude 3 model"}
            ]
        }
        result = client._parse_response(claude3_response)
        assert "text" in result
        assert result["text"] == "Sample response from Claude 3 model" 