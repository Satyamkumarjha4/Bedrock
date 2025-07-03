import pytest
from unittest.mock import MagicMock, patch
import os
import json
from vcare_ai.client import BedrockClient
from vcare_ai.config import config, ModelConfig, ModelProvider

@pytest.fixture
def mock_bedrock_client():
    """Returns a mocked BedrockClient instance."""
    client = MagicMock(spec=BedrockClient)
    return client

@pytest.fixture
def test_config():
    """Returns a test configuration."""
    test_config = ModelConfig()
    test_config.model_provider = ModelProvider.CLAUDE
    test_config.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    test_config.region = "us-east-1"
    test_config.max_tokens = 1024
    test_config.temperature = 0.7
    return test_config

@pytest.fixture
def claude_response():
    """Returns a sample Claude model response."""
    return {
        "completion": "Sample response from Claude model",
        "stop_reason": "stop_sequence"
    }

@pytest.fixture
def llama_response():
    """Returns a sample Llama model response."""
    return {
        "generation": "Sample response from Llama model",
        "stop_reason": "stop_sequence"
    }

@pytest.fixture
def api_error_response():
    """Returns a sample API error response."""
    return {
        "error": {
            "message": "An error occurred during model invocation",
            "type": "InternalServerError",
            "code": 500
        }
    }

@pytest.fixture
def aws_credentials():
    """Mocked AWS Credentials for testing."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

@pytest.fixture
def boto3_bedrock_client(aws_credentials):
    """Mocked boto3 bedrock client."""
    with patch("boto3.client") as mock_client:
        client = MagicMock()
        mock_client.return_value = client
        yield client 

@pytest.fixture(autouse=True)
def setup_template_dir():
    """Create a temporary templates directory for testing"""
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(template_dir, exist_ok=True)
    yield
    # Cleanup after tests
    if os.path.exists(template_dir):
        for file in os.listdir(template_dir):
            os.remove(os.path.join(template_dir, file))
        os.rmdir(template_dir) 