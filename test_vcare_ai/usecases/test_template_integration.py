import pytest
from unittest.mock import patch, MagicMock
import json
import string
from vcare_ai.template_manager import TemplateManager, PromptTemplate
from vcare_ai.usecases.clinical_recommender import ClinicalRecommender
from vcare_ai.client import BedrockClient, BedrockClientError

@pytest.fixture
def mock_template_manager():
    """Mock the TemplateManager singleton"""
    with patch('vcare_ai.template_manager.TemplateManager') as mock_manager:
        # Create a test template
        test_template = PromptTemplate(
            name="test_clinical_template",
            description="Test clinical template",
            use_case="clinical_recommender",
            template_text="Create recommendations for patient with ${conditions} and age ${age}",
            model_provider="anthropic",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=1000,
            temperature=0.7,
            response_format={"prescriptions": [], "tests": [], "referrals": []},
            is_active=True
        )
        
        # Mock the get_template method
        instance = mock_manager.return_value
        instance.get_template.return_value = test_template
        
        yield instance

@pytest.fixture
def sample_patient_data():
    return {
        "age": 60,
        "conditions": ["type 2 diabetes"],
        "lab_results": {"hba1c": 9.0},
        "medications": ["metformin"]
    }

@pytest.fixture
def sample_response():
    return {
        "text": """
        Here are my recommendations:
        
        {
            "prescriptions": ["insulin"],
            "tests": ["HbA1c"],
            "referrals": ["endocrinologist"]
        }
        """
    }

class TestTemplateIntegration:
    @patch("vcare_ai.usecases.base.TemplateManager.get_template")
    def test_template_based_execution(self, mock_get_template, sample_patient_data, sample_response):
        """Test that template-based execution works"""
        # Mock the client
        mock_client = MagicMock(spec=BedrockClient)
        mock_client.invoke.return_value = sample_response

        # Set up the mock template
        mock_template = PromptTemplate(
            name="test_clinical_template",
            description="Test template for clinical recommendations",
            use_case="clinical_recommender",
            template_text="Provide clinical recommendations for patient aged ${age} with ${conditions}",
            model_provider="anthropic",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=1000,
            temperature=0.7,
            response_format={"prescriptions": [], "tests": [], "referrals": []},
            is_active=True
        )
        mock_get_template.return_value = mock_template

        # Create recommender with template
        recommender = ClinicalRecommender(client=mock_client, template_name="test_clinical_template")

        # Act
        result = recommender.run(sample_patient_data)

        # Assert
        assert recommender.template is not None
        mock_get_template.assert_called_once_with("test_clinical_template")
        mock_client.invoke.assert_called_once()
        
        # Verify response parsing
        assert "prescriptions" in result
        assert "tests" in result
        assert "referrals" in result
        assert result["prescriptions"] == ["insulin"]
        assert result["tests"] == ["HbA1c"]
        assert result["referrals"] == ["endocrinologist"]
    
    def test_fallback_to_standard_execution(self, mock_template_manager, sample_patient_data, sample_response):
        """Test fallback to standard execution when template not found"""
        # Mock template manager to return None
        mock_template_manager.get_template.return_value = None
        
        # Mock the client
        mock_client = MagicMock(spec=BedrockClient)
        mock_client.invoke.return_value = sample_response
        
        # Create recommender with nonexistent template
        recommender = ClinicalRecommender(client=mock_client, template_name="nonexistent_template")
        
        # Verify template is None
        assert recommender.template is None
        
        # Run the recommender
        result = recommender.run(sample_patient_data)
        
        # Verify standard prompt was used
        mock_client.invoke.assert_called_once()
        prompt_arg = mock_client.invoke.call_args[0][0]
        assert "You are a clinical assistant" in prompt_arg
        
        # Verify response parsing
        assert "prescriptions" in result
        assert result["prescriptions"] == ["insulin"]
    
    @patch("vcare_ai.usecases.base.TemplateManager.get_template")
    def test_format_prompt_with_template(self, mock_get_template):
        """Test the format_prompt_with_template method"""
        # Set up the mock template
        mock_template = PromptTemplate(
            name="test_clinical_template",
            description="Test template for clinical recommendations",
            use_case="clinical_recommender",
            template_text="Provide clinical recommendations for patient aged ${age} with ${conditions}",
            model_provider="anthropic",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=1000,
            temperature=0.7,
            response_format={"prescriptions": [], "tests": [], "referrals": []},
            is_active=True
        )
        mock_get_template.return_value = mock_template

        # Create recommender with template
        recommender = ClinicalRecommender(template_name="test_clinical_template")

        # Test data
        test_data = {
            "age": 60,
            "conditions": ["type 2 diabetes"],
            "lab_results": {"hba1c": 9.0},
            "medications": ["metformin"]
        }

        # Act
        prompt = recommender.format_prompt_with_template(test_data)

        # Assert
        assert "Provide clinical recommendations for patient aged 60 with [\"type 2 diabetes\"]" in prompt
        mock_get_template.assert_called_once_with("test_clinical_template")
    
    @patch("vcare_ai.usecases.base.TemplateManager.get_template")
    def test_parse_response_with_template(self, mock_get_template):
        """Test the parse_response_with_template method"""
        # Set up the mock template
        mock_template = PromptTemplate(
            name="test_clinical_template",
            description="Test template for clinical recommendations",
            use_case="clinical_recommender",
            template_text="Provide clinical recommendations for patient aged ${age} with ${conditions}",
            model_provider="anthropic",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=1000,
            temperature=0.7,
            response_format={"prescriptions": [], "tests": [], "referrals": []},
            is_active=True
        )
        mock_get_template.return_value = mock_template

        # Create recommender with template
        recommender = ClinicalRecommender(template_name="test_clinical_template")

        # Test response
        test_response = {
            "text": """
            Here are my recommendations:
            
            {
                "prescriptions": ["insulin"],
                "tests": ["HbA1c"],
                "referrals": ["endocrinologist"]
            }
            """
        }

        # Act
        result = recommender.parse_response_with_template(test_response)

        # Assert
        assert "prescriptions" in result
        assert "tests" in result
        assert "referrals" in result
        assert result["prescriptions"] == ["insulin"]
        assert result["tests"] == ["HbA1c"]
        assert result["referrals"] == ["endocrinologist"]
        mock_get_template.assert_called_once_with("test_clinical_template")
    
    @patch("vcare_ai.client.BedrockClient.invoke_stream")
    def test_run_stream_with_template(self, mock_stream, mock_template_manager, sample_patient_data):
        """Test streaming with template support"""
        # Mock the streaming response
        mock_stream.return_value = [
            {"text": "{"}, 
            {"text": '"prescriptions": ["insulin"],'},
            {"text": '"tests": ["HbA1c"],'},
            {"text": '"referrals": ["endocrinologist"]'},
            {"text": "}", "stop_reason": "stop_sequence"}
        ]
        
        # Create a callback for testing
        received_chunks = []
        def callback(chunk):
            received_chunks.append(chunk)
        
        # Create recommender with template
        recommender = ClinicalRecommender(template_name="test_clinical_template")
        
        # Run the recommender with streaming
        result = recommender.run_stream(sample_patient_data, callback=callback)
        
        # Verify streaming response
        assert len(received_chunks) == 5
        assert received_chunks[0] == "{"
        
        # Verify final result
        assert "prescriptions" in result
        assert result["prescriptions"] == ["insulin"]
        assert result["tests"] == ["HbA1c"]
        assert result["referrals"] == ["endocrinologist"] 