import pytest
import json
from unittest.mock import patch, MagicMock
from vcare_ai.usecases.clinical_recommender import ClinicalRecommender, ClinicalRecommenderError
from vcare_ai.client import BedrockClientError
from vcare_ai.template_manager import PromptTemplate

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
        "prescriptions": ["insulin"],
        "tests": ["HbA1c"],
        "referrals": ["endocrinologist"]
    }

@pytest.fixture
def sample_text_response():
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

@pytest.fixture
def mock_template():
    return PromptTemplate(
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

class TestClinicalRecommender:
    @patch("vcare_ai.client.BedrockClient.invoke")
    def test_successful_recommendation(self, mock_invoke, sample_patient_data, sample_response):
        # Arrange
        mock_invoke.return_value = sample_response
        recommender = ClinicalRecommender()
        
        # Act
        result = recommender.run(sample_patient_data)
        
        # Assert
        assert "prescriptions" in result
        assert "tests" in result
        assert "referrals" in result
        assert result["prescriptions"] == ["insulin"]
        assert result["tests"] == ["HbA1c"]
        assert result["referrals"] == ["endocrinologist"]
        
        # Verify prompt formatting
        mock_invoke.assert_called_once()
        prompt_arg = mock_invoke.call_args[0][0]
        assert "type 2 diabetes" in prompt_arg
        assert "metformin" in prompt_arg

    @patch("vcare_ai.client.BedrockClient.invoke")
    def test_empty_input_data(self, mock_invoke, sample_response):
        # Arrange
        mock_invoke.return_value = sample_response
        recommender = ClinicalRecommender()
        
        # Act/Assert
        with pytest.raises(ClinicalRecommenderError) as exc_info:
            recommender.run({})
        
        # Verify the error message
        assert "Failed to format prompt" in str(exc_info.value)
        # Verify the model was not called
        mock_invoke.assert_not_called()

    @patch("vcare_ai.client.BedrockClient.invoke")
    def test_missing_fields_in_response(self, mock_invoke, sample_patient_data):
        # Arrange
        mock_invoke.return_value = {"prescriptions": ["insulin"]}  # Missing fields
        recommender = ClinicalRecommender()
        
        # Act
        result = recommender.run(sample_patient_data)
        
        # Assert
        assert "prescriptions" in result
        assert result["prescriptions"] == ["insulin"]
        assert "tests" in result
        assert result["tests"] == []  # Should be an empty list
        assert "referrals" in result
        assert result["referrals"] == []  # Should be an empty list

    @patch("vcare_ai.client.BedrockClient.invoke")
    def test_api_error_handling(self, mock_invoke, sample_patient_data):
        # Arrange
        mock_invoke.side_effect = BedrockClientError("API Error")
        recommender = ClinicalRecommender()
        
        # Act/Assert
        with pytest.raises(ClinicalRecommenderError) as exc_info:
            recommender.run(sample_patient_data)
        
        # Verify the error message contains the original API error
        assert "API Error" in str(exc_info.value)
        # Verify the model was called
        mock_invoke.assert_called_once()

    def test_dependency_injection(self, sample_patient_data, sample_response):
        # Arrange
        mock_client = MagicMock()
        mock_client.invoke.return_value = sample_response
        recommender = ClinicalRecommender(client=mock_client)
        
        # Act
        result = recommender.run(sample_patient_data)
        
        # Assert
        assert result == sample_response
        mock_client.invoke.assert_called_once()
        
    # New tests for template functionality
    
    @patch("vcare_ai.template_manager.TemplateManager.get_template")
    @patch("vcare_ai.client.BedrockClient.invoke")
    def test_template_based_recommendation(self, mock_invoke, mock_get_template, 
                                         sample_patient_data, sample_text_response, mock_template):
        # Arrange
        mock_get_template.return_value = mock_template
        mock_invoke.return_value = sample_text_response
        
        recommender = ClinicalRecommender(template_name="test_clinical_template")
        
        # Act
        result = recommender.run(sample_patient_data)
        
        # Assert
        assert recommender.template is mock_template
        
        # Verify template-based prompt
        mock_invoke.assert_called_once()
        prompt_arg = mock_invoke.call_args[0][0]
        assert "Provide clinical recommendations for patient aged 60" in prompt_arg
        assert "type 2 diabetes" in prompt_arg
        
        # Verify response parsing
        assert "prescriptions" in result
        assert "tests" in result
        assert "referrals" in result
        assert result["prescriptions"] == ["insulin"]
        assert result["tests"] == ["HbA1c"]
        assert result["referrals"] == ["endocrinologist"]
    
    @patch("vcare_ai.template_manager.TemplateManager.get_template")
    def test_nonexistent_template(self, mock_get_template, sample_patient_data, sample_response):
        # Arrange
        mock_get_template.return_value = None
        mock_client = MagicMock()
        mock_client.invoke.return_value = sample_response
        
        recommender = ClinicalRecommender(client=mock_client, template_name="nonexistent_template")
        
        # Act
        result = recommender.run(sample_patient_data)
        
        # Assert - should fall back to standard execution
        assert recommender.template is None
        mock_client.invoke.assert_called_once()
        
        # Verify standard prompt was used, not template-based
        prompt_arg = mock_client.invoke.call_args[0][0]
        assert "You are a clinical assistant" in prompt_arg
        
        # Verify response
        assert result == sample_response
    
    def test_parse_json_from_text_response(self, sample_text_response):
        # Arrange
        recommender = ClinicalRecommender()
        
        # Act
        result = recommender.parse_response(sample_text_response)
        
        # Assert
        assert "prescriptions" in result
        assert "tests" in result
        assert "referrals" in result
        assert result["prescriptions"] == ["insulin"]
        assert result["tests"] == ["HbA1c"]
        assert result["referrals"] == ["endocrinologist"]
    
    @patch("vcare_ai.client.BedrockClient.invoke_stream")
    def test_streaming_response(self, mock_stream, sample_patient_data):
        # Arrange
        mock_stream.return_value = [
            {"text": "{"}, 
            {"text": '"prescriptions": ["insulin"],'},
            {"text": '"tests": ["HbA1c"],'},
            {"text": '"referrals": ["endocrinologist"]'},
            {"text": "}", "stop_reason": "stop_sequence"}
        ]
        
        # Callback for collecting chunks
        received_chunks = []
        def callback(chunk):
            received_chunks.append(chunk)
        
        recommender = ClinicalRecommender()
        
        # Act
        result = recommender.run_stream(sample_patient_data, callback=callback)
        
        # Assert
        assert len(received_chunks) == 5
        assert ''.join(received_chunks) == '{"prescriptions": ["insulin"],"tests": ["HbA1c"],"referrals": ["endocrinologist"]}'
        
        # Verify parsed response
        assert "prescriptions" in result
        assert "tests" in result
        assert "referrals" in result
        assert result["prescriptions"] == ["insulin"]
        assert result["tests"] == ["HbA1c"]
        assert result["referrals"] == ["endocrinologist"]
    
    def test_input_validation(self):
        # Arrange
        recommender = ClinicalRecommender()
        invalid_input = "not a dictionary"
        
        # Act/Assert
        with pytest.raises(ClinicalRecommenderError):
            recommender._validate_input(invalid_input)
    
    def test_format_prompt(self, sample_patient_data):
        # Arrange
        recommender = ClinicalRecommender()
        
        # Act
        prompt = recommender.format_prompt(sample_patient_data)
        
        # Assert
        assert "You are a clinical assistant" in prompt
        assert "type 2 diabetes" in prompt
        assert "metformin" in prompt
        assert '"prescriptions": ["medication1", "medication2"]' in prompt
        assert '"tests": ["test1", "test2"]' in prompt
        assert '"referrals": ["specialist1", "specialist2"]' in prompt
        assert '"reasoning": "clinical reasoning for recommendations"' in prompt