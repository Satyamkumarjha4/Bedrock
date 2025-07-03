import pytest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from src.vcare_ai.usecases.clinical_recommender import ClinicalRecommender
from src.vcare_ai.utils.template_utils import (
    create_template, list_templates, delete_template,
    export_templates, import_templates
)
from vcare_ai.template_manager import TemplateManager, PromptTemplate

@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test"""
    # Setup
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(template_dir, exist_ok=True)
    template_file = os.path.join(template_dir, "templates.json")
    
    # Clear any existing templates
    with open(template_file, 'w') as f:
        json.dump({}, f)
    
    yield
    
    # Teardown
    if os.path.exists(template_file):
        os.remove(template_file)
    if os.path.exists(template_dir):
        os.rmdir(template_dir)

@pytest.fixture
def mock_template_manager():
    """Mock the TemplateManager singleton"""
    with patch('vcare_ai.utils.template_utils.TemplateManager') as mock_manager:
        # Create a list of test templates
        test_templates = [
            PromptTemplate(
                name="test_template_1",
                description="Test template 1",
                use_case="clinical_recommender",
                template_text="Template text 1",
                model_provider="anthropic",
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                max_tokens=1000,
                temperature=0.7,
                response_format={},
                is_active=True
            ),
            PromptTemplate(
                name="test_template_2",
                description="Test template 2",
                use_case="lab_analyzer",
                template_text="Template text 2",
                model_provider="meta",
                model_id="meta.llama2-70b-chat-v1",
                max_tokens=2000,
                temperature=0.5,
                response_format={},
                is_active=True
            )
        ]
        
        # Mock the methods
        instance = mock_manager.return_value
        instance.list_templates.return_value = test_templates
        instance.add_template.return_value = None
        instance.remove_template.return_value = True
        
        yield instance

class TestTemplateUtils:
    def test_create_template(self, mock_template_manager):
        """Test creating a template"""
        result = create_template(
            name="new_template",
            description="A new template",
            use_case="clinical_recommender",
            template_text="Template text",
            model_provider="anthropic",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=1000,
            temperature=0.7
        )
        
        assert result is True
        mock_template_manager.add_template.assert_called_once()
        
        # Check template properties
        template = mock_template_manager.add_template.call_args[0][0]
        assert template.name == "new_template"
        assert template.description == "A new template"
        assert template.use_case == "clinical_recommender"
        assert template.template_text == "Template text"
        assert template.model_provider == "anthropic"
        assert template.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert template.max_tokens == 1000
        assert template.temperature == 0.7
        assert template.response_format == {}
    
    def test_create_template_with_response_format(self, mock_template_manager):
        """Test creating a template with response format"""
        response_format = {"field1": [], "field2": "default"}
        
        result = create_template(
            name="new_template",
            description="A new template",
            use_case="clinical_recommender",
            template_text="Template text",
            model_provider="anthropic",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=1000,
            temperature=0.7,
            response_format=response_format
        )
        
        assert result is True
        template = mock_template_manager.add_template.call_args[0][0]
        assert template.response_format == response_format
    
    def test_list_templates(self, mock_template_manager):
        """Test listing templates"""
        templates = list_templates()
        
        assert len(templates) == 2
        assert templates[0]["name"] == "test_template_1"
        assert templates[1]["name"] == "test_template_2"
        
        # Test with use case filter
        mock_template_manager.list_templates.return_value = [
            template for template in mock_template_manager.list_templates.return_value
            if template.use_case == "clinical_recommender"
        ]
        
        templates = list_templates(use_case="clinical_recommender")
        assert len(templates) == 1
        assert templates[0]["name"] == "test_template_1"
    
    def test_delete_template(self, mock_template_manager):
        """Test deleting a template"""
        result = delete_template("test_template_1")
        
        assert result is True
        mock_template_manager.remove_template.assert_called_once_with("test_template_1")
        
        # Test deleting nonexistent template
        mock_template_manager.remove_template.return_value = False
        result = delete_template("nonexistent_template")
        assert result is False
    
    def test_export_templates(self, mock_template_manager):
        """Test exporting templates"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_name = tmp.name
        
        try:
            result = export_templates(tmp_name)
            
            assert result is True
            
            # Check file contents
            with open(tmp_name, 'r') as f:
                data = json.load(f)
            
            assert len(data) == 2
            assert data[0]["name"] == "test_template_1"
            assert data[1]["name"] == "test_template_2"
        finally:
            # Clean up
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
    
    @patch('builtins.open')
    def test_import_templates(self, mock_open, mock_template_manager):
        """Test importing templates"""
        # Mock file content
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps([
            {
                "name": "imported_template_1",
                "description": "Imported template 1",
                "use_case": "clinical_recommender",
                "template_text": "Imported template text 1",
                "model_provider": "anthropic",
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "max_tokens": 1000,
                "temperature": 0.7,
                "response_format": {},
                "is_active": True
            },
            {
                "name": "imported_template_2",
                "description": "Imported template 2",
                "use_case": "lab_analyzer",
                "template_text": "Imported template text 2",
                "model_provider": "meta",
                "model_id": "meta.llama2-70b-chat-v1",
                "max_tokens": 2000,
                "temperature": 0.5,
                "response_format": {},
                "is_active": True
            }
        ])
        
        result = import_templates(mock_open.return_value.__enter__.return_value)
        
        assert result == 2
        assert mock_template_manager.add_template.call_count == 2
        
        # Get all template names that were added
        added_names = [call_args[0][0].name for call_args in mock_template_manager.add_template.call_args_list]
        assert "imported_template_1" in added_names
        assert "imported_template_2" in added_names
        
        # Optionally, check properties for both templates
        for call_args in mock_template_manager.add_template.call_args_list:
            template = call_args[0][0]
            if template.name == "imported_template_1":
                assert template.description == "Imported template 1"
                assert template.use_case == "clinical_recommender"
            elif template.name == "imported_template_2":
                assert template.description == "Imported template 2"
                assert template.use_case == "lab_analyzer"

        # Check prompt format
        for call_args in mock_template_manager.add_template.call_args_list:
            template = call_args[0][0]
            prompt = template.template_text
            assert "Format your response as a JSON object with the following structure:" in prompt
            assert '"prescriptions": ["medication1", "medication2"]' in prompt
            assert '"tests": ["test1", "test2"]' in prompt
            assert '"referrals": ["specialist1", "specialist2"]' in prompt

    def test_format_prompt(self, sample_patient_data):
        # Arrange
        recommender = ClinicalRecommender()

        # Act
        prompt = recommender.format_prompt(sample_patient_data)

        # Assert
        assert "You are a clinical assistant" in prompt
        assert "type 2 diabetes" in prompt
        assert "metformin" in prompt
        assert "Format your response as a JSON object with the following structure:" in prompt
        assert '"prescriptions": ["medication1", "medication2"]' in prompt
        assert '"tests": ["test1", "test2"]' in prompt
        assert '"referrals": ["specialist1", "specialist2"]' in prompt

def test_create_template():
    """Test creating a new template"""
    template_data = {
        "name": "test_template",
        "description": "Test template",
        "use_case": "test_case",
        "template_text": "Test template with ${variable}",
        "model_provider": "anthropic",
        "model_id": "claude-3-sonnet",
        "max_tokens": 2048,
        "temperature": 0.5,
        "response_format": {"type": "object"}
    }
    
    assert create_template(**template_data)
    
    # Verify template was created
    templates = list_templates()
    assert any(t["name"] == "test_template" for t in templates)

def test_format_prompt():
    """Test formatting a prompt with variables"""
    template_data = {
        "name": "test_template",
        "description": "Test template",
        "use_case": "test_case",
        "template_text": "Hello ${name}, your age is ${age}",
        "model_provider": "anthropic",
        "model_id": "claude-3-sonnet"
    }

    # Create template
    success = create_template(**template_data)
    assert success, "Template creation failed"

    # Verify template file exists and contains the template
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    template_file = os.path.join(template_dir, "templates.json")
    assert os.path.exists(template_file), "Template file was not created"

    # Read and verify file contents
    with open(template_file, 'r') as f:
        saved_templates = json.load(f)
        print(f"Saved templates: {saved_templates}")  # Debug print
    assert "test_template" in saved_templates, "Template was not saved to file"

    # Get template and verify
    manager = TemplateManager()
    template = manager.get_template("test_template")
    assert template is not None, "Template not found after creation"
    assert template.name == "test_template"
    assert template.template_text == "Hello ${name}, your age is ${age}"

    # Verify template is in manager's templates
    assert "test_template" in manager._templates, "Template not found in manager's templates" 