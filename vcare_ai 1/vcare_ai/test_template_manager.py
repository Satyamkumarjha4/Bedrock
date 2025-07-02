import pytest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from vcare_ai.template_manager import TemplateManager, PromptTemplate

@pytest.fixture
def temp_template_file():
    """Create a temporary template file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp.write(json.dumps({
            "test_template": {
                "name": "test_template",
                "description": "A test template",
                "use_case": "clinical_recommender",
                "template_text": "This is a test template for ${patient_type}",
                "model_provider": "anthropic",
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "max_tokens": 1000,
                "temperature": 0.7,
                "response_format": {"prescriptions": [], "tests": [], "referrals": []},
                "is_active": True
            }
        }).encode())
        tmp_name = tmp.name
    
    yield tmp_name
    
    # Clean up
    if os.path.exists(tmp_name):
        os.unlink(tmp_name)

class TestTemplateManager:
    def test_singleton_pattern(self):
        """Test that TemplateManager follows the singleton pattern"""
        manager1 = TemplateManager()
        manager2 = TemplateManager()
        
        assert manager1 is manager2
        
    def test_load_templates(self, temp_template_file):
        """Test loading templates from a file"""
        manager = TemplateManager(template_file=temp_template_file)
        
        # Check that the test template was loaded
        template = manager.get_template("test_template")
        assert template is not None
        assert template.name == "test_template"
        assert template.use_case == "clinical_recommender"
        assert template.model_provider == "anthropic"
        
    def test_get_nonexistent_template(self):
        """Test getting a template that doesn't exist"""
        manager = TemplateManager()
        
        template = manager.get_template("nonexistent_template")
        assert template is None
        
    def test_add_template(self, temp_template_file):
        """Test adding a new template"""
        manager = TemplateManager(template_file=temp_template_file)
        
        new_template = PromptTemplate(
            name="new_template",
            description="A new template",
            use_case="lab_analyzer",
            template_text="Analyze these lab results: ${lab_results}",
            model_provider="meta",
            model_id="meta.llama2-70b-chat-v1",
            max_tokens=2000,
            temperature=0.5,
            response_format={"normal": [], "abnormal": []},
            is_active=True
        )
        
        manager.add_template(new_template)
        
        # Check that the template was added
        template = manager.get_template("new_template")
        assert template is not None
        assert template.name == "new_template"
        assert template.use_case == "lab_analyzer"
        assert template.model_provider == "meta"
        
    def test_remove_template(self, temp_template_file):
        """Test removing a template"""
        manager = TemplateManager(template_file=temp_template_file)
        
        # Verify template exists
        assert manager.get_template("test_template") is not None
        
        # Remove the template
        result = manager.remove_template("test_template")
        assert result is True
        
        # Check that it's gone
        assert manager.get_template("test_template") is None
        
    def test_remove_nonexistent_template(self):
        """Test removing a template that doesn't exist"""
        manager = TemplateManager()
        
        result = manager.remove_template("nonexistent_template")
        assert result is False
        
    def test_list_templates(self, temp_template_file):
        """Test listing templates"""
        manager = TemplateManager(template_file=temp_template_file)
        
        # Add another template
        new_template = PromptTemplate(
            name="another_template",
            description="Another test template",
            use_case="clinical_recommender",
            template_text="Another template text ${data}",
            model_provider="anthropic",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=1000,
            temperature=0.7,
            response_format={},
            is_active=True
        )
        manager.add_template(new_template)
        
        # List all templates
        templates = manager.list_templates()
        assert len(templates) == 2
        template_names = [t.name for t in templates]
        assert "test_template" in template_names
        assert "another_template" in template_names
        
        # List templates by use case
        templates = manager.list_templates(use_case="clinical_recommender")
        assert len(templates) == 2
        
        templates = manager.list_templates(use_case="lab_analyzer")
        assert len(templates) == 0
        
    def test_inactive_template(self, temp_template_file):
        """Test that inactive templates are not returned by get_template"""
        manager = TemplateManager(template_file=temp_template_file)
        
        # Get the template and set it to inactive
        template = manager._templates["test_template"]
        template.is_active = False
        
        # Should not be returned by get_template
        assert manager.get_template("test_template") is None
        
        # But should still be in the list (when filtering only active)
        templates = manager.list_templates()
        assert len(templates) == 0 