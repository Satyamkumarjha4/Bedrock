import json
import logging
import os
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    """Data class for storing prompt templates"""
    name: str
    description: str
    use_case: str
    template_text: str
    model_provider: str
    model_id: str
    max_tokens: int = 2048
    temperature: float = 0.5
    response_format: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class TemplateManager:
    """Manager for prompt templates"""
    _instance = None
    _templates: Dict[str, PromptTemplate] = {}
    _template_file: str = "templates.json"
    
    def __new__(cls, template_file: Optional[str] = None):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(TemplateManager, cls).__new__(cls)
            if template_file:
                cls._instance._template_file = template_file
            # Use test directory if running tests
            if 'pytest' in sys.modules:
                cls._instance._template_dir = os.path.join(os.path.dirname(__file__), "..", "test", "vcare_ai", "templates")
            else:
                cls._instance._template_dir = os.path.join(os.path.dirname(__file__), "templates")
            cls._instance._load_templates()
        return cls._instance
    
    def _load_templates(self) -> None:
        """Load templates from file"""
        try:
            # Create template directory if it doesn't exist
            if not os.path.exists(self._template_dir):
                os.makedirs(self._template_dir)
            
            template_path = os.path.join(self._template_dir, self._template_file)
            
            # Create default file if it doesn't exist
            if not os.path.exists(template_path):
                with open(template_path, 'w') as f:
                    json.dump({}, f)
                logger.info(f"Created empty template file at {template_path}")
                return
            
            # Load templates
            with open(template_path, 'r') as f:
                templates_data = json.load(f)
            
            # Clear existing templates and load new ones
            self._templates.clear()
            for name, data in templates_data.items():
                self._templates[name] = PromptTemplate(**data)
            
            logger.info(f"Loaded {len(self._templates)} templates")
            
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
    
    def _save_templates(self) -> None:
        """Save templates to file"""
        try:
            template_path = os.path.join(self._template_dir, self._template_file)
            logger.info(f"Saving templates to {template_path}")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(template_path), exist_ok=True)
            
            templates_data = {name: template.to_dict() for name, template in self._templates.items()}
            logger.info(f"Template data to save: {templates_data}")
            
            with open(template_path, 'w') as f:
                json.dump(templates_data, f, indent=2)
            
            logger.info(f"Saved {len(self._templates)} templates to {template_path}")
            
        except Exception as e:
            logger.error(f"Error saving templates: {str(e)}")
            raise  # Re-raise the exception to help with debugging
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name"""
        template = self._templates.get(name)
        if template and template.is_active:
            return template
        return None
    
    def add_template(self, template: PromptTemplate) -> None:
        """Add or update a template"""
        logger.info(f"Adding template {template.name} to manager")
        self._templates[template.name] = template
        logger.info(f"Current templates before save: {self._templates}")
        self._save_templates()
        logger.info(f"Current templates after save: {self._templates}")
    
    def remove_template(self, name: str) -> bool:
        """Remove a template"""
        if name in self._templates:
            del self._templates[name]
            self._save_templates()
            return True
        return False
    
    def list_templates(self, use_case: Optional[str] = None) -> List[PromptTemplate]:
        """List all templates, optionally filtered by use case"""
        if use_case:
            return [t for t in self._templates.values() if t.is_active and t.use_case == use_case]
        return [t for t in self._templates.values() if t.is_active] 