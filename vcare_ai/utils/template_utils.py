import json
import os
from typing import Dict, Any, List, Optional
from ..template_manager import TemplateManager, PromptTemplate
import logging

logger = logging.getLogger(__name__)

def create_template(
    name: str,
    description: str,
    use_case: str,
    template_text: str,
    model_provider: str,
    model_id: str,
    max_tokens: int = 2048,
    temperature: float = 0.5,
    response_format: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Create a new prompt template
    
    Args:
        name: Template name
        description: Template description
        use_case: Use case name (e.g., 'clinical_recommender')
        template_text: Prompt template text with ${variable} placeholders
        model_provider: Model provider (e.g., 'anthropic')
        model_id: Model ID
        max_tokens: Maximum tokens to generate
        temperature: Temperature parameter
        response_format: Expected response format structure
        
    Returns:
        True if successful, False otherwise
    """
    try:
        template = PromptTemplate(
            name=name,
            description=description,
            use_case=use_case,
            template_text=template_text,
            model_provider=model_provider,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format or {},
            is_active=True
        )
        
        manager = TemplateManager()
        logger.info(f"Adding template: {template.to_dict()}")
        manager.add_template(template)
        logger.info(f"Template added successfully. Current templates: {manager._templates}")
        return True
    except Exception as e:
        logger.error(f"Error creating template: {str(e)}")
        return False

def list_templates(use_case: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List available templates
    
    Args:
        use_case: Optional use case to filter by
        
    Returns:
        List of template dictionaries
    """
    manager = TemplateManager()
    templates = manager.list_templates(use_case)
    return [template.to_dict() for template in templates]

def delete_template(name: str) -> bool:
    """
    Delete a template by name
    
    Args:
        name: Template name
        
    Returns:
        True if deleted, False otherwise
    """
    manager = TemplateManager()
    return manager.remove_template(name)

def export_templates(output_file: str) -> bool:
    """
    Export templates to a JSON file
    
    Args:
        output_file: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        templates = list_templates()
        with open(output_file, 'w') as f:
            json.dump(templates, f, indent=2)
        return True
    except Exception as e:
        print(f"Error exporting templates: {str(e)}")
        return False

def import_templates(input_file: str) -> int:
    """
    Import templates from a JSON file
    
    Args:
        input_file: Input file path
        
    Returns:
        Number of templates imported
    """
    try:
        with open(input_file, 'r') as f:
            templates_data = json.load(f)
        
        manager = TemplateManager()
        imported_count = 0
        
        for template_data in templates_data:
            template = PromptTemplate(**template_data)
            manager.add_template(template)
            imported_count += 1
            
        return imported_count
    except Exception as e:
        print(f"Error importing templates: {str(e)}")
        return 0 