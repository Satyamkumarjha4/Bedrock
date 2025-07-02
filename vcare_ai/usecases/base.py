from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterator, Callable
import string
import logging
import time
import json
from ..client import BedrockClient
from ..template_manager import TemplateManager, PromptTemplate
from ..config import ModelConfig

logger = logging.getLogger(__name__)

class UseCase(ABC):
    """Base class for all AI use cases"""
    
    def __init__(self, client: Optional[BedrockClient] = None, template_name: Optional[str] = None):
        """
        Initialize the use case
        
        Args:
            client: Optional BedrockClient instance. If not provided, a new one will be created.
            template_name: Optional template name to use
        """
        self.template_manager = TemplateManager()
        self.template = None
        
        # Load template if name is provided
        if template_name:
            self.template = self.template_manager.get_template(template_name)
            
            # Create client with template config if template found and no client provided
            if self.template and not client:
                custom_config = ModelConfig()
                custom_config.model_provider = self.template.model_provider
                custom_config.model_id = self.template.model_id
                custom_config.max_tokens = self.template.max_tokens
                custom_config.temperature = self.template.temperature
                
                self.client = BedrockClient(custom_config=custom_config)
            else:
                self.client = client or BedrockClient()
        else:
            self.client = client or BedrockClient()
    
    def format_prompt_with_template(self, data: Dict[str, Any]) -> str:
        """
        Format prompt using a template
        
        Args:
            data: Data to format the template with
            
        Returns:
            Formatted prompt string
        """
        if not self.template:
            # Fall back to standard format_prompt if no template
            return self.format_prompt(data)
            
        try:
            # Use the template with string formatting
            template = string.Template(self.template.template_text)
            
            # Convert data to strings for formatting
            str_data = {}
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    str_data[k] = json.dumps(v)
                else:
                    str_data[k] = str(v)
            
            return template.safe_substitute(str_data)
        except Exception as e:
            logger.error(f"Error formatting with template: {str(e)}")
            # Fall back to standard format_prompt
            return self.format_prompt(data)
    
    @abstractmethod
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the use case logic
        
        Args:
            data: Input data for the use case
            
        Returns:
            The result of the use case execution
        """
        pass
    
    def format_prompt(self, data: Dict[str, Any]) -> str:
        """
        Format the prompt for the specific use case
        
        Args:
            data: Input data to format into a prompt
            
        Returns:
            Formatted prompt string
        """
        raise NotImplementedError("Subclasses must implement format_prompt")
    
    def parse_response_with_template(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the response using the template's response format
        
        Args:
            response: Raw response from the model
            
        Returns:
            Parsed and formatted response
        """
        if not self.template or not self.template.response_format:
            return self.parse_response(response)
            
        try:
            # Extract text from response
            text = response.get('text', '')
            
            # Try to extract JSON from text
            import re
            
            # Find JSON in text if it exists
            json_match = re.search(r'({.*})', text, re.DOTALL)
            if json_match:
                extracted_json = json.loads(json_match.group(1))
                
                # Apply the template format structure
                result = {}
                for key, default in self.template.response_format.items():
                    result[key] = extracted_json.get(key, default)
                
                return result
            
            # Fall back to default parsing
            return self.parse_response(response)
            
        except Exception as e:
            logger.error(f"Error applying response format: {str(e)}")
            return self.parse_response(response)
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the model response for the specific use case
        
        Args:
            response: Raw model response
            
        Returns:
            Parsed and formatted response
        """
        return response
    
    def run_with_template(self, data: Dict[str, Any], use_cache: bool = True) -> Dict[str, Any]:
        """
        Run the use case with template support
        
        Args:
            data: Input data
            use_cache: Whether to use response caching
            
        Returns:
            Processed response
        """
        start_time = time.time()
        request_id = f"req_{id(data):x}"
        logger.info(f"Processing template-based request {request_id}")
        
        try:
            # Use template-based formatting if available
            prompt = self.format_prompt_with_template(data)
            
            # Invoke the model
            response = self.client.invoke(prompt, use_cache=use_cache)
            
            # Parse using template if available
            result = self.parse_response_with_template(response)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Completed request {request_id} in {elapsed_ms}ms")
            
            return result
            
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Error in request {request_id} after {elapsed_ms}ms: {str(e)}")
            raise
    
    def run_stream(self, data: Dict[str, Any], callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Execute the use case with streaming responses
        
        Args:
            data: Input data for the use case
            callback: Optional function to call with each chunk of the response
            
        Returns:
            The final result after all chunks are received
        """
        raise NotImplementedError("Streaming not implemented for this use case")
    
    def close(self):
        """Close and clean up resources"""
        if hasattr(self, 'client') and self.client:
            self.client.close()

