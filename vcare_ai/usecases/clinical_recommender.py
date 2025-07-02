import json
import logging
from typing import Dict, Any, Optional,  Callable
from vcare_ai.client import BedrockClient
from vcare_ai.usecases.base import UseCase

logger = logging.getLogger(__name__)

class ClinicalRecommenderError(Exception):
    """Errors specific to clinical recommendations"""
    pass

class ClinicalRecommender(UseCase):
    """UseCase implementation for generating clinical recommendations"""
    
    def __init__(self, client: Optional[BedrockClient] = None, template_name: Optional[str] = None):
        """Initialize with optional client instance and template name"""
        super().__init__(client, template_name)
        logger.debug(f"Initialized ClinicalRecommender with template: {template_name or 'None'}")
    
    def format_prompt(self, data: Dict[str, Any]) -> str:
        """
        Format patient data into a prompt for the model
        
        Args:
            data: Patient information dictionary
            
        Returns:
            Formatted prompt string
        """
        try:
            # Validate required data
            self._validate_input(data)
            validated_data = data
            
            # Format the prompt with specific instructions
            prompt = (
                    "You are a clinical assistant helping a doctor analyze patient data "
                    "and provide evidence-based recommendations.\n\n"
                    "Patient Information:\n"
                    f"- Age: {validated_data['age']}\n"
                    f"- Medical Conditions: {', '.join(validated_data['conditions'])}\n"
                    f"- Lab Results: {', '.join(f'{k}: {v}' for k, v in validated_data['lab_results'].items())}\n"
                    f"- Current Medications: {', '.join(validated_data['medications']) if validated_data.get('medications') else 'None'}\n\n"
                    "Based on this information, provide:\n"
                    "1. Recommended prescriptions or medication changes\n"
                    "2. Suggested medical tests or monitoring\n"
                    "3. Specialist referrals if needed\n"
                    "4. Clinical reasoning for your recommendations\n\n"
                    "Format your response as a JSON object with the following structure:\n"
                    "{\n"
                    '  "prescriptions": ["medication1", "medication2"],\n'
                    '  "tests": ["test1", "test2"],\n'
                    '  "referrals": ["specialist1", "specialist2"],\n'
                    '  "reasoning": "clinical reasoning for recommendations"\n'
                    "}\n"
                    "Respond ONLY with a valid JSON object in the format above. Do not include any explanation or text outside the JSON."
                )
            logger.debug(f"Created prompt with {len(data)} data points")
            return prompt
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            raise ClinicalRecommenderError(f"Failed to format prompt: {str(e)}")
    
    def _validate_input(self, data: Dict[str, Any]) -> None:
        """
        Validate input data for required fields
        
        Args:
            data: Input data to validate
            
        Raises:
            ClinicalRecommenderError: If validation fails
        """
        # Example validation - can be customized based on requirements
        if not isinstance(data, dict):
            raise ClinicalRecommenderError("Input must be a dictionary")
        
        # Optional - check for minimum required fields
        # required_fields = ["age", "conditions"]
        # missing = [field for field in required_fields if field not in data]
        # if missing:
        #     raise ClinicalRecommenderError(f"Missing required fields: {', '.join(missing)}")
        
        logger.debug("Input data validation successful")
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate the model response
        
        Args:
            response: Raw model response
            
        Returns:
            Parsed response with standardized fields
        """
        try:
            # Extract text field added by client._parse_response
            text_content = response.get("text", "")
            
            # If response is already parsed as JSON by the model
            if isinstance(response, dict) and any(k in response for k in ["prescriptions", "tests", "referrals"]):
                result = response
            else:
                # Handle case where we might need to parse JSON from text
                try:
                    # Try to parse JSON from the text field if it exists
                    if text_content and isinstance(text_content, str):
                        if "{" in text_content:
                            # Extract JSON portion if embedded in text
                            json_str = text_content[text_content.find("{"):text_content.rfind("}")+1]
                            result = json.loads(json_str)
                        else:
                            result = {"text": text_content}
                    else:
                        result = response
                except json.JSONDecodeError:
                    logger.warning("Could not parse JSON from response text")
                    result = {"text": text_content} if text_content else response
            
            # Ensure minimum expected fields
            for field in ["prescriptions", "tests", "referrals"]:
                if field not in result:
                    result[field] = []
            
            logger.debug(f"Parsed response with {len(result)} fields")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return {"error": str(e), "raw_response": str(response)[:100]}
    
    def run(self, data: Dict[str, Any], use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate clinical recommendations based on patient data
        
        Args:
            data: Patient information
            use_cache: Whether to use cached responses
            
        Returns:
            Dictionary with recommendations
            
        Raises:
            ClinicalRecommenderError: If processing fails
        """
        # If we have a template, use template-based execution
        if self.template:
            try:
                return self.run_with_template(data, use_cache)
            except Exception as e:
                raise ClinicalRecommenderError(f"Template-based execution failed: {str(e)}")
        
        # Otherwise, use standard execution
        request_id = f"req_{id(data):x}"
        logger.info(f"Processing request {request_id}")
        
        try:
            # Format the prompt
            prompt = self.format_prompt(data)
            
            # Invoke the model
            logger.debug(f"Invoking model for request {request_id}")
            response = self.client.invoke(prompt, use_cache=use_cache)
            
            # Parse and validate the response
            result = self.parse_response(response)
            
            logger.info(f"Successfully processed request {request_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in request {request_id}: {str(e)}")
            raise ClinicalRecommenderError(f"Failed to generate recommendations: {str(e)}")
    
    def run_stream(self, data: Dict[str, Any], callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Generate clinical recommendations with streaming response
        
        Args:
            data: Patient information
            callback: Optional callback function for streaming chunks
            
        Returns:
            Final dictionary with recommendations
            
        Raises:
            ClinicalRecommenderError: If processing fails
        """
        request_id = f"stream_{id(data):x}"
        logger.info(f"Processing streaming request {request_id}")
        
        try:
            # Format the prompt (with or without template)
            prompt = self.format_prompt_with_template(data) if self.template else self.format_prompt(data)
            
            # Get streaming response
            accumulated_text = ""
            final_response = None
            
            for chunk in self.client.invoke_stream(prompt):
                chunk_text = chunk.get("text", "")
                accumulated_text += chunk_text
                
                # Call the callback with the chunk if provided
                if callback and callable(callback):
                    callback(chunk_text)
                
                # Check if we have a complete response
                if chunk.get("stop_reason"):
                    final_response = chunk
            
            # Parse the accumulated text as the final response
            if final_response:
                result = self.parse_response_with_template({"text": accumulated_text}) if self.template else self.parse_response({"text": accumulated_text})
            else:
                result = self.parse_response_with_template({"text": accumulated_text}) if self.template else self.parse_response({"text": accumulated_text})
            
            logger.info(f"Successfully processed streaming request {request_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in streaming request {request_id}: {str(e)}")
            raise ClinicalRecommenderError(f"Failed to generate streaming recommendations: {str(e)}")
