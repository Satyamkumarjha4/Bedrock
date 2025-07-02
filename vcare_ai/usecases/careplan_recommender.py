import json
import logging
from typing import Dict, Any, Optional, Callable
from vcare_ai.client import BedrockClient
from vcare_ai.usecases.base import UseCase

logger = logging.getLogger(__name__)

class CarePlanRecommenderError(Exception):
    """Errors specific to careplan recommendations"""
    pass

class CarePlanRecommender(UseCase):
    """
    UseCase implementation for recommending a FHIR CarePlan
    """
    def __init__(self, client: Optional[BedrockClient] = None, template_name: Optional[str] = None):
        """Initialize with optional client instance and template name"""
        super().__init__(client, template_name)
        logger.debug(f"Initialized CarePlanRecommender with template: {template_name or 'None'}")
    

    def format_prompt(self, data: Dict[str, Any]) -> str:
        """
        Format careplan summary and context into a prompt for the model

        Args:
            data: Careplan summary dictionary
            
        Returns:
            Formatted prompt string
        """
        try:
            self._validate_input(data)
            validated_data = data
            # Add other context as needed (labs, conditions, etc.)
            prompt = (
                    "You are a clinical assistant helping design a personalized, standards-compliant FHIR CarePlan for a patient with chronic conditions.\n\n"
                    
                    "The patient's current careplan activities (LOINC/SNOMED-coded) are:\n"
                    f"{json.dumps(validated_data['careplan_summary'], indent=2)}\n\n"
                    
                    "The latest clinical recommendations from the AI system are:\n"
                    f"{json.dumps(validated_data['clinical_recommendation'], indent=2)}\n\n"
                    
                    "Based on this information, return an UPDATED FHIR CarePlan in JSON format that:\n"
                    "- Uses only LOINC or SNOMED codes in the `activity.detail.code`\n"
                    "- Includes `title`, `description`, and optional `goal` mapping for each activity\n"
                    "- Clearly includes `status: scheduled` or `completed` per activity\n"
                    "- Is fully self-contained (with `resourceType`, `status`, `intent`, `activity`, and optionally `goal` sections)\n"
                    "- Groups activities logically (e.g., monitoring, education, screening)\n\n"
                    
                    "Respond ONLY with a valid FHIR CarePlan JSON object. Do not include any explanation or text outside the JSON block.\n"
                )
            logger.debug(f"Created careplan prompt with {len(validated_data)} data points")
            return prompt
        except Exception as e:
            logger.error(f"Error formatting careplan prompt: {str(e)}")
            raise CarePlanRecommenderError(f"Failed to format careplan prompt: {str(e)}")
        
    def _validate_input(self, data: Dict[str, Any]) -> None:
        """
        Validate input data for required fields
        
        Args:
            data: Input data to validate
            
        Raises:
            CarePlanRecommenderError: If validation fails
        """
        # Example validation - can be customized based on requirements
        if not isinstance(data, dict):
            raise CarePlanRecommenderError("Input must be a dictionary")
        
        # Optional - check for minimum required fields
        required_fields = ["careplan_summary", "clinical_recommendation"]
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise CarePlanRecommenderError(f"Missing required fields: {', '.join(missing)}")
        
        logger.debug("Input data validation successful")

    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate the model response (FHIR CarePlan JSON)

        Args:
            response: Raw model response
            
        Returns:
            Parsed response with standardized fields
        """
        try:
            text_content = response.get("text", "")
            # Try to parse JSON from the text field
             # If response is already parsed as JSON by the model
            if isinstance(response, dict) and any(k in response for k in ["careplan_summary", "clinical_recommendation"]):
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
            for field in ["careplan_summary", "clinical_recommendation"]:
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
            CarePlanRecommenderError: If processing fails
        """
        # If we have a template, use template-based execution
        if self.template:
            try:
                return self.run_with_template(data, use_cache)
            except Exception as e:
                raise CarePlanRecommenderError(f"Template-based execution failed: {str(e)}")
        
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
            raise CarePlanRecommenderError(f"Failed to generate recommendations: {str(e)}")
    
    def run_stream(self, data: Dict[str, Any], callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Generate clinical recommendations with streaming response
        
        Args:
            data: Patient information
            callback: Optional callback function for streaming chunks
            
        Returns:
            Final dictionary with recommendations
            
        Raises:
            CarePlanRecommenderError: If processing fails
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
            raise CarePlanRecommenderError(f"Failed to generate streaming recommendations: {str(e)}")
        