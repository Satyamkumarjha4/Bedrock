from typing import Dict, Any, Optional
import logging
from .base import UseCase
from ..client import BedrockClient

logger = logging.getLogger(__name__)

class SpeechToTextUseCase(UseCase):
    """Use case for transcribing and summarizing doctor-patient conversations"""
    
    def __init__(self, client: Optional[BedrockClient] = None):
        """
        Initialize the speech-to-text use case
        
        Args:
            client: Optional BedrockClient instance. If not provided, a new one will be created.
        """
        super().__init__(client=client, template_name="speech_to_text")
    
    def format_prompt(self, data: Dict[str, Any]) -> str:
        """
        Format the prompt for speech-to-text conversion and medical information extraction
        """
        audio_data = data.get('audio_data', '')
        context = data.get('context', '')
        
        prompt = f"""Please analyze and extract information from the following doctor-patient conversation.
        
Context: {context}

Audio Data: {audio_data}

Please provide a detailed analysis in the following JSON structure:
{{
    "transcription": "full conversation text",
    "summary": "brief summary of key points",
    "medical_information": {{
        "current_medications": [
            {{
                "name": "medication name",
                "dosage": "dosage information",
                "frequency": "frequency of administration"
            }}
        ],
        "diagnoses": [
            {{
                "condition": "diagnosed condition",
                "severity": "mild/moderate/severe",
                "status": "new/existing/resolved"
            }}
        ],
        "symptoms": [
            {{
                "symptom": "symptom description",
                "duration": "how long",
                "severity": "mild/moderate/severe"
            }}
        ],
        "vital_signs": {{
            "blood_pressure": "value",
            "heart_rate": "value",
            "temperature": "value",
            "other": "other vital signs"
        }},
        "lab_results": [
            {{
                "test_name": "name of the test",
                "value": "test value",
                "unit": "unit of measurement",
                "status": "normal/abnormal"
            }}
        ],
        "follow_up_actions": [
            {{
                "action": "action description",
                "timeline": "when to do",
                "priority": "high/medium/low"
            }}
        ]
    }}
}}
"""
        return prompt
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the speech-to-text conversion and summarization
        
        Args:
            data: Input data containing audio information and optional context
            
        Returns:
            Dictionary containing transcription and summary
        """
        try:
            # Format the prompt
            prompt = self.format_prompt(data)
            
            # Invoke the model
            response = self.client.invoke(prompt)
            
            # Parse and return the response
            return self.parse_response(response)
            
        except Exception as e:
            logger.error(f"Error in speech-to-text processing: {str(e)}")
            raise
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the model response into a structured format
        
        Args:
            response: Raw model response
            
        Returns:
            Parsed response containing transcription and summary
        """
        try:
            # Extract the text from the response
            text = response.get('text', '')
            
            # The response should already be in JSON format from our prompt
            import json
            result = json.loads(text)
            
            return {
                'transcription': result.get('transcription', ''),
                'summary': result.get('summary', ''),
                'medical_information': result.get('medical_information', {})
            }
            
        except Exception as e:
            logger.error(f"Error parsing speech-to-text response: {str(e)}")
            return {
                'transcription': '',
                'summary': '',
                'medical_information': {}
            }
