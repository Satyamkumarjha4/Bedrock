import logging
from typing import Dict, Any, Optional, Callable
from vcare_ai.client import BedrockClient
from vcare_ai.usecases.base import UseCase

logger = logging.getLogger(__name__)

class FoodAnalyserError(Exception):
    """Errors specific to food analysis"""
    pass

class FoodAnalyser(UseCase):
    """UseCase implementation for analysing food"""
    
    def __init__(self, client: Optional[BedrockClient] = None, template_name: Optional[str] = None):
        super().__init__(client=client, template_name=template_name)
        logger.debug(f"Initialized FoodAnalyser with template: {template_name or 'None'}")
    
    def format_prompt(self, data: Dict[str, Any]) -> str:
        """
        Format the prompt for food analysis
        """
        food_data = data.get('food_data', '')
        req_data = {
            "carbohydrates": 200,
            "proteins": 150,
            "fats": 70,
            "fibre": 30,
            "calories": 2500
        }
        prompt = (
            "You are a nutrition assistant. Given the following meal information and the user's recommended daily intake, "
            "analyze the meal and estimate the amounts of macronutrients (carbohydrates, proteins, fats), fibre, and calories present. "
            "Then, compare these values to the recommended intake and provide the deviation for each nutrient. "
            "Respond strictly in JSON format with two keys: 'current_nutrients' and 'deviation'.\n"
            f"Meal data: {food_data}\n"
            f"Recommended intake: {req_data}\n"
            "Example response:\n"
            '{"current_nutrients": {"carbohydrates": 40, "proteins": 30, "fats": 30, "fibre": 54, "calories": 2000}, '
            '"deviation": {"carbohydrates": 10, "proteins": -20, "fats": 5, "fibre": 0, "calories": -500}}'
        )
        return prompt
    
    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the response from the model that is in JSON format
        """
        import json
        text = response.get('text', '')
        try:
            return json.loads(text)
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {text}")
            return {"error": str(e), "raw_response": text}
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.format_prompt(data)
        response = self.client.invoke(prompt, use_cache=True)
        return self.parse_response(response)
    
    
        
    
    
