import logging
from typing import Dict, Any, Optional, Callable, List
import json
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

    def detect_food_items_from_image(self, image_url: str) -> List[str]:
        """
        Uses a vision model to detect food items from an image URL.
        """
        prompt = (
            "You are a vision model tasked with identifying Indian food items in a meal image. "
            "Given the image URL, list all recognizable food items present in the plate. "
            "Respond strictly in a JSON format as: {\"food_items\": [item......]}."
            f"Image URL: {image_url}\n"
        )

        response = self.client.invoke(prompt, use_cache=False, image_url=image_url)
        text = response.get('text', '')

        try:
            parsed = json.loads(text)
            food_items = parsed.get("food_items", [])
            if isinstance(food_items, list) and food_items:
                return food_items
            else:
                logger.warning(f"Empty or invalid 'food_items': {parsed}")
                return []  # Return empty list instead of raising error
        except json.JSONDecodeError:
            logger.error(f"JSON parsing failed. Raw response: {text}")
            return []  # Gracefully handle partial/malformed responses
        except Exception as e:
            logger.exception("Unexpected error while parsing vision model response")
            return []


    def format_prompt(self, food_items: List[str], req_data: Dict[str, Any]) -> str:
        """
        Format the prompt for nutrient estimation and recommendation
        """
        if not food_items or not req_data:
            raise FoodAnalyserError("Both 'food_items' and 'req_data' must be provided.")

        if not isinstance(food_items, list) or not isinstance(req_data, dict):
            raise FoodAnalyserError("'food_items' must be a list and 'req_data' must be a dictionary.")

        prompt = (
            "You are a nutrition assistant. Based on the list of food items identified in an Indian thali meal, "
            "and the user's recommended nutrient intake, perform the following:\n"
            "1. Estimate the total amounts of macronutrients (carbohydrates, proteins, fats), fibre, and calories in the meal.\n"
            "2. Compare these estimated values to the recommended intake and calculate the deviation for each nutrient (meal value minus recommended value).\n"
            "3. Provide a brief recommendation based on the deviations.\n"
            "Respond strictly in JSON format with the following keys:\n"
            "  - 'current_nutrients': dictionary of estimated nutrient values\n"
            "  - 'deviation': dictionary of deviations for each nutrient\n"
            "  - 'recommendation': a short recommendation string\n"
            f"Food items: {food_items}\n"
            f"Recommended intake: {req_data}\n"
        )
        return prompt

    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the JSON response from the model, stripping markdown formatting if needed.
        """
        text = response.get('text', '').strip()

        # Remove ```json ... ``` if present
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            # Remove first and last line (```json and ```)
            text = "\n".join(lines[1:-1]).strip()

        try:
            return json.loads(text)
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {text}")
            return {"error": str(e), "raw_response": text}


    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image_url = data.get('food_data', '')
        req_data = data.get('req_data', {})

        if not image_url or not req_data:
            return {"error": "Both 'food_data' (image URL) and 'req_data' must be provided."}

        food_items = self.detect_food_items_from_image(image_url)
        if not food_items:
            return {
                "error": "No food items detected from the image.",
                "note": "Vision model returned empty or unrecognized list."
            }

        prompt = self.format_prompt(food_items, req_data)
        response = self.client.invoke(prompt, use_cache=True, image_url=image_url)
        final_output = self.parse_response(response)
        final_output["food_items"] = food_items
        return final_output
