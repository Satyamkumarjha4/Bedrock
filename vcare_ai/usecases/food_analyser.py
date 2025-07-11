import logging
from typing import Dict, Any, Optional, Callable, List
import json
from vcare_ai.client import BedrockClient
from vcare_ai.usecases.base import UseCase

import re

def extract_json_from_text(text: str) -> dict:
    """
    Extracts the first valid JSON object from a Claude-style response.
    """
    try:
        return json.loads(text)  # Try direct parse
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*?\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return {"error": "Failed to parse extracted JSON"}
    return {"error": "No JSON found in response"}


logger = logging.getLogger(__name__)

class FoodAnalyserError(Exception):
    """Errors specific to food analysis"""
    pass

class FoodAnalyser(UseCase):
    """UseCase implementation for analysing food"""
    
    def __init__(self, client: Optional[BedrockClient] = None, template_name: Optional[str] = None):
        super().__init__(client=client, template_name=template_name)
        logger.debug(f"Initialized FoodAnalyser with template: {template_name or 'None'}")

    def detect_food_items_from_image(self, image_url: str) -> Dict[str, str]:
        """
        Uses a vision model to detect food items and their quantities from an image URL.
        Returns a dictionary where keys are food items and values are quantities.
        """
        prompt = (
            "You are a vision model tasked with identifying Indian food items in a meal image.\n"
            "Return **ONLY** a single JSON object – **no markdown fences, no comments, no extra text** – "
            "with the structure:\n"
            "{\n"
            '  "food_items": {\n'
            '     "<item1>": "<quantity1>",\n'
            '     "<item2>": "<quantity2>",\n'
            '     "<item3>": "<quantity3>"\n'
            '  }\n'
            "}\n"
            "Use double quotes around every key and value, use the units g / ml / pcs, "
            "and do **not** put a comma after the last item. give as many items as in the image give atleast 1\n"
            f"Image URL: {image_url}\n"
        )


        response = self.client.invoke(prompt, use_cache=False, image_url=image_url)
        text = response.get('text', '')

        try:
            parsed = extract_json_from_text(text)
            food_items = parsed.get("food_items", {})
            if isinstance(food_items, dict) and food_items:
                return food_items
            else:
                logger.warning(f"Empty or invalid 'food_items': {parsed}")
                return {}
        except json.JSONDecodeError:
            logger.error(f"JSON parsing failed. Raw response: {text}")
            return {}
        except Exception as e:
            logger.exception("Unexpected error while parsing vision model response")
            return {}



    def format_prompt(self, food_items: Dict[str, str], req_data: Dict[str, Any]) -> str:
        if not food_items or not req_data:
            raise FoodAnalyserError("Both 'food_items' and 'req_data' must be provided.")

        if not isinstance(food_items, dict) or not isinstance(req_data, dict):
            raise FoodAnalyserError("'food_items' must be a dict and 'req_data' must be a dictionary.")

        prompt = (
            "You are a nutrition assistant. Based on the list of food items identified in an Indian thali meal, "
            "with their estimated quantities, and the user's recommended nutrient intake, perform the following:\n"
            "1. Estimate the total amounts of macronutrients (carbohydrates, proteins, fats), fibre, and calories in the meal.\n"
            "2. Compare these estimated values to the recommended intake and calculate the deviation for each nutrient (meal value minus recommended value).\n"
            "3. Provide a brief recommendation based on the deviations.\n"
            "Respond strictly in JSON format with the following keys:\n"
            "  - 'current_nutrients': dictionary of estimated nutrient values\n"
            "  - 'deviation': dictionary of deviations for each nutrient\n"
            "  - 'recommendation': a short recommendation string\n"
            f"Food items with quantities: {food_items}\n"
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
