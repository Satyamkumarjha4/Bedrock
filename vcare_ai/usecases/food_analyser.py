# food_analyser.py
import logging
import json
import re
from typing import Dict, Any, List, Tuple, Optional
from vcare_ai.client import BedrockClient
from vcare_ai.usecases.base import UseCase
from vcare_ai.utils.vectorDB_utils import VectorDBUtils
from vcare_ai.utils.ingredient_parser import parse_ingredients
from vcare_ai.utils.ingredient_fallback import IngredientFallback

logger = logging.getLogger(__name__)

class FoodAnalyserError(Exception):
    """Errors specific to food analysis"""
    pass

class FoodAnalyser(UseCase):
    """Enhanced food analysis with multi-stage ingredient handling"""
    
    def __init__(self, client: Optional[BedrockClient] = None, template_name: Optional[str] = None):
        super().__init__(client=client, template_name=template_name)
        try:
            self.vector_db = VectorDBUtils()
        except Exception as e:
            logger.error(f"Failed to initialize VectorDB: {str(e)}")
            raise FoodAnalyserError("Database initialization failed")
            
        self.fallback_parser = IngredientFallback()
        self.fallback_nutrients_cache = {}
        logger.info("Initialized FoodAnalyser with fallback support")

    def get_dish_details_from_image(self, image_base64: str) -> Dict[str, Any]:
        """
        Enhanced vision analysis to get dish name and ingredients
        Returns: {
            "dish_name": str,
            "ingredients": [{"name": str, "quantity": float}],
            "confidence": int (0-100)
        }
        """
        prompt = """Analyze this food image and return JSON with:
        - dish_name: Most probable name
        - ingredients: List of {name, quantity_in_grams}
        - confidence: Estimation confidence (0-100)
        
        Example: {
            "dish_name": "Chicken Biryani",
            "ingredients": [
                {"name": "rice", "quantity": 200},
                {"name": "chicken", "quantity": 150}
            ],
            "confidence": 85
        }"""
        
        try:
            response = self.client.invoke(prompt, use_cache=False, image_url=image_base64)
            logger.debug(f"Raw LLM response: {response}")
            
            # Extract JSON from response text
            response_text = response.get('text', '')
            if not response_text:
                raise ValueError("Empty response from vision model")
                
            # Try to extract JSON from the response
            try:
                # Look for JSON in the response text
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    data = json.loads(json_str)
                else:
                    # If no JSON brackets found, try parsing the entire response
                    data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from response: {response_text}")
                raise ValueError(f"Invalid JSON in vision response: {str(e)}")
            
            # Validate and normalize response structure
            if not isinstance(data, dict):
                raise ValueError("Response must be a JSON object")
                
            # Extract dish name
            dish_name = data.get('dish_name', data.get('name', 'Unknown Dish'))
            if not isinstance(dish_name, str):
                dish_name = str(dish_name)
                
            # Extract and normalize ingredients
            ingredients_raw = data.get('ingredients', [])
            if not isinstance(ingredients_raw, list):
                logger.warning(f"Ingredients not a list: {ingredients_raw}")
                ingredients_raw = []
                
            normalized_ingredients = []
            for i, ingredient in enumerate(ingredients_raw):
                try:
                    if isinstance(ingredient, dict):
                        # Standard format: {"name": "rice", "quantity": 200}
                        name = ingredient.get('name', ingredient.get('ingredient', f'unknown_{i}'))
                        quantity = ingredient.get('quantity', ingredient.get('amount', ingredient.get('weight', 100)))
                        
                    elif isinstance(ingredient, str):
                        # String format: "rice - 200g" or just "rice"
                        name = ingredient
                        quantity = 100  # default quantity
                        
                    else:
                        logger.warning(f"Unexpected ingredient format: {ingredient}")
                        continue
                        
                    # Normalize name and quantity
                    if isinstance(name, str):
                        name = name.lower().strip()
                    else:
                        name = str(name).lower().strip()
                        
                    try:
                        quantity = float(quantity)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid quantity for {name}: {quantity}, using default 100g")
                        quantity = 100.0
                        
                    # Ensure positive quantity
                    if quantity <= 0:
                        quantity = 100.0
                        
                    normalized_ingredients.append({
                        "name": name,
                        "quantity": quantity
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing ingredient {ingredient}: {str(e)}")
                    continue
            
            # If no ingredients found, add a default one
            if not normalized_ingredients:
                logger.warning("No valid ingredients found, adding default")
                normalized_ingredients = [{"name": dish_name.lower(), "quantity": 200.0}]
                
            # Extract confidence
            confidence = data.get('confidence', 50)
            try:
                confidence = int(confidence)
                confidence = min(100, max(0, confidence))
            except (ValueError, TypeError):
                confidence = 50
                
            result = {
                "dish_name": dish_name.strip(),
                "ingredients": normalized_ingredients,
                "confidence": confidence
            }
            
            logger.info(f"Successfully parsed dish: {result['dish_name']} with {len(result['ingredients'])} ingredients")
            return result
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {str(e)}")
            # Return a fallback result instead of raising an exception
            return {
                "dish_name": "Unknown Dish",
                "ingredients": [{"name": "mixed food", "quantity": 200.0}],
                "confidence": 0,
                "error": str(e)
            }

    # food_analyser.py
    def get_nutrients_with_fallback(self, ingredient: str) -> Dict[str, float]:
        """Get nutrients with 3-level fallback"""
        # Level 1: Exact match
        try:
            nutrient_data = self.vector_db.get_nutrient_data(ingredient)
            if nutrient_data and nutrient_data.get('nutrients'):
                return nutrient_data['nutrients']
        except Exception as e:
            logger.warning(f"Error in exact match for {ingredient}: {str(e)}")
        
        # Level 2: Similar ingredient match
        try:
            similar_items = self.vector_db.search_similar(ingredient, k=1)
            if similar_items and similar_items[0].get('nutrients'):
                logger.info(f"Using similar ingredient {similar_items[0]['food_name']} for {ingredient}")
                return similar_items[0]['nutrients']
        except Exception as e:
            logger.warning(f"Error in similar match for {ingredient}: {str(e)}")
        
        # Level 3: LLM fallback
        if ingredient not in self.fallback_nutrients_cache:
            try:
                prompt = f"""Provide macronutrients per 100g for {ingredient} as JSON:
                {{
                    "carbohydrates": float,
                    "proteins": float,
                    "fats": float,
                    "fibre": float,
                    "calories": float
                }}"""
                
                response = self.client.invoke(prompt, use_cache=True)
                fallback_data = self.fallback_parser.parse_llm_response(response['text'])
                
                # Ensure all required nutrients are present and valid
                required_nutrients = ['carbohydrates', 'proteins', 'fats', 'fibre', 'calories']
                for nutrient in required_nutrients:
                    try:
                        # Handle both numeric and string values
                        value = fallback_data.get(nutrient, 0.0)
                        if isinstance(value, str):
                            # Extract first numeric value from strings like "10g"
                            if num_match := re.search(r'(\d+\.?\d*)', value):
                                fallback_data[nutrient] = float(num_match.group(1))
                            else:
                                fallback_data[nutrient] = 0.0
                        else:
                            fallback_data[nutrient] = float(value)
                    except (ValueError, TypeError):
                        fallback_data[nutrient] = 0.0
                    
                    # Ensure non-negative values
                    if fallback_data[nutrient] < 0:
                        fallback_data[nutrient] = 0.0
                
                self.fallback_nutrients_cache[ingredient] = fallback_data
                logger.warning(f"Used LLM fallback for: {ingredient}")
            except Exception as e:
                logger.error(f"Fallback failed for {ingredient}: {str(e)}")
                # Return reasonable defaults if all fallbacks fail
                self.fallback_nutrients_cache[ingredient] = {
                    'carbohydrates': 15.0,
                    'proteins': 5.0,
                    'fats': 3.0,
                    'fibre': 2.0,
                    'calories': 80.0
                }
        
        return self.fallback_nutrients_cache[ingredient]

    def calculate_total_nutrients(self, ingredients: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Calculate total nutrients without cooking adjustments"""
        totals = {k: 0.0 for k in ['carbohydrates', 'proteins', 'fats', 'fibre', 'calories']}
        ingredient_details = []
        
        for name, qty in ingredients:
            try:
                # Get nutrients per 100g
                nutrients_per_100g = self.get_nutrients_with_fallback(name)
                
                # Ensure all values are floats
                for nutrient, value in nutrients_per_100g.items():
                    if isinstance(value, str):
                        try:
                            nutrients_per_100g[nutrient] = float(value)
                        except ValueError:
                            nutrients_per_100g[nutrient] = 0.0
                
                # Calculate for actual quantity
                for nutrient, value in nutrients_per_100g.items():
                    total_for_ingredient = float(value) * (qty / 100)
                    totals[nutrient] += total_for_ingredient
                
                # Store individual ingredient details for reporting
                ingredient_details.append({
                    'name': name,
                    'quantity': qty,
                    'nutrients': nutrients_per_100g,
                    'total_nutrients': {
                        k: float(v) * (qty / 100)
                        for k, v in nutrients_per_100g.items()
                    }
                })
            except Exception as e:
                logger.error(f"Error processing {name}: {str(e)}")
                # Use default values for this ingredient
                defaults = {
                    'carbohydrates': 15.0,
                    'proteins': 5.0,
                    'fats': 3.0,
                    'fibre': 2.0,
                    'calories': 80.0
                }
                for nutrient, value in defaults.items():
                    totals[nutrient] += value * (qty / 100)
                
                ingredient_details.append({
                    'name': name,
                    'quantity': qty,
                    'error': str(e),
                    'nutrients': defaults,
                    'total_nutrients': {
                        k: v * (qty / 100)
                        for k, v in defaults.items()
                    }
                })
                    
        return {
            'totals': {k: round(v, 2) for k, v in totals.items()},
            'ingredient_details': ingredient_details
        }


    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main analysis pipeline"""
        try:
            # Step 1: Image analysis
            dish_data = self.get_dish_details_from_image(data['food_data'])
            
            # Step 2: Calculate nutrients
            ingredients = [(i['name'], i['quantity']) for i in dish_data['ingredients']]
            nutrient_results = self.calculate_total_nutrients(ingredients)
            
            # Ensure we have all required keys
            if 'totals' not in nutrient_results:
                nutrient_results['totals'] = {
                    'carbohydrates': 0.0,
                    'proteins': 0.0,
                    'fats': 0.0,
                    'fibre': 0.0,
                    'calories': 0.0
                }
            if 'ingredient_details' not in nutrient_results:
                nutrient_results['ingredient_details'] = []
            
            # Step 3: Compare to requirements
            reqs = data['req_data']
            deviation = {k: nutrient_results['totals'][k] - reqs.get(k, 0) 
                        for k in nutrient_results['totals']}
            
            return {
                "dish_name": dish_data['dish_name'],
                "confidence": dish_data['confidence'],
                "ingredients": ingredients,
                "current_nutrients": nutrient_results['totals'],
                "req_nutrients": reqs,
                "deviation": deviation,
                "used_fallback": list(self.fallback_nutrients_cache.keys())
            }
            
        except FoodAnalyserError as e:
            return {
                "error": str(e),
                "stage": "food_analysis",
                "ingredient_details": []
            }
        except Exception as e:
            logger.exception("Analysis pipeline failed")
            return {
                "error": f"Unexpected error: {str(e)}",
                "ingredient_details": [],
                "current_nutrients": {
                    'carbohydrates': 0.0,
                    'proteins': 0.0,
                    'fats': 0.0,
                    'fibre': 0.0,
                    'calories': 0.0
                }
            }