# food_analyser.py
import logging
import json
import re
from typing import Dict, Any, List, Tuple, Optional
from vcare_ai.client import BedrockClient
from vcare_ai.usecases.base import UseCase
from vcare_ai.utils.vectorDB_utils import VectorDBUtils
from vcare_ai.utils.ingredient_parser import parse_ingredients
from vcare_ai.utils.cooking_factor import get_cooking_factor
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
            data = json.loads(response['text'])
            
            # Validate response structure
            if not all(k in data for k in ['dish_name', 'ingredients']):
                raise ValueError("Missing required fields in vision response")
                
            return {
                "dish_name": data['dish_name'].strip(),
                "ingredients": [
                    {"name": i['name'].lower().strip(), 
                     "quantity": float(i['quantity'])}
                    for i in data['ingredients']
                ],
                "confidence": min(100, max(0, data.get('confidence', 0)))}
        except Exception as e:
            logger.error(f"Vision analysis failed: {str(e)}")
            raise FoodAnalyserError("Could not analyze food image")

    def get_nutrients_with_fallback(self, ingredient: str) -> Dict[str, float]:
        """
        Get nutrients with 3-level fallback:
        1. Exact match in VectorDB
        2. Similar ingredient in VectorDB
        3. LLM estimation
        
        Returns nutrient data per 100g
        """
        # Level 1: Exact match
        nutrient_data = self.vector_db.get_nutrient_data(ingredient)
        if nutrient_data and nutrient_data.get('nutrients'):
            return nutrient_data['nutrients']
            
        # Level 2: Similar ingredient match
        similar_items = self.vector_db.search_similar(ingredient, k=1)
        if similar_items and similar_items[0].get('nutrients'):
            logger.info(f"Using similar ingredient {similar_items[0]['food_name']} for {ingredient}")
            return similar_items[0]['nutrients']
            
        # Level 3: LLM fallback
        if ingredient not in self.fallback_nutrients_cache:
            prompt = f"""Provide macronutrients per 100g for {ingredient} as JSON:
            {{
                "carbohydrates": float,
                "proteins": float,
                "fats": float,
                "fibre": float,
                "calories": float
            }}"""
            
            try:
                response = self.client.invoke(prompt, use_cache=True)
                fallback_data = self.fallback_parser.parse_llm_response(response['text'])
                # Ensure all required nutrients are present
                required_nutrients = ['carbohydrates', 'proteins', 'fats', 'fibre', 'calories']
                for nutrient in required_nutrients:
                    if nutrient not in fallback_data:
                        fallback_data[nutrient] = 0.0
                self.fallback_nutrients_cache[ingredient] = fallback_data
                logger.warning(f"Used LLM fallback for: {ingredient}")
            except Exception as e:
                logger.error(f"Fallback failed for {ingredient}: {str(e)}")
                # Return zero values if all fallbacks fail
                self.fallback_nutrients_cache[ingredient] = {
                    'carbohydrates': 0.0,
                    'proteins': 0.0,
                    'fats': 0.0,
                    'fibre': 0.0,
                    'calories': 0.0
                }
        
        return self.fallback_nutrients_cache[ingredient]

    def calculate_total_nutrients(self, ingredients: List[Tuple[str, float]], 
                                cooking_factors: Dict[str, float]) -> Dict[str, float]:
        """Calculate total nutrients with cooking adjustments"""
        totals = {k: 0.0 for k in ['carbohydrates', 'proteins', 'fats', 'fibre', 'calories']}
        ingredient_details = []
        
        for name, qty in ingredients:
            try:
                # Get nutrients per 100g
                nutrients_per_100g = self.get_nutrients_with_fallback(name)
                
                # Calculate for actual quantity and apply cooking factors
                for nutrient, value in nutrients_per_100g.items():
                    adjusted = float(value) * cooking_factors.get(nutrient, 1.0) * (qty / 100)
                    totals[nutrient] += adjusted
                
                # Store individual ingredient details for reporting
                ingredient_details.append({
                    'name': name,
                    'quantity': qty,
                    'nutrients': nutrients_per_100g,
                    'adjusted_nutrients': {
                        k: float(v) * cooking_factors.get(k, 1.0) * (qty / 100)
                        for k, v in nutrients_per_100g.items()
                    }
                })
            except Exception as e:
                logger.error(f"Skipping {name}: {str(e)}")
                ingredient_details.append({
                    'name': name,
                    'quantity': qty,
                    'error': str(e)
                })
                
        return {
            'totals': {k: round(v, 2) for k, v in totals.items()},
            'ingredient_details': ingredient_details
        }

    def generate_recommendation(self, deviation: Dict[str, float]) -> str:
        """Generate actionable nutrition advice"""
        advice = []
        for nutrient, diff in deviation.items():
            if diff > 0:
                advice.append(f"reduce {nutrient} by {abs(diff):.1f}g")
            elif diff < 0:
                advice.append(f"increase {nutrient} by {abs(diff):.1f}g")
                
        return "Consider to " + ", ".join(advice) if advice else "Nutrition targets met"

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main analysis pipeline"""
        try:
            # Step 1: Image analysis
            dish_data = self.get_dish_details_from_image(data['food_data'])
            
            # Step 2: Get cooking factors
            factors = get_cooking_factor(dish_data['dish_name'])
            
            # Step 3: Calculate nutrients
            ingredients = [(i['name'], i['quantity']) for i in dish_data['ingredients']]
            nutrient_results = self.calculate_total_nutrients(ingredients, factors)
            
            # Step 4: Compare to requirements
            reqs = data['req_data']
            deviation = {k: nutrient_results['totals'][k] - reqs.get(k, 0) 
                        for k in nutrient_results['totals']}
            
            return {
                "dish_name": dish_data['dish_name'],
                "confidence": dish_data['confidence'],
                "ingredients": ingredients,
                "ingredient_details": nutrient_results['ingredient_details'],
                "cooking_method": next(iter(factors.keys()), "Unknown"),
                "current_nutrients": nutrient_results['totals'],
                "req_nutrients": reqs,
                "deviation": deviation,
                "recommendation": self.generate_recommendation(deviation),
                "used_fallback": list(self.fallback_nutrients_cache.keys())
            }
            
        except FoodAnalyserError as e:
            return {"error": str(e), "stage": "food_analysis"}
        except Exception as e:
            logger.exception("Analysis pipeline failed")
            return {"error": f"Unexpected error: {str(e)}"}