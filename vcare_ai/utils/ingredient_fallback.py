# ingredient_fallback.py
import re
import json
from typing import Dict

class IngredientFallback:
    NUTRIENT_PATTERNS = {
        'carbohydrates': r'carbs?|carbohydrates|sugars?',
        'proteins': r'proteins?',
        'fats': r'fats?|lipids',
        'fibre': r'fib(er|re)|dietary fiber',
        'calories': r'calories|energy'
    }
    
    def parse_llm_response(self, text: str) -> Dict[str, float]:
        """Extract nutrients from unstructured LLM response"""
        nutrients = {}
        text = text.lower()
        
        for nutrient, pattern in self.NUTRIENT_PATTERNS.items():
            if matches := re.search(rf"{pattern}[:]?\s*(\d+\.?\d*)", text):
                nutrients[nutrient] = float(matches.group(1))
        
        return nutrients