# ingredient_fallback.py
import re
import json
from typing import Dict

class IngredientFallback:
    NUTRIENT_PATTERNS = {
        'carbohydrates': r'carbs?|carbohydrates|sugars?|carb',
        'proteins': r'proteins?|prot',
        'fats': r'fats?|lipids|fat\s+content|lipid',
        'fibre': r'fib(?:er|re)|dietary\s+fib(?:er|re)|roughage',  # Fixed: non-capturing groups
        'calories': r'calories|energy|kcal|kilocalories'
    }
    
    def parse_llm_response(self, text: str) -> Dict[str, float]:
        """Robust nutrient extraction from LLM responses"""
        nutrients = {}
        text = text.lower()
        
        # First try to parse as JSON
        try:
            data = json.loads(text)
            for nutrient in self.NUTRIENT_PATTERNS:
                if nutrient in data:
                    try:
                        nutrients[nutrient] = float(data[nutrient])
                    except (ValueError, TypeError):
                        pass
            if nutrients:
                return nutrients
        except json.JSONDecodeError:
            pass
        
        # Pattern 0: Key=value format
        for nutrient, pattern in self.NUTRIENT_PATTERNS.items():
            if match := re.search(rf'\b({pattern})\b\s*=\s*(\d+\.?\d*)', text):
                nutrients[nutrient] = float(match.group(2))

        # Pattern 1: Key: value format (with colon/equals)
        for nutrient, pattern in self.NUTRIENT_PATTERNS.items():
            if match := re.search(rf'\b({pattern})\b\s*[:=]\s*(\d+\.?\d*)', text):
                nutrients[nutrient] = float(match.group(2))
        
        # Pattern 2: Value before key (with nutrient word boundary)
        for nutrient, pattern in self.NUTRIENT_PATTERNS.items():
            if nutrient in nutrients:  # Skip if already found
                continue
            if match := re.search(rf'\b(\d+\.?\d*)\s*(?:g|grams?|mg)?\s*\b({pattern})\b', text):
                nutrients[nutrient] = float(match.group(1))
        
        # Pattern 3: Key space value (without punctuation)
        for nutrient, pattern in self.NUTRIENT_PATTERNS.items():
            if nutrient in nutrients:  # Skip if already found
                continue
            if match := re.search(rf'\b({pattern})\b\s+(\d+\.?\d*)', text):
                nutrients[nutrient] = float(match.group(2))
        
        # Pattern 4: JSON-like format
        for nutrient, pattern in self.NUTRIENT_PATTERNS.items():
            if nutrient in nutrients:  # Skip if already found
                continue
            if match := re.search(rf'"{pattern}"\s*:\s*(\d+\.?\d*)', text):
                nutrients[nutrient] = float(match.group(1))
        
        return nutrients