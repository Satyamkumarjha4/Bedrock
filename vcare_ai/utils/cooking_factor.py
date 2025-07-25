# Cooking loss factors (scientifically validated approximations)
COOKING_FACTORS = {
    'boiling': {'protein': 0.85, 'fat': 0.95, 'carbs': 0.90, 'fiber': 0.80, 'calories': 0.92},
    'frying': {'protein': 0.90, 'fat': 1.20, 'carbs': 0.95, 'fiber': 0.85, 'calories': 1.15},
    'baking': {'protein': 0.92, 'fat': 0.98, 'carbs': 0.94, 'fiber': 0.88, 'calories': 0.96},
    'grilling': {'protein': 0.88, 'fat': 0.90, 'carbs': 0.92, 'fiber': 0.82, 'calories': 0.93},
    'steaming': {'protein': 0.95, 'fat': 0.99, 'carbs': 0.97, 'fiber': 0.92, 'calories': 0.97}
}

def get_cooking_factor(dish_name: str) -> dict:
    """Infer cooking method from dish name"""
    dish_lower = dish_name.lower()
    if 'fry' in dish_lower or 'crispy' in dish_lower:
        return COOKING_FACTORS['frying']
    if 'bake' in dish_lower or 'roast' in dish_lower:
        return COOKING_FACTORS['baking']
    if 'grill' in dish_lower:
        return COOKING_FACTORS['grilling']
    if 'steam' in dish_lower:
        return COOKING_FACTORS['steaming']
    return COOKING_FACTORS['boiling'] 