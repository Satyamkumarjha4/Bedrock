# tests/test_ingredient_fallback.py
import pytest
from vcare_ai.utils.ingredient_fallback import IngredientFallback

@pytest.mark.parametrize("input_text, expected", [
    ("Proteins: 25g Fats: 10g", {"proteins": 25.0, "fats": 10.0}),
    ("Carbs 30, Calories: 200", {"carbohydrates": 30.0, "calories": 200.0}),
    ('{"carbohydrates": 45, "fibre": 5}', {"carbohydrates": 45.0, "fibre": 5.0}),
    ("Calories=300 Fibre: 8g", {"calories": 300.0, "fibre": 8.0}),
    ("Energy 250 kcal Sugars: 15", {"calories": 250.0, "carbohydrates": 15.0})
])
def test_parse_llm_response(input_text, expected):
    parser = IngredientFallback()
    result = parser.parse_llm_response(input_text)
    assert result == expected