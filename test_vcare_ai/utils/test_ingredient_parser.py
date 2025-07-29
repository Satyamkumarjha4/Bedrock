# tests/test_ingredient_parser.py
import pytest
from vcare_ai.utils.ingredient_parser import parse_quantity, parse_ingredients

@pytest.mark.parametrize("input_val, expected", [
    ("200g", 200),
    ("1.5kg", 1500),
    ("2 cups", 480),
    ("3 tbsp", 45),
    ("1/2 cup", 120)
])
def test_parse_quantity(input_val, expected):
    result = parse_quantity(input_val)
    assert result == expected

@pytest.mark.parametrize("input_str, expected", [
    ("Flour: 200g", [("flour", 200)]),
    ("Sugar 150g, Salt 5g", [("sugar", 150), ("salt", 5)]),
    ("2 eggs, 1 cup milk", [("eggs", 100), ("milk", 240)]),
    ("3 apples", [("apples", 150)])
])
def test_parse_ingredients(input_str, expected):
    result = parse_ingredients(input_str)
    assert len(result) == len(expected)
    for (res_name, res_qty), (exp_name, exp_qty) in zip(result, expected):
        assert res_name == exp_name
        assert res_qty == pytest.approx(exp_qty, abs=0.1)