# tests/test_food_analyser.py
import pytest
import json
from unittest.mock import MagicMock, patch
from vcare_ai.usecases.food_analyser import FoodAnalyser, FoodAnalyserError

@pytest.fixture
def mock_client():
    return MagicMock()

@pytest.fixture
def food_analyser(mock_client):
    with patch('vcare_ai.usecases.food_analyser.VectorDBUtils') as mock_db:
        mock_db.return_value = MagicMock()
        return FoodAnalyser(client=mock_client)

def test_food_analyser_init_success(mock_client):
    with patch('vcare_ai.usecases.food_analyser.VectorDBUtils') as mock_db:
        mock_db.return_value = MagicMock()
        analyser = FoodAnalyser(client=mock_client)
        assert analyser.vector_db is not None
        assert analyser.fallback_parser is not None

def test_food_analyser_init_failure(mock_client):
    with patch('vcare_ai.usecases.food_analyser.VectorDBUtils', side_effect=Exception("DB error")):
        with pytest.raises(FoodAnalyserError):
            FoodAnalyser(client=mock_client)

def test_get_dish_details_valid_json(food_analyser, mock_client):
    mock_response = {'text': json.dumps({
        "dish_name": "Pasta",
        "ingredients": [{"name": "pasta", "quantity": 200}],
        "confidence": 85
    })}
    mock_client.invoke.return_value = mock_response
    
    result = food_analyser.get_dish_details_from_image("base64image")
    assert result["dish_name"] == "Pasta"
    assert len(result["ingredients"]) == 1
    assert result["confidence"] == 85

def test_get_dish_details_invalid_json(food_analyser, mock_client):
    mock_client.invoke.return_value = {'text': "Invalid response"}
    result = food_analyser.get_dish_details_from_image("base64image")
    assert result["dish_name"] == "Unknown Dish"

def test_get_nutrients_exact_match(food_analyser):
    food_analyser.vector_db.get_nutrient_data.return_value = {
        'nutrients': {'proteins': 10}
    }
    result = food_analyser.get_nutrients_with_fallback("chicken")
    assert result['proteins'] == 10

def test_get_nutrients_similar_match(food_analyser):
    food_analyser.vector_db.get_nutrient_data.return_value = None
    food_analyser.vector_db.search_similar.return_value = [
        {'food_name': 'chicken', 'nutrients': {'proteins': 8}}
    ]
    result = food_analyser.get_nutrients_with_fallback("chkn")
    assert result['proteins'] == 8

def test_get_nutrients_llm_fallback(food_analyser, mock_client):
    food_analyser.vector_db.get_nutrient_data.return_value = None
    food_analyser.vector_db.search_similar.return_value = []
    
    # Test different response formats
    response_formats = [
        # JSON response
        {'text': json.dumps({"proteins": 7.5, "carbohydrates": 12})},
        # Free-text response
        {'text': "Proteins: 7.2g, Carbs: 12.5g, Fats: 5.1g"},
        # Messy response
        {'text': "Nutrition: proteins=7.3, carbs:12.2, fats 5.2"}
    ]
    
    for response in response_formats:
        mock_client.invoke.return_value = response
        food_analyser.fallback_nutrients_cache = {}  # Reset cache
        
        result = food_analyser.get_nutrients_with_fallback("unicorn meat")
        # Should be around 7 with some tolerance
        assert 7.0 <= result['proteins'] <= 7.5
        assert "unicorn meat" in food_analyser.fallback_nutrients_cache

def test_calculate_total_nutrients(food_analyser):
    with patch.object(food_analyser, 'get_nutrients_with_fallback') as mock_fallback:
        mock_fallback.return_value = {
            'proteins': 10, 'carbohydrates': 20, 
            'fats': 5, 'fibre': 3, 'calories': 150
        }
        result = food_analyser.calculate_total_nutrients([("chicken", 200)])
        assert result['totals']['proteins'] == 20.0
        assert len(result['ingredient_details']) == 1

def test_run_success(food_analyser):
    with patch.object(food_analyser, 'get_dish_details_from_image') as mock_dish:
        mock_dish.return_value = {
            "dish_name": "Salad",
            "ingredients": [{"name": "lettuce", "quantity": 100}],
            "confidence": 90
        }
        with patch.object(food_analyser, 'calculate_total_nutrients') as mock_nutrients:
            mock_nutrients.return_value = {
                'totals': {'proteins': 5},
                'ingredient_details': []
            }
            result = food_analyser.run({
                "food_data": "base64img",
                "req_data": {"proteins": 10}
            })
            assert result["dish_name"] == "Salad"
            assert result["deviation"]["proteins"] == -5