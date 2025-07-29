# tests/test_food_report.py
import pytest
from unittest.mock import MagicMock, patch
from vcare_ai.usecases.food_analyser import FoodAnalyser

@patch('vcare_ai.food_report.FoodAnalyser')
@patch('vcare_ai.food_report.ImageProcessor')
@patch('vcare_ai.food_report.NutrientRequirements')
@patch('vcare_ai.food_report.ReportGenerator')
def test_main_success(mock_report, mock_nutrients, mock_image, mock_analyser):
    mock_image.optimize_image_to_base64.return_value = "base64image"
    mock_nutrients.load_from_db.return_value = {"proteins": 50}
    mock_analyser.return_value.run.return_value = {
        "dish_name": "Pizza",
        "confidence": 80,
        "current_nutrients": {"proteins": 20}
    }
    
    from vcare_ai.food_report import main
    main()
    
    mock_report.save_report.assert_called()
    mock_report.print_summary.assert_called()

@patch('vcare_ai.food_report.ImageProcessor')
def test_image_processor_success(mock_requests):
    from vcare_ai.food_report import ImageProcessor
    with patch('PIL.Image.open'), patch('io.BytesIO'):
        result = ImageProcessor.optimize_image_to_base64("valid_url")
        assert result != ""

def test_image_processor_failure():
    from vcare_ai.food_report import ImageProcessor
    result = ImageProcessor.optimize_image_to_base64("invalid_url")
    assert result == ""

@patch('vcare_ai.food_report.psycopg2.connect')
def test_nutrient_requirements_db(mock_connect):
    from vcare_ai.food_report import NutrientRequirements
    mock_connect.return_value.cursor.return_value.fetchone.return_value = [
        '{"proteins": 60}'
    ]
    result = NutrientRequirements.load_from_db("user123")
    assert result["proteins"] == 60

def test_nutrient_requirements_default():
    from vcare_ai.food_report import NutrientRequirements
    result = NutrientRequirements.load_defaults()
    assert "proteins" in result