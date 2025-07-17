# food_report.py
from vcare_ai.usecases.food_analyser import FoodAnalyser
from vcare_ai.utils.ingredient_fallback import IngredientFallback
import os
import sys
from PIL import Image
import requests
from io import BytesIO
import base64
import json
import time
from vcare_ai.config import DB_CONFIG
import psycopg2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    @staticmethod
    def optimize_image_to_base64(url: str, max_dim: int = 768) -> str:
        """Download, resize, and compress the image before base64 encoding."""
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            image = image.convert("RGB")
            
            # Maintain aspect ratio while resizing
            width, height = image.size
            if width > height:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            else:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
                
            image = image.resize((new_width, new_height))
            
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=70, optimize=True)
            base64_bytes = base64.b64encode(buffer.getvalue())
            return base64_bytes.decode("utf-8")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return ""

class NutrientRequirements:
    @staticmethod
    def load_defaults() -> dict:
        """Load default nutrient requirements"""
        return {
            "carbohydrates": 200,
            "proteins": 60,
            "fats": 20,
            "fibre": 30,
            "calories": 1200
        }
    
    @staticmethod
    def load_from_db(user_id: str = None) -> dict:
        """Load personalized requirements from DB"""
        if not user_id:
            return NutrientRequirements.load_defaults()
            
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            with conn.cursor() as cur:
                # First check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'user_profiles'
                    );
                """)
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    return NutrientRequirements.load_defaults()
                    
                cur.execute(
                    "SELECT nutrients FROM user_profiles WHERE user_id = %s",
                    (user_id,))
                if result := cur.fetchone():
                    return json.loads(result[0])
            return NutrientRequirements.load_defaults()
        except Exception as e:
            logger.error(f"Failed to load nutrients from DB: {str(e)}")
            return NutrientRequirements.load_defaults()
    
    @staticmethod
    def load_from_db(user_id: str = None) -> dict:
        """Load personalized requirements from DB"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT nutrients FROM user_profiles WHERE user_id = %s",
                    (user_id,))
                if result := cur.fetchone():
                    return json.loads(result[0])
            return NutrientRequirements.load_defaults()
        except Exception as e:
            logger.error(f"Failed to load nutrients from DB: {str(e)}")
            return NutrientRequirements.load_defaults()

class ReportGenerator:
    @staticmethod
    def save_report(data: dict, filename: str = None) -> str:
        """Save analysis report to JSON file"""
        if not filename:
            filename = f"food_report_{int(time.time())}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            return filename
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")
            return ""

    @staticmethod
    def print_summary(result: dict):
        """Print human-readable summary"""
        if "error" in result:
            print(f"\nError: {result['error']}")
            return

        req_data = result.get('req_nutrients', {})
        
        print("\n=== FOOD ANALYSIS SUMMARY ===")
        print(f"Dish Name: {result.get('dish_name', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 0)}%")
        
        if used_fallback := result.get('used_fallback'):
            print(f"\n⚠ Used fallback for: {', '.join(used_fallback)}")

        if "ingredients" in result:
            print("\nIngredients (with quantities):")
            for ingredient, quantity in result["ingredients"]:
                print(f"- {ingredient}: {quantity}g")

        if "current_nutrients" in result:
            print("\nNutrient Analysis:")
            for nutrient, value in result["current_nutrients"].items():
                recommended = req_data.get(nutrient, "N/A")
                print(f"- {nutrient.capitalize()}: {value:.1f}g (Recommended: {recommended}g)")

        if "deviation" in result:
            print("\nDeviation from Recommendations:")
            for nutrient, diff in result["deviation"].items():
                status = "Above" if diff > 0 else "Below"
                print(f"- {nutrient.capitalize()}: {abs(diff):.1f}g {status} recommendation")

        if "recommendation" in result:
            print(f"\nRecommendation: {result['recommendation']}")

def main():
    # Configuration
    IMAGE_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTyrQe_3y27UhwkxGhuh7Sngv-rh2-jHYJBKg&s"
    USER_ID = None  # Replace with actual user ID for personalized requirements
    
    # Process image
    print("Processing food image...")
    image_base64 = ImageProcessor.optimize_image_to_base64(IMAGE_URL)
    
    if not image_base64:
        print("Failed to process image. Exiting.")
        return
    
    # Load nutrient requirements
    req_data = NutrientRequirements.load_from_db(USER_ID)
    
    # Create and run analyzer
    analyzer = FoodAnalyser()
    print("Analyzing food content...")
    
    try:
        result = analyzer.run({
            "food_data": image_base64, 
            "req_data": req_data
        })
        
        # Save and display results
        report_file = ReportGenerator.save_report(result)
        print(f"\nAnalysis complete! Report saved to {report_file}")
        ReportGenerator.print_summary(result)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        print(f"\nAnalysis failed: {str(e)}")

if __name__ == "__main__":
    main()