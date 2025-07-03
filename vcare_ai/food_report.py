from vcare_ai.usecases.food_analyser import FoodAnalyser
import os
import sys
from PIL import Image
import requests
from io import BytesIO
import base64

def optimize_image_to_base64(url: str, max_dim: int = 384) -> str:
    """Download, resize, and compress the image aggressively before base64 encoding."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    image = Image.open(BytesIO(response.content))
    image = image.convert("RGB")
    image.thumbnail((max_dim, max_dim))  # Resize to smaller square

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=30, optimize=True)  # Aggressive compression
    base64_bytes = base64.b64encode(buffer.getvalue())
    return base64_bytes.decode("utf-8")
    
req_data = {
    "carbohydrates": 200,
    "proteins": 150,
    "fats": 70,
    "fibre": 30,
    "calories": 2500
}

analyzer = FoodAnalyser()
result = analyzer.run({"food_data": optimize_image_to_base64("https://vegecravings.com/wp-content/uploads/2024/07/Dosa-Recipe-Step-By-Step-Instructions-scaled.jpg"), "req_data": req_data})
print(result)