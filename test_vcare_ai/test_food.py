from vcare_ai.usecases.food_analyser import FoodAnalyser
import base64
import os

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

analyzer = FoodAnalyser()
result = analyzer.run({"image_data": "https://www.plattershare.com/storage/2021/09/5b0e7647d2c64-1.jpg"})
print(result)