from vcare_ai.usecases.food_analyser import FoodAnalyser
import base64
import os

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

analyzer = FoodAnalyser()
result = analyzer.run({"image_data": encode_image("/home/satyamkumarjha/Traning/Datasets/Day1/images/IMG-20250401-WA0011.jpg")})
print(result)