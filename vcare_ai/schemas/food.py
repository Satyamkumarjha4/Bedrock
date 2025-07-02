from pydantic import BaseModel

class Food(BaseModel):
    food_image: str #image in base64 or url
    req_data: dict

class NutrientResult(BaseModel):
    carbohydrates: float
    proteins: float
    fats: float
    fibre: float
    calories: float

class Deviation(BaseModel):
    carbohydrates: float
    proteins: float
    fats: float
    fibre: float
    calories: float

class FoodAnalysis(BaseModel):
    food: Food
    nutrient_result: NutrientResult
    deviation: Deviation

