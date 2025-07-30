from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    JSON,
    TIMESTAMP,
    func
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import UserDefinedType

Base = declarative_base()

# ✅ Define custom Vector type for pgvector
class Vector(UserDefinedType):
    def get_col_spec(self):
        return "vector(384)"  # Dimension can be 384 or 768 depending on your embedding model


# ---------- 1. Main nutrients table ----------
class FoodNutrient(Base):
    __tablename__ = "food_nutrients"

    id = Column(Integer, primary_key=True)
    food_name = Column(Text, nullable=False)
    description = Column(Text)
    nutrients = Column(JSON, nullable=False)
    embedding = Column(Vector)  # ✅ custom Vector type
    source = Column(String(50))
    last_updated = Column(TIMESTAMP, server_default=func.now())


# ---------- 2. Ingredient mapping ----------
class IngredientMapping(Base):
    __tablename__ = "ingredient_mapping"

    id = Column(Integer, primary_key=True)
    canonical_name = Column(Text, nullable=False)
    variants = Column(ARRAY(Text))  # ✅ PostgreSQL-specific
    category = Column(String(50))


# ---------- 3. User profile preferences ----------
class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id = Column(String(50), primary_key=True)
    nutrients = Column(JSON)
    last_updated = Column(TIMESTAMP, server_default=func.now())


# ---------- 4. Basic food item table ----------
class FoodItem(Base):
    __tablename__ = "food_items"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    calories = Column(Float)
    protein = Column(Float)
    carbs = Column(Float)
    fats = Column(Float)
