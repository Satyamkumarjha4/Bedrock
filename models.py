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

# Declare the SQLAlchemy base
Base = declarative_base()

# -----------------------------------
# ✅ Define PostgreSQL pgvector type
# -----------------------------------
class Vector(UserDefinedType):
    def get_col_spec(self):
        return "vector(384)"  # Match dimension with your SentenceTransformer model


# -------------------------------------------------
# 1. 🍽️ FoodNutrient: main nutrient + embedding table
#     - used by ingest_food_data.py to insert embedded food records
# -------------------------------------------------
class FoodNutrient(Base):
    __tablename__ = "food_nutrients"

    id = Column(Integer, primary_key=True)
    food_name = Column(Text, nullable=False)
    description = Column(Text)
    nutrients = Column(JSON, nullable=False)
    embedding = Column(Vector)  # 🔥 pgvector embedding (dim=384)
    source = Column(String(50))
    last_updated = Column(TIMESTAMP, server_default=func.now())

    def __repr__(self):
        return f"<FoodNutrient(id={self.id}, name={self.food_name})>"


# --------------------------------------------------
# 2. 🧂 IngredientMapping: maps variants to canonical name
# --------------------------------------------------
class IngredientMapping(Base):
    __tablename__ = "ingredient_mapping"

    id = Column(Integer, primary_key=True)
    canonical_name = Column(Text, nullable=False)
    variants = Column(ARRAY(Text))  # 🧠 PostgreSQL array type
    category = Column(String(50))


# --------------------------------------------------
# 3. 👤 UserProfile: stores per-user preferences or history
# --------------------------------------------------
class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id = Column(String(50), primary_key=True)
    nutrients = Column(JSON)
    last_updated = Column(TIMESTAMP, server_default=func.now())


# --------------------------------------------------
# 4. 🧾 FoodItem: stores basic macro info (not embedded)
# --------------------------------------------------
class FoodItem(Base):
    __tablename__ = "food_items"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    calories = Column(Float)
    protein = Column(Float)
    carbs = Column(Float)
    fats = Column(Float)
