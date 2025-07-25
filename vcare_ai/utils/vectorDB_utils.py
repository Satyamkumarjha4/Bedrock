# vectorDB_utils.py
import re
import json
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
from psycopg2.extras import execute_values
from ..config import DB_CONFIG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBUtils:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        try:
            self._create_extension()
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._create_tables()
            self._prepare_similarity_search()
        except Exception as e:
            self.conn.close()
            raise RuntimeError(f"Failed to initialize VectorDBUtils: {str(e)}")

    def _create_extension(self):
        """Create required PostgreSQL extensions"""
        with self.conn.cursor() as cur:
            extensions = ['vector', 'pg_trgm']
            for ext in extensions:
                try:
                    cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext};")
                    self.conn.commit()
                except Exception as e:
                    logger.error(f"Error creating extension {ext}: {e}")
                    self.conn.rollback()
                    raise

    def _create_tables(self):
        """Initialize database schema"""
        with self.conn.cursor() as cur:
            # Main nutrients table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS food_nutrients (
                    id SERIAL PRIMARY KEY,
                    food_name TEXT NOT NULL,
                    description TEXT,
                    nutrients JSONB NOT NULL,
                    embedding VECTOR(384),
                    source VARCHAR(50),
                    last_updated TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Ingredients mapping table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ingredient_mapping (
                    id SERIAL PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    variants TEXT[],
                    category VARCHAR(50)
                );
            """)
            
            # User profiles table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id VARCHAR(50) PRIMARY KEY,
                    nutrients JSONB,
                    last_updated TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_food_name_trgm 
                ON food_nutrients USING GIN (food_name gin_trgm_ops);
            """)
            self.conn.commit()

    def _prepare_similarity_search(self):
        """Prepare similarity search functions"""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE OR REPLACE FUNCTION ingredient_search(query TEXT)
                RETURNS TABLE (
                    food_name TEXT,
                    nutrients JSONB,
                    similarity FLOAT
                ) AS $$
                BEGIN
                    RETURN QUERY
                    SELECT 
                        fn.food_name,
                        fn.nutrients,
                        SIMILARITY(fn.food_name, query)::FLOAT AS similarity 
                    FROM food_nutrients fn
                    LEFT JOIN ingredient_mapping im ON fn.food_name = im.canonical_name
                    WHERE 
                        fn.food_name % query OR
                        query = ANY(im.variants)
                    ORDER BY similarity DESC
                    LIMIT 1;
                END;
                $$ LANGUAGE plpgsql;
            """)
            self.conn.commit()

    def extract_text_and_tables(self, pdf_path: str) -> List[str]:
        """Extract text and tables from PDF documents"""
        chunks = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text()
                chunks.append(text)
                tables = re.findall(
                    r'(\b\w+\b\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+)',
                    text
                )
                chunks.extend(tables)
        return chunks

    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generate sentence embeddings"""
        return self.embedding_model.encode([text])[0]

    def ingest_data(self, pdf_path: str):
        """Ingest data from PDF source"""
        chunks = self.extract_text_and_tables(pdf_path)
        with self.conn.cursor() as cur:
            for chunk in chunks:
                food_match = re.search(
                    r'(\b\w+\b)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)',
                    chunk
                )
                if food_match:
                    food_name = food_match.group(1)
                    nutrients = {
                        'carbohydrates': float(food_match.group(2)),
                        'proteins': float(food_match.group(3)),
                        'fats': float(food_match.group(4)),
                        'fibre': float(food_match.group(5)),
                        'calories': int(food_match.group(6))
                    }
                    embedding = self.generate_embeddings(food_name)
                    cur.execute(
                        "INSERT INTO food_nutrients (food_name, nutrients, embedding, source) "
                        "VALUES (%s, %s, %s, %s)",
                        (food_name, json.dumps(nutrients), embedding.tolist(), "IFCT2017")
                    )
            self.conn.commit()

    def search_similar(self, query: str, k: int = 3, threshold: float = 0.4) -> List[Dict]:
        """Enhanced similarity search with threshold"""
        query_embed = self.generate_embeddings(query)
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    food_name,
                    nutrients,
                    1 - (embedding <=> %s) AS similarity
                FROM food_nutrients
                WHERE 1 - (embedding <=> %s) > %s
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_embed.tolist(), query_embed.tolist(), threshold, k))
            return [
                {
                    'food_name': row[0],
                    'nutrients': row[1],
                    'similarity': float(row[2])
                } for row in cur.fetchall()
            ]

    def get_nutrient_data(self, ingredient: str) -> Optional[Dict]:
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT * FROM ingredient_search(%s)", (ingredient,))
                if row := cur.fetchone():
                    return {
                        'food_name': row[0],
                        'nutrients': row[1],
                        'similarity': float(row[2])
                    }
        except Exception as e:
            logger.error(f"Error retrieving {ingredient}: {str(e)}")
            self.conn.rollback()  # Reset transaction state
        return None

    def get_recipe(self, dish_name: str) -> Optional[Dict]:
        """Retrieve complete recipe with ingredients"""
        results = self.search_similar(dish_name, k=1, threshold=0.5)
        if results:
            return {
                "name": results[0]['food_name'],
                "ingredients": results[0]['nutrients'].get('ingredients', ''),
                "serving_size": results[0]['nutrients'].get('serving_size', 'Standard portion'),
                "similarity": results[0]['similarity']
            }
        return None

    def get_ingredient_nutrients_bulk(self, ingredients: List[str]) -> Dict[str, Dict]:
        """Batch retrieve nutrients for multiple ingredients"""
        results = {}
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """SELECT search.* 
                FROM (VALUES %s) AS t(ingredient)
                CROSS JOIN LATERAL ingredient_search(t.ingredient) AS search""",
                [(ing,) for ing in ingredients],
                page_size=100
            )
            for row in cur.fetchall():
                results[row[0]] = row[1]  # food_name -> nutrients
        return results

    def add_ingredient_mapping(self, canonical_name: str, variants: List[str], category: str = None):
        """Add ingredient name variations"""
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ingredient_mapping (canonical_name, variants, category) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (canonical_name) DO UPDATE "
                "SET variants = EXCLUDED.variants, category = EXCLUDED.category",
                (canonical_name, variants, category)
            )
            self.conn.commit()


    def ingest_final_dataset(self, csv_path: str = "Data/final_food_dataset.csv"):
        """Ingest recipe dataset with proper embeddings"""
        df = pd.read_csv(csv_path)
        with self.conn.cursor() as cur:
            for _, row in df.iterrows():
                embedding = self.generate_embeddings(row['Name'])
                cur.execute(
                    "INSERT INTO food_nutrients (food_name, nutrients, embedding, source) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (food_name) DO UPDATE "
                    "SET nutrients = EXCLUDED.nutrients, embedding = EXCLUDED.embedding",
                    (
                        row['Name'],
                        json.dumps({
                            'ingredients': row['Ingredients'],
                            'serving_size': 'Standard portion'
                        }),
                        embedding.tolist(),
                        'RECIPE_DB'
                    )
                )
            self.conn.commit()