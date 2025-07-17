from vcare_ai.utils.vectorDB_utils import VectorDBUtils
from vcare_ai.config import DB_CONFIG
import psycopg2

def setup_database():
    # First connect without vector operations to create extension
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        print("Vector extension created successfully")
    except Exception as e:
        print(f"Error creating extension: {e}")
    finally:
        conn.close()
    
    # Now initialize the vector DB
    try:
        VectorDBUtils()
        print("Database tables initialized successfully")
    except Exception as e:
        print(f"Error initializing tables: {e}")

if __name__ == "__main__":
    setup_database()