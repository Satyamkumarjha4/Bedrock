# ingest_food_data.py (updated)
from vcare_ai.utils.vectorDB_utils import VectorDBUtils
import json
import pandas as pd
import numpy as np

if __name__ == "__main__":
    vdb = VectorDBUtils()
    
    # Ingest IFCT2017 data
    vdb.ingest_data("Data/IFCT2017.pdf")
    
    # Ingest final recipe dataset
    df = pd.read_csv("Data/final_food_dataset.csv")
    with vdb.conn.cursor() as cur:
        for _, row in df.iterrows():
            # Generate embedding and convert to PostgreSQL vector format
            embedding = vdb.generate_embeddings(row['Name'])
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'  # Convert to string
            
            cur.execute(
                "INSERT INTO food_nutrients (food_name, nutrients, embedding, source) "
                "VALUES (%s, %s, %s, 'vectordb')",
                (
                    row['Name'],
                    json.dumps({
                        'ingredients': row['Ingredients'],
                        'serving_size': 'Standard portion'
                    }),
                    embedding_str  # Use the converted string
                )
            )
        vdb.conn.commit()
    
    print("All data ingested successfully!")