import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class RecipeRetriever:
    def __init__(self, dataset_path="Data/final_food_dataset.csv"):
        self.df = pd.read_csv(dataset_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._create_embeddings()
    
    def _create_embeddings(self):
        # Create embeddings if not exists
        embed_path = "Data/recipe_embeddings.npy"
        if not os.path.exists(embed_path):
            dish_names = self.df['Name'].fillna("") + " " + self.df['Ingredients'].fillna("")
            self.embeddings = self.model.encode(dish_names.tolist())
            np.save(embed_path, self.embeddings)
        else:
            self.embeddings = np.load(embed_path)
    
    def retrieve_recipe(self, dish_name: str, top_k=1) -> dict:
        query_embed = self.model.encode([dish_name])
        similarities = cosine_similarity(query_embed, self.embeddings)
        top_idx = np.argmax(similarities)
        
        return {
            "name": self.df.iloc[top_idx]['Name'],
            "ingredients": self.df.iloc[top_idx]['Ingredients'],
            "serving_size": "Standard portion"  # Could be extended
        }