# read the csv indian_food.csv and then only keep the name and ingredients columns
#the ingredients column is in  a format like this: "Maida, Sugar, Ghee, Milk, Cardamom"
# we need to split the ingredients by comma and strip any whitespace
#and then save thes new data set as name food_data_1.csv

import re
import pandas as pd
import ast
import os

def clean_ingredient(ingredient):
    """
    Cleans a single ingredient string by removing parentheses, punctuation,
    normalizing whitespace, and converting to lowercase.
    """
    if not isinstance(ingredient, str):
        return ''
    ingredient = re.sub(r'\([^)]*\)', '', ingredient)  # Remove anything in parentheses
    ingredient = re.sub(r'[^a-zA-Z0-9\s]', '', ingredient)  # Remove punctuation
    ingredient = re.sub(r'\s+', ' ', ingredient)  # Normalize whitespace
    return ingredient.strip().lower()

def clean_ingredients_column(df, col='ingredients'):
    """
    Cleans the ingredients column in the dataframe.
    """
    cleaned_ingredients = []
    for row in df[col]:
        try:
            # Convert string to list using ast.literal_eval
            ingredients = ast.literal_eval(row)
        except:
            ingredients = []
        cleaned_list = [clean_ingredient(i) for i in ingredients]
        cleaned_ingredients.append(cleaned_list)
    df['cleaned_ingredients'] = cleaned_ingredients
    return df

print("Step 1: Reading 'indian_food.csv' and cleaning ingredients...")
data = pd.read_csv('Data/indian_food.csv')
data = data[['name', 'ingredients']]

# Split ingredients by comma and strip whitespace
data['ingredients'] = data['ingredients'].apply(lambda x: [ingredient.strip() for ingredient in x.split(',')])

# Clean ingredients
data = clean_ingredients_column(data, col='ingredients')
data = data[['name', 'ingredients']]

# Save the cleaned data
data.to_csv('Data/food_data_1.csv', index=False)
print("Saved cleaned data to 'food_data_1.csv'.")

print("Step 2: Reading 'Cleaned_Indian_Food_Dataset.csv' and cleaning ingredients...")
data2 = pd.read_csv('Data/Cleaned_Indian_Food_Dataset.csv')
data2 = data2[['TranslatedRecipeName', 'Cleaned-Ingredients']]
data2.columns = ['Name', 'Ingredients']

# Split and clean ingredients
data2['Ingredients'] = data2['Ingredients'].apply(lambda x: [clean_ingredient(ing.strip()) for ing in str(x).split(',')])

# Save cleaned data
data2.to_csv('Data/food_data_2.csv', index=False)
print("Saved cleaned data to 'food_data_2.csv'.")

print("Step 3: Loading both cleaned datasets for merging...")
df1 = pd.read_csv('Data/food_data_1.csv') 
df2 = pd.read_csv('Data/food_data_2.csv')  

# Ensure columns are named correctly
df1.columns = ['Name', 'Ingredients']
df2.columns = ['Name', 'Ingredients']

# Convert stringified lists into comma-separated strings in df2
df2['Ingredients'] = df2['Ingredients'].apply(lambda x: ', '.join(ast.literal_eval(x)))

# Optional: Do the same to df1 if needed
if isinstance(df1['Ingredients'].iloc[0], str) and df1['Ingredients'].iloc[0].startswith('['):
    df1['Ingredients'] = df1['Ingredients'].apply(lambda x: ', '.join(ast.literal_eval(x)))

print("Step 4: Merging datasets...")
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save final merged file
merged_df.to_csv('Data/final_food_dataset.csv', index=False)
print("Saved merged dataset to 'final_food_dataset.csv'.")

print("Step 5: Cleaning up temporary files...")
os.remove('Data/food_data_1.csv')
os.remove('Data/food_data_2.csv')
print("Temporary files deleted.")

print("Data cleaning and merging completed successfully.")


