
import os
import pandas as pd
from gensim.models import FastText

# Project paths
DATA_PATH = "data/processed/cleaned_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fasttext.model")

# Load the cleaned dataset
df = pd.read_csv(DATA_PATH)

# Remove missing and empty tweets
texts = df["clean_text"].fillna("").astype(str)
sentences = [
    text.split()
    for text in texts
    if text.strip()
]

# FastText training settings
model = FastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,
    min_n=3,
    max_n=6,
    bucket=50000,
    epochs=20,
    seed=42
)

# Save the trained model
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)

# Display training information
print("FastText training completed successfully!")
print("Total tweets:", len(df))
print("Training sentences:", len(sentences))
print("Vocabulary size:", len(model.wv.key_to_index))
print("Vector size:", model.vector_size)
print("Character n-grams:", model.wv.min_n, "to", model.wv.max_n)
print("Model saved at:", MODEL_PATH)

