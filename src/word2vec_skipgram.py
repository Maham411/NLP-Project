
import os
import pandas as pd
from gensim.models import Word2Vec

# Load the cleaned Assignment 1 dataset
DATA_PATH = "data/processed/cleaned_dataset.csv"

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["clean_text"])

# Convert each cleaned tweet into a list of words
sentences = [
    text.split()
    for text in df["clean_text"].astype(str)
    if text.strip()
]

print("Total tweets:", len(sentences))

# Skip-gram hyperparameters
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 20

# Train Word2Vec using Skip-gram
model = Word2Vec(
    sentences=sentences,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    workers=4,
    sg=1,
    epochs=EPOCHS,
    seed=42
)

# Save the trained model
os.makedirs("models", exist_ok=True)
model.save("models/skipgram.model")

# Display training results
print("\nSkip-gram training completed!")
print("Model type: Word2Vec Skip-gram")
print("sg:", model.sg)
print("Vector size:", VECTOR_SIZE)
print("Window:", WINDOW)
print("Minimum word count:", MIN_COUNT)
print("Epochs:", EPOCHS)
print("Vocabulary size:", len(model.wv))
print("Model saved: models/skipgram.model")
