
from gensim.models import Word2Vec

# Load the trained Skip-gram model
model = Word2Vec.load("models/skipgram.model")

# Two related pairs and two unrelated pairs
similar_pairs = [
    ("customer", "service"),
    ("flight", "airline")
]

unrelated_pairs = [
    ("customer", "weather"),
    ("flight", "money")
]

print("SKIP-GRAM COSINE SIMILARITY ANALYSIS")

print("\n--- Similar Word Pairs ---")
for word1, word2 in similar_pairs:
    if word1 in model.wv and word2 in model.wv:
        score = model.wv.similarity(word1, word2)
        print(f"{word1} - {word2}: {score:.4f}")
    else:
        print(f"Word not found: {word1} or {word2}")

print("\n--- Unrelated Word Pairs ---")
for word1, word2 in unrelated_pairs:
    if word1 in model.wv and word2 in model.wv:
        score = model.wv.similarity(word1, word2)
        print(f"{word1} - {word2}: {score:.4f}")
    else:
        print(f"Word not found: {word1} or {word2}")
