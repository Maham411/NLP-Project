
from gensim.models import Word2Vec

# Load the trained Skip-gram model
model = Word2Vec.load("models/skipgram.model")

# Domain-related query words
query_words = [
    "flight",
    "airline",
    "delay",
    "service",
    "customer"
]

print("SKIP-GRAM SIMILARITY ANALYSIS")

for word in query_words:
    print("\nQuery word:", word)

    if word in model.wv:
        similar_words = model.wv.most_similar(word, topn=5)

        for similar_word, score in similar_words:
            print(f"{similar_word}: {score:.4f}")
    else:
        print("Word not found in vocabulary")
