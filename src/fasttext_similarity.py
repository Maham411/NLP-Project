
from gensim.models import FastText

# Load the trained FastText model
model = FastText.load("models/fasttext.model")

# Airline-related query words
query_words = [
    "flight",
    "airline",
    "airport",
    "ticket",
    "customer"
]

print("FASTTEXT WORD SIMILARITY ANALYSIS")
print("=" * 50)

for word in query_words:
    print("\nQuery word:", word)

    # Find the five most similar words
    neighbours = model.wv.most_similar(word, topn=5)

    for neighbour, score in neighbours:
        print(f"{neighbour}: {score:.4f}")

