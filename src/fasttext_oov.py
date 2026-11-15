
import os
import pandas as pd
from gensim.models import FastText, Word2Vec

# File paths
DATA_PATH = "data/processed/cleaned_dataset.csv"
FASTTEXT_PATH = "models/fasttext.model"
WORD2VEC_PATH = "models/word2vec_comparison.model"

# Load the cleaned corpus
df = pd.read_csv(DATA_PATH)
texts = df["clean_text"].fillna("").astype(str)

sentences = [
    text.split()
    for text in texts
    if text.strip()
]

# Load the already trained FastText model
fasttext_model = FastText.load(FASTTEXT_PATH)

# Train Word2Vec using the same corpus and settings
word2vec_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,
    epochs=20,
    seed=42
)

# Save Word2Vec model
word2vec_model.save(WORD2VEC_PATH)

# Select a common word and a rare word
common_word = "flight"

rare_words = [
    word for word, count in word2vec_model.wv.key_to_index.items()
    if word != common_word
    and word2vec_model.wv.get_vecattr(word, "count") <= 3
]

rare_word = rare_words[0] if rare_words else common_word

# Test a misspelled word and an unseen word
test_words = [
    ("Common word", common_word),
    ("Rare word", rare_word),
    ("Misspelled word", "fligth"),
    ("OOV word", "airlne")
]

print("FASTTEXT VS WORD2VEC COMPARISON")
print("=" * 50)

for category, word in test_words:
    print("\n", category, ":", word)

    # FastText can create vectors using character n-grams
    fasttext_vector = fasttext_model.wv[word]
    print("FastText vector available: Yes")
    print("FastText vector shape:", fasttext_vector.shape)

    # Word2Vec only has vectors for vocabulary words
    if word in word2vec_model.wv:
        print("Word2Vec vector available: Yes")
        print("Word2Vec vector shape:", word2vec_model.wv[word].shape)
    else:
        print("Word2Vec vector available: No (OOV)")

    # Compare misspelled and unseen words with their intended words
    if word in ["fligth", "airlne"]:
        target = "flight" if word == "fligth" else "airline"

        if target in fasttext_model.wv:
            similarity = fasttext_model.wv.similarity(word, target)
            print("FastText similarity with", target, ":", round(similarity, 4))

        if word in word2vec_model.wv and target in word2vec_model.wv:
            similarity = word2vec_model.wv.similarity(word, target)
            print("Word2Vec similarity with", target, ":", round(similarity, 4))
        else:
            print("Word2Vec similarity: Not available")

print("\nWord2Vec model saved at:", WORD2VEC_PATH)

