
from gensim.models import FastText

# Load the trained FastText model
model = FastText.load("models/fasttext.model")

# Words used in the analogy
positive_words = ["americanairline", "southwest"]
negative_words = ["american"]

print("FASTTEXT WORD ANALOGY ANALYSIS")
print("=" * 45)

print("Analogy: americanairline - american + southwest")
print("\nVocabulary availability:")

for word in positive_words + negative_words:
    status = "In vocabulary" if word in model.wv else "OOV (subword vector)"
    print(word, ":", status)

try:
    results = model.wv.most_similar(
        positive=positive_words,
        negative=negative_words,
        topn=5
    )

    print("\nTop 5 analogy results:")

    for word, score in results:
        print(f"{word}: {score:.4f}")

    if results:
        print("\nTop predicted word:", results[0][0])

except Exception as error:
    print("\nAnalogy could not be calculated:", error)

print("\nAnalogy testing completed.")

