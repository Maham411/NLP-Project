
import unicodedata
import spacy
from normalization import normalize_unicode, normalize_text, nlp

# NFC normalization
print("1. NFC NORMALIZATION")
sample = "cafe\u0301"
normalized = normalize_unicode(sample)
print("Before:", repr(sample))
print("After:", repr(normalized))
print("NFC confirmed:", unicodedata.is_normalized("NFC", normalized))

# Case folding
print("\n2. CASE FOLDING")
text = "THE FLIGHTS ARE GOOD"
print("Before:", text)
print("After:", text.casefold())

# Stopword removal and negation preservation
print("\n3. STOPWORD REMOVAL")
text = "I am not happy with this flight"
doc = nlp(text)
kept = [
    token.text for token in doc
    if not token.is_stop or token.text.lower() in {"not", "no", "never"}
]
print("Before:", text)
print("After:", " ".join(kept))

# Lemmatization
print("\n4. LEMMATIZATION")
for word in ["flights", "passengers", "waiting", "delayed"]:
    print(word, "->", nlp(word)[0].lemma_)

# Complete normalization
print("\n5. COMPLETE NORMALIZATION")
text = "The flights are not very good and passengers were waiting"
print("Before:", text)
print("After:", normalize_text(text))
