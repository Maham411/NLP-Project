
import unicodedata
import spacy

nlp = spacy.load("en_core_web_sm")

NEGATIONS = {
    "not", "no", "never", "nor", "without",
    "n't", "cannot"
}

IMPORTANT_WORDS = {
    "very", "really", "too", "so", "extremely",
    "quite", "but", "however", "although", "yet"
}

def normalize_unicode(text):
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)

def normalize_text(text):
    text = normalize_unicode(text)
    text = text.casefold()

    doc = nlp(text)
    cleaned_tokens = []

    for token in doc:
        word = token.text.casefold()

        if token.is_space or token.is_punct:
            continue

        if word == "n't":
            cleaned_tokens.append("not")
            continue

        if not (token.is_alpha or token.like_num):
            continue

        if token.is_stop and word not in NEGATIONS and word not in IMPORTANT_WORDS:
            continue

        lemma = token.lemma_.casefold()

        if lemma:
            cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)
