
import pandas as pd
import re
from normalization import normalize_text

# Load the original dataset
df = pd.read_csv("data/raw/Tweets.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Select required columns
df = df[["text", "airline_sentiment"]].dropna(subset=["text"])

# Function to extract words
def get_words(text):
    return re.findall(r"\b[\w']+\b", str(text).casefold())

# Calculate vocabulary before normalization
before_words = []

for tweet in df["text"]:
    before_words.extend(get_words(tweet))

before_vocab = set(before_words)

# Apply text normalization
df["cleaned_text"] = df["text"].apply(normalize_text)

# Calculate vocabulary after normalization
after_words = []

for tweet in df["cleaned_text"]:
    after_words.extend(get_words(tweet))

after_vocab = set(after_words)

# Save normalized dataset
df[["cleaned_text", "airline_sentiment"]].to_csv(
    "data/processed/normalized_tweets.csv",
    index=False
)

# Display results
print("\nTotal documents:", len(df))
print("Vocabulary before cleaning:", len(before_vocab))
print("Vocabulary after cleaning:", len(after_vocab))
print("Words removed:", len(before_vocab) - len(after_vocab))

# Display before and after examples
print("\nBEFORE AND AFTER EXAMPLES")

for i in range(min(5, len(df))):
    print("\nOriginal:", df.iloc[i]["text"])
    print("Normalized:", df.iloc[i]["cleaned_text"])

print("\nCleaned dataset saved to data/processed/normalized_tweets.csv")
