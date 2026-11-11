
import pandas as pd
from noise_removal import remove_noise

# Load original dataset
df = pd.read_csv("data/raw/Tweets.csv")

# Select required columns
df = df[["text", "airline_sentiment"]].dropna(subset=["text"])

# Apply noise removal
df["noise_removed_text"] = df["text"].apply(remove_noise)

# Save cleaned dataset
df[["noise_removed_text", "airline_sentiment"]].to_csv(
    "data/processed/noise_removed_tweets.csv",
    index=False
)

# Display results
print("Total tweets processed:", len(df))

print("\nBefore and After Examples:")

for i in range(5):
    print("\nOriginal:", df.iloc[i]["text"])
    print("Cleaned:", df.iloc[i]["noise_removed_text"])

print("\nSentiment Distribution:")
print(df["airline_sentiment"].value_counts())

print("\nNoise removal completed successfully!")
print("File saved: data/processed/noise_removed_tweets.csv")
