
from noise_removal import remove_noise

from regex_patterns import (
    URL_PATTERN,
    MENTION_PATTERN,
    HASHTAG_PATTERN,
    NUMBER_PATTERN,
    HTML_ENTITY_PATTERN,
    REPEATED_PUNCTUATION_PATTERN
)

examples = [
    "@VirginAmerica What @dhepburn said.",
    "Visit https://example.com for updates!!!",
    "My flight is delayed &amp; I am unhappy!",
    "#WorstAirline has delayed my flight!!!",
    "Flight 123 is delayed by 2 hours.",
    "This is sooo bad!!!"
]

print("REGULAR EXPRESSION TESTING")

for text in examples:
    print("\nOriginal Text:", text)

    print("URLs:", URL_PATTERN.findall(text))
    print("Mentions:", MENTION_PATTERN.findall(text))
    print("Hashtags:", HASHTAG_PATTERN.findall(text))
    print("Numbers:", NUMBER_PATTERN.findall(text))
    print("HTML Entities:", HTML_ENTITY_PATTERN.findall(text))
    print("Repeated Punctuation:", REPEATED_PUNCTUATION_PATTERN.findall(text))

    print("Cleaned Text:", remove_noise(text))

print("\nAll regex tests completed!")
