import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

MENTION_PATTERN = re.compile(r"@\w+")

HASHTAG_PATTERN = re.compile(r"#(\w+)")

NUMBER_PATTERN = re.compile(r"\b\d+(?:[.:/-]\d+)*\b")

HTML_ENTITY_PATTERN = re.compile(r"&(?:amp|lt|gt|quot|#39);", re.IGNORECASE)

WHITESPACE_PATTERN = re.compile(r"\s+")

REPEATED_PUNCTUATION_PATTERN = re.compile(r"([!?.,])\1{1,}")
