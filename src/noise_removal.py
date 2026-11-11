
import re
import html

from regex_patterns import (
    URL_PATTERN,
    MENTION_PATTERN,
    HASHTAG_PATTERN,
    NUMBER_PATTERN,
    WHITESPACE_PATTERN
)


def remove_noise(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Remove URLs
    text = URL_PATTERN.sub("", text)

    # Remove @mentions
    text = MENTION_PATTERN.sub("", text)

    # Remove numbers
    text = NUMBER_PATTERN.sub("", text)

    # Keep hashtag words but remove # symbol
    text = HASHTAG_PATTERN.sub(r"\1", text)

    # Replace newlines and tabs with spaces
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Remove extra spaces
    text = WHITESPACE_PATTERN.sub(" ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([!?.,])", r"\1", text)

    return text.strip()
