"""
preprocessing.py - Turns messy student text into clean text.

Steps:
  1. lowercase everything
  2. remove punctuation (keep only letters and digits)
  3. split into words
  4. remove very common "filler" words (stopwords)
  5. make plural words simpler (fees -> fee)

IMPORTANT: the exact same function is used when training AND when chatting.
"""

import re

# Filler words that carry almost no meaning for finding the intent.
# NOTE: we deliberately KEEP words like "how", "what", "when", "where",
# "which", "who", "much" because they help tell questions apart.
STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be",
    "to", "of", "for", "in", "on", "at", "and", "or", "with",
    "do", "does", "did", "have", "has", "had",
    "i", "me", "my", "we", "our", "you", "your", "it", "its",
    "can", "could", "would", "will", "shall", "please",
    "tell", "about", "there", "any", "some", "this", "that",
    "these", "those", "get",
}


def simple_singular(word):
    """
    A very small, easy-to-explain rule set to turn plurals into singular.
    Examples: universities -> university, classes -> class, fees -> fee.
    It is not perfect English, but it is CONSISTENT, and that is what matters.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"          # universities -> university
    if word.endswith("sses"):
        return word[:-2]                # classes -> class
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]                # fees -> fee
    return word


def clean_text(text):
    """Convert one sentence into cleaned text (a string of useful words)."""
    # 1. lowercase
    text = text.lower()

    # 2. replace anything that is not a letter, digit or space with a space
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 3. split into words
    words = text.split()

    # 4 + 5. drop stopwords and 1-letter words, simplify plurals
    cleaned_words = []
    for word in words:
        if word in STOPWORDS or len(word) < 2:
            continue
        cleaned_words.append(simple_singular(word))

    return " ".join(cleaned_words)


if __name__ == "__main__":
    print(clean_text("What are the Admission Requirements for Universities??"))
    # Expected: what admission requirement university
