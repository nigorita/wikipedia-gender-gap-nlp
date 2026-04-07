import re
import string

import pandas as pd
import nltk

from nltk import pos_tag
from nltk.corpus import stopwords


INPUT_FILE = "cleaned_data.csv"
FALLBACK_INPUT_FILE = "data.csv"
OUTPUT_FILE = "features.csv"
PUNKT_AVAILABLE = False
STOPWORDS_AVAILABLE = False
TAGGER_AVAILABLE = False


FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "of",
    "on",
    "she",
    "that",
    "the",
    "their",
    "to",
    "was",
    "were",
    "with",
}


COMMON_ADJECTIVES = {
    "english",
    "german",
    "polish",
    "french",
    "serbian",
    "american",
    "theoretical",
    "scientific",
    "mechanical",
    "modern",
    "molecular",
    "first",
    "famous",
    "influential",
    "central",
    "analytical",
    "general",
    "pure",
}


def check_nltk_data():
    # Check which nltk resources are available.
    global PUNKT_AVAILABLE
    global STOPWORDS_AVAILABLE
    global TAGGER_AVAILABLE

    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab")
        PUNKT_AVAILABLE = True
    except LookupError:
        PUNKT_AVAILABLE = False

    try:
        nltk.data.find("corpora/stopwords")
        STOPWORDS_AVAILABLE = True
    except LookupError:
        STOPWORDS_AVAILABLE = False

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        TAGGER_AVAILABLE = True
    except LookupError:
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
            TAGGER_AVAILABLE = True
        except LookupError:
            TAGGER_AVAILABLE = False


def load_data():
    # Load cleaned data if it exists.
    try:
        return pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        return pd.read_csv(FALLBACK_INPUT_FILE)


def lowercase_text(text):
    # Convert text to lowercase.
    return str(text).lower()


def tokenize_text(text):
    # Split text into tokens.
    if PUNKT_AVAILABLE:
        return nltk.word_tokenize(text)
    return re.findall(r"[a-zA-Z]+", text)


def remove_stopwords(tokens):
    # Remove English stopwords.
    if STOPWORDS_AVAILABLE:
        stop_words = set(stopwords.words("english"))
    else:
        stop_words = FALLBACK_STOPWORDS
    return [word for word in tokens if word not in stop_words]


def remove_punctuation(tokens):
    # Remove punctuation tokens.
    return [word for word in tokens if word not in string.punctuation]


def tag_tokens(tokens):
    # Add POS tags to tokens.
    if TAGGER_AVAILABLE:
        return pos_tag(tokens)
    return simple_pos_tag(tokens)


def simple_pos_tag(tokens):
    # Use a basic fallback when the nltk tagger is not available.
    tagged_tokens = []

    adjective_suffixes = ("al", "ary", "ful", "ic", "ical", "ive", "less", "ous")

    for word in tokens:
        if word in COMMON_ADJECTIVES or word.endswith(adjective_suffixes):
            tagged_tokens.append((word, "JJ"))
        else:
            tagged_tokens.append((word, "NN"))

    return tagged_tokens


def extract_adjectives(tagged_tokens):
    # Keep adjective tokens.
    return [word for word, tag in tagged_tokens if tag.startswith("JJ")]


def extract_nouns(tagged_tokens):
    # Keep noun tokens.
    return [word for word, tag in tagged_tokens if tag.startswith("NN")]


def process_row(text):
    # Run the preprocessing and feature extraction steps.
    text = lowercase_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = remove_punctuation(tokens)
    tagged_tokens = tag_tokens(tokens)

    adjectives = extract_adjectives(tagged_tokens)
    nouns = extract_nouns(tagged_tokens)
    selected_words = adjectives + nouns

    return pd.Series(
        {
            "processed_text": " ".join(tokens),
            "adjective_text": " ".join(adjectives),
            "noun_text": " ".join(nouns),
            "selected_text": " ".join(selected_words),
        }
    )


def main():
    check_nltk_data()
    df = load_data()
    df = df.dropna(subset=["text"]).copy()

    feature_df = df["text"].apply(process_row)
    df = pd.concat([df, feature_df], axis=1)
    df.to_csv(OUTPUT_FILE, index=False)

    if TAGGER_AVAILABLE:
        print("Used nltk POS tagging")
    else:
        print("Used simple fallback POS tagging")

    print("Rows processed:", len(df))
    print("Saved features to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
