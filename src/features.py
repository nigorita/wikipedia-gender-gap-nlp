import re
import random
from nltk import word_tokenize, pos_tag


# ------------------------
# Constants
# ------------------------

GENDER_WORDS = {"female", "male", "woman", "man", "she", "he", "her", "his"}

# remove bias-heavy adjectives (nationality, location, weak words)
BAD_ADJECTIVES = {
    # nationality
    "american", "french", "german", "italian", "spanish", "polish",
    "british", "russian", "hungarian", "african", "asian", "european", "swedish", "dutch", "japanese", "chinese", 

    # locations / generic noise
    "paris", "london", "united", "north", "south", "european", "asian", "american", "sweden", 
    # weak / meaningless
    "large", "high", "important", "general", "original", "particular",

    # dataset noise
    "born", "international", "national",

    # other common but uninformative words
    "second", "different", "standard", "foreign", "memorial",

      # weak / generic
    "old", "older", "early", "higher", "earned",

    # geo / historical
    "soviet", "lycée"

    # months
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"
}

REFERENCE_KEYWORDS = {
    "academic": ["advisor", "supervisor", "professor", "student"],
    "family": ["father", "mother", "wife", "husband", "children"],
    "general": ["worked with", "collaborated with"]
}


# ------------------------
# Text sampling
# ------------------------

def sample_text(text, max_words=300):
    words = text.split()
    if len(words) <= max_words:
        return text
    start = random.randint(0, len(words) - max_words)
    return " ".join(words[start:start + max_words])


# ------------------------
# Cleaning
# ------------------------

def remove_names(text, names):
    text = text.lower()
    for name in names:
        for part in name.lower().split("_"):
            text = re.sub(rf"\b{re.escape(part)}\b", "", text)
    return text


def remove_gender_words(text):
    return " ".join(w for w in text.split() if w not in GENDER_WORDS)


# ------------------------
# Feature extraction
# ------------------------

def extract_adjectives(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    adjectives = [
        word.lower()
        for word, tag in tagged
        if tag.startswith("JJ")
    ]

    return " ".join(adjectives)


def filter_adjectives(text):
    words = text.split()

    filtered = [
        w for w in words
        if w not in BAD_ADJECTIVES
        and len(w) > 2                 # remove very short noise
        and not w.isdigit()           # remove numbers
    ]

    return " ".join(filtered)


# ------------------------
# Reference features
# ------------------------

def count_references(text):
    text = text.lower()

    academic = sum(text.count(w) for w in REFERENCE_KEYWORDS["academic"])
    family = sum(text.count(w) for w in REFERENCE_KEYWORDS["family"])
    general = sum(text.count(w) for w in REFERENCE_KEYWORDS["general"])

    return {
        "academic": academic,
        "family": family,
        "general": general,
        "total": academic + family + general
    }


def add_reference_features(df):
    refs = df["clean"].apply(count_references)

    df["total_refs"] = refs.apply(lambda x: x["total"])
    df["words"] = df["clean"].apply(lambda x: len(x.split()))

    # avoid division by zero
    df["ref_ratio"] = df["total_refs"] / df["words"].replace(0, 1)

    return df