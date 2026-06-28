import re
import random
from nltk import word_tokenize, pos_tag


# ------------------------
# Konstanten
# ------------------------

# Basic Gender-Wörter aus der Phase-1-Reinigung.
GENDER_WORDS = {"female", "male", "woman", "man", "she", "he", "her", "his"}

# Strengere Liste für Phase 2.
# Sie entfernt zusätzliche Gender- und Familienwörter, um Leakage zu reduzieren.
STRICT_GENDER_WORDS = GENDER_WORDS | {
    "women",
    "men",
    "hers",
    "him",
    "wife",
    "husband",
    "mother",
    "father",
    "daughter",
    "son",
    "married",
}

# Wörter, die bei der Adjektiv-Analyse eher Rauschen erzeugen können.
# Diese Liste gehört vor allem zur Phase-1-Adjektiv-Baseline.
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
    "soviet", "lycee"

    # months
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"
}

REFERENCE_KEYWORDS = {
    "academic": ["advisor", "supervisor", "professor", "student"],
    "family": ["father", "mother", "wife", "husband", "children"],
    "general": ["worked with", "collaborated with"]
}


# ------------------------
# Text Sampling
# ------------------------

def sample_text(text, max_words=300):
    # Wählt zufällig einen Ausschnitt mit maximal 300 Wörtern.
    # Diese Funktion stammt aus Phase 1.
    words = text.split()
    if len(words) <= max_words:
        return text
    start = random.randint(0, len(words) - max_words)
    return " ".join(words[start:start + max_words])


# ------------------------
# Cleaning
# ------------------------

def remove_names(text, names):
    # Senkt den Text auf Kleinbuchstaben und entfernt Namensbestandteile.
    # Dadurch soll das Modell nicht direkt über Namen lernen.
    text = text.lower()
    for name in names:
        for part in name.lower().split("_"):
            text = re.sub(rf"\b{re.escape(part)}\b", "", text)
    return text


def remove_gender_words(text):
    # Basic Cleaning: entfernt einfache Gender-Wörter aus Phase 1.
    return " ".join(w for w in text.split() if w not in GENDER_WORDS)


def remove_gender_words_strict(text):
    # Strict Cleaning: entfernt zusützlich Familien- und weitere Gender-Begriffe.
    return " ".join(w for w in text.split() if w not in STRICT_GENDER_WORDS)


def clean_basic(text, names):
    # Komplette Basic-Cleaning-Funktion für die Experimente.
    text = remove_names(text, names)
    return remove_gender_words(text)


def clean_strict(text, names):
    # Komplette Strict-Cleaning-Funktion für die Phase-2-Vergleiche.
    text = remove_names(text, names)
    return remove_gender_words_strict(text)


# ------------------------
# Feature Extraction
# ------------------------

def extract_adjectives(text):
    # Extrahiert Adjektive mit NLTK.
    # Diese Funktion wird vor allem für die alte Adjektiv-Baseline verwendet.
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    adjectives = [
        word.lower()
        for word, tag in tagged
        if tag.startswith("JJ")
    ]

    return " ".join(adjectives)


def filter_adjectives(text):
    # Entfernt sehr kurze, numerische oder wenig informative Adjektive.
    words = text.split()

    filtered = [
        w for w in words
        if w not in BAD_ADJECTIVES
        and len(w) > 2                 # remove very short noise
        and not w.isdigit()           # remove numbers
    ]

    return " ".join(filtered)


# ------------------------
# Reference Features
# ------------------------

def count_references(text):
    # Zühlt Wörter, die auf akademische, familiäre oder allgemeine Bezüge hinweisen.
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
    # Fügt einfache Referenz-Features zum DataFrame hinzu.
    # Diese Funktion gehört zur bisherigen Analyse und bleibt für Kompatibilität erhalten.
    refs = df["clean"].apply(count_references)

    df["total_refs"] = refs.apply(lambda x: x["total"])
    df["words"] = df["clean"].apply(lambda x: len(x.split()))

    # avoid division by zero
    df["ref_ratio"] = df["total_refs"] / df["words"].replace(0, 1)

    return df
