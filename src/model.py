from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def clean_top_words(word_weights):
    cleaned = []

    for word, weight in word_weights:
        word = word.strip()

        # remove duplicates like "math math"
        if len(set(word.split())) == 1 and len(word.split()) > 1:
            continue

        # remove garbage tokens
        if any(bad in word for bad in ["isbn", "doi", "retrieved"]):
            continue

        cleaned.append((word, float(weight)))

    return cleaned


def print_top_words(word_weights, top_n=10):
    sorted_words = sorted(word_weights, key=lambda x: x[1])
    cleaned = clean_top_words(sorted_words)

    female_words = cleaned[:top_n]
    male_words = cleaned[-top_n:]

    print("\nTop words for FEMALE:")
    for w, v in female_words:
        print(f"{w:25s} {v:.3f}")

    print("\nTop words for MALE:")
    for w, v in reversed(male_words):
        print(f"{w:25s} {v:.3f}")


def train_model(df, text_column, label):
    X = df[text_column]
    y = df["gender"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 1),   # cleaner output
        min_df=3,
        max_df=0.8
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    print(f"\n=== {label} ===")
    print(classification_report(y_test, preds))

    # feature importance
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    word_weights = list(zip(feature_names, coefs))

    print_top_words(word_weights)