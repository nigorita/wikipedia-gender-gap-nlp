import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


INPUT_FILE = "features.csv"
OUTPUT_FILE = "baseline_results.csv"


def load_data():
    # Load the feature file.
    return pd.read_csv(INPUT_FILE)


def prepare_data(df):
    # Prepare text and labels.
    if "selected_text" in df.columns:
        texts = df["selected_text"].fillna("")
        if (texts.str.strip() == "").all():
            texts = df["text"].fillna("")
    else:
        texts = df["text"].fillna("")

    labels = df["gender"]
    return texts, labels


def evaluate_model(model_name, vectorizer_name, model, vectorizer, x_train, x_test, y_train, y_test):
    # Train the model and calculate the scores.
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    model.fit(x_train_vectorized, y_train)
    predictions = model.predict(x_test_vectorized)

    return {
        "model": model_name,
        "vectorizer": vectorizer_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1": f1_score(y_test, predictions, average="macro", zero_division=0),
    }


def run_experiments(texts, labels):
    # Run all baseline combinations.
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.33,
        random_state=42,
        stratify=labels,
    )

    print("Training rows:", len(x_train))
    print("Test rows:", len(x_test))
    print()

    experiments = [
        ("LogisticRegression", "BagOfWords", LogisticRegression(max_iter=1000), CountVectorizer()),
        ("LogisticRegression", "TFIDF", LogisticRegression(max_iter=1000), TfidfVectorizer()),
        ("MultinomialNB", "BagOfWords", MultinomialNB(), CountVectorizer()),
        ("MultinomialNB", "TFIDF", MultinomialNB(), TfidfVectorizer()),
    ]

    results = []

    for model_name, vectorizer_name, model, vectorizer in experiments:
        result = evaluate_model(
            model_name,
            vectorizer_name,
            model,
            vectorizer,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        results.append(result)

    return pd.DataFrame(results)


def main():
    df = load_data()
    texts, labels = prepare_data(df)

    results_df = run_experiments(texts, labels)
    print(results_df)

    results_df.to_csv(OUTPUT_FILE, index=False)
    print()
    print("Saved baseline results to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
