"""Modellexperimente für Phase 2.

Zuständig: Tugba

Diese Datei vergleicht lexikalische und semantische Repräsentationen:
- TF-IDF Unigramme / Bigramme
- Logistic Regression / SVM
- Sentence-BERT Embeddings aus embeddings.py als semantische Erweiterung
"""

from pathlib import Path
import os
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from features import clean_basic, clean_strict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "data_math.csv"
OUTPUT_METRICS = PROJECT_ROOT / "outputs" / "metrics"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_ANALYSIS = PROJECT_ROOT / "outputs" / "analysis"
MPL_CONFIG = PROJECT_ROOT / "outputs" / ".matplotlib"

LABELS = ["female", "male"]
RANDOM_STATE = 42
N_SPLITS = 5
N_ABLATION_FEATURES_PER_CLASS = 20


def ensure_output_dirs() -> None:
    # Erstellt alle Ausgabeordner, damit Tabellen und Grafiken gespeichert werden können.
    OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
    MPL_CONFIG.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))


def load_clean_data() -> pd.DataFrame:
    # Lädt den Datensatz und erzeugt verschiedene bereinigte Textversionen.
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].fillna("")

    # Zusätzliche Version für die Textlängenkontrolle:
    # Jede Biografie wird auf die ersten 300 Wörter begrenzt.
    df["text_first_300"] = df["text"].apply(first_n_words)

    names = list(df["name"])

    # Basic Cleaning entspricht ungeführ der Phase-1-Reinigung.
    df["clean_basic"] = df["text"].apply(lambda text: clean_basic(text, names))

    # Strict Cleaning entfernt zusätzliche Gender- und Familienwörter.
    df["clean_strict"] = df["text"].apply(lambda text: clean_strict(text, names))

    # Dieselben Cleaning-Versionen werden auch für die 300-Wörter-Texte erzeugt.
    df["clean_basic_first_300"] = df["text_first_300"].apply(
        lambda text: clean_basic(text, names)
    )
    df["clean_strict_first_300"] = df["text_first_300"].apply(
        lambda text: clean_strict(text, names)
    )

    return df


def first_n_words(text: str, n: int = 300) -> str:
    # Schneidet einen Text nach den ersten n Wörtern ab.
    return " ".join(text.split()[:n])


def split_data(df: pd.DataFrame):
    # Teilt den Datensatz in Training und Test auf.
    # stratify sorgt dafür, dass female/male im Split ähnlich verteilt bleiben.
    return train_test_split(
        df,
        test_size=0.2,
        stratify=df["gender"],
        random_state=RANDOM_STATE,
    )


def save_confusion_matrix(y_test, preds, output_name: str) -> None:
    # Speichert eine Confusion Matrix als Bild.
    # Damit sieht man, welche Klassen richtig oder falsch vorhergesagt wurden.
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, preds, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(values_format="d", cmap="Blues")
    plt.title(output_name.replace("_", " "))
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / f"confusion_matrix_{output_name}.png", dpi=200)
    plt.close()


def metric_row(
    model_name: str,
    representation: str,
    cleaning: str,
    text_scope: str,
    classifier: str,
    y_test,
    preds,
    status: str = "ok",
    note: str = "",
) -> dict:
    # Berechnet die wichtigsten Evaluationsmetriken für ein Modell.
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_test,
        preds,
        average="macro",
        zero_division=0,
    )

    return {
        "model": model_name,
        "representation": representation,
        "cleaning": cleaning,
        "text_scope": text_scope,
        "classifier": classifier,
        "accuracy": accuracy_score(y_test, preds),
        "precision_macro": precision,
        "recall_macro": recall,
        "macro_f1": macro_f1,
        "status": status,
        "note": note,
    }


def skipped_row(
    model_name: str,
    representation: str,
    cleaning: str,
    text_scope: str,
    classifier: str,
    note: str,
) -> dict:
    # Wird verwendet, falls ein Experiment nicht ausgeführt werden konnte.
    # So bleibt die Ergebnistabelle trotzdem vollständig.
    return {
        "model": model_name,
        "representation": representation,
        "cleaning": cleaning,
        "text_scope": text_scope,
        "classifier": classifier,
        "accuracy": None,
        "precision_macro": None,
        "recall_macro": None,
        "macro_f1": None,
        "status": "skipped",
        "note": note,
    }


def tfidf_pipeline(ngram_range, classifier):
    # Baut eine klassische Textklassifikations-Pipeline:
    # 1. TF-IDF wandelt Text in Zahlen um.
    # 2. Der Klassifikator lernt female/male vorherzusagen.
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=1500,
                    stop_words="english",
                    ngram_range=ngram_range,
                    min_df=3,
                    max_df=0.8,
                ),
            ),
            ("clf", classifier),
        ]
    )


def tfidf_experiment_definitions() -> list[tuple]:
    # Definiert die drei lexikalischen Modellvarianten.
    return [
        (
            "tfidf_unigram_logreg",
            "tfidf_unigram",
            "logistic_regression",
            tfidf_pipeline((1, 1), LogisticRegression(max_iter=1000)),
        ),
        (
            "tfidf_unigram_bigram_logreg",
            "tfidf_unigram_bigram",
            "logistic_regression",
            tfidf_pipeline((1, 2), LogisticRegression(max_iter=1000)),
        ),
        (
            "tfidf_unigram_bigram_svm",
            "tfidf_unigram_bigram",
            "svm",
            tfidf_pipeline((1, 2), LinearSVC()),
        ),
    ]


def run_tfidf_experiments(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[dict]:
    # Trainiert und testet alle TF-IDF-Modelle auf einem festen Train/Test-Split.
    experiments = tfidf_experiment_definitions()

    rows = []

    # Vier Input-Versionen werden verglichen:
    # full text/basic, full text/strict, first 300/basic, first 300/strict.
    for cleaning, text_scope, text_column in [
        ("basic", "full_text", "clean_basic"),
        ("strict", "full_text", "clean_strict"),
        ("basic", "first_300_words", "clean_basic_first_300"),
        ("strict", "first_300_words", "clean_strict_first_300"),
    ]:
        x_train = train_df[text_column]
        x_test = test_df[text_column]
        y_train = train_df["gender"]
        y_test = test_df["gender"]

        for model_name, representation, classifier_name, model in experiments:
            full_name = f"{model_name}_{cleaning}_{text_scope}"

            # Modell lernt auf dem Trainingsset und wird auf dem Testset bewertet.
            model.fit(x_train, y_train)
            preds = model.predict(x_test)

            rows.append(
                metric_row(
                    full_name,
                    representation,
                    cleaning,
                    text_scope,
                    classifier_name,
                    y_test,
                    preds,
                )
            )
            save_confusion_matrix(y_test, preds, full_name)

    return rows


def run_tfidf_cross_validation(df: pd.DataFrame) -> pd.DataFrame:
    # 5-fold Cross-Validation prüft, ob die Ergebnisse stabil sind.
    # Das ist wichtig, weil der Datensatz relativ klein ist.
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "macro_f1": "f1_macro",
    }

    rows = []

    for cleaning, text_scope, text_column in [
        ("basic", "full_text", "clean_basic"),
        ("strict", "full_text", "clean_strict"),
        ("basic", "first_300_words", "clean_basic_first_300"),
        ("strict", "first_300_words", "clean_strict_first_300"),
    ]:
        for model_name, representation, classifier_name, model in tfidf_experiment_definitions():
            scores = cross_validate(
                model,
                df[text_column],
                df["gender"],
                cv=cv,
                scoring=scoring,
                n_jobs=None,
            )

            rows.append(
                {
                    "model": f"{model_name}_{cleaning}",
                    "representation": representation,
                    "cleaning": cleaning,
                    "text_scope": text_scope,
                    "classifier": classifier_name,
                    "folds": N_SPLITS,
                    "accuracy_mean": scores["test_accuracy"].mean(),
                    "accuracy_std": scores["test_accuracy"].std(),
                    "precision_macro_mean": scores["test_precision_macro"].mean(),
                    "precision_macro_std": scores["test_precision_macro"].std(),
                    "recall_macro_mean": scores["test_recall_macro"].mean(),
                    "recall_macro_std": scores["test_recall_macro"].std(),
                    "macro_f1_mean": scores["test_macro_f1"].mean(),
                    "macro_f1_std": scores["test_macro_f1"].std(),
                }
            )

    return pd.DataFrame(rows)


def get_top_tfidf_features(texts, labels, top_n: int = N_ABLATION_FEATURES_PER_CLASS) -> pd.DataFrame:
    # Trainiert das beste lexikalische Modell und sucht die stärksten Features.
    # Negative Gewichte zeigen eher in Richtung female, positive eher in Richtung male.
    model = tfidf_pipeline((1, 2), LinearSVC())
    model.fit(texts, labels)

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["clf"]

    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]

    female_idx = coefs.argsort()[:top_n]
    male_idx = coefs.argsort()[-top_n:][::-1]

    rows = []
    for idx in female_idx:
        rows.append(
            {
                "feature": feature_names[idx],
                "weight": coefs[idx],
                "direction": "female",
            }
        )

    for idx in male_idx:
        rows.append(
            {
                "feature": feature_names[idx],
                "weight": coefs[idx],
                "direction": "male",
            }
        )

    return pd.DataFrame(rows)


def remove_features_from_text(text: str, features: list[str]) -> str:
    # Entfernt die automatisch gefundenen Top-Features aus einem Text.
    cleaned = text
    for feature in features:
        pattern = r"\b" + r"\s+".join(re.escape(part) for part in feature.split()) + r"\b"
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def run_feature_ablation(df: pd.DataFrame) -> pd.DataFrame:
    # Feature Ablation testet, wie stark das Modell von wenigen Top-Wörtern abhängt.
    # Dafür werden die stärksten 20 female- und 20 male-Features entfernt.
    feature_df = get_top_tfidf_features(df["clean_basic"], df["gender"])
    feature_df.to_csv(OUTPUT_ANALYSIS / "ablated_features.csv", index=False)

    features_to_remove = feature_df["feature"].tolist()
    df = df.copy()
    df["clean_basic_ablated"] = df["clean_basic"].apply(
        lambda text: remove_features_from_text(text, features_to_remove)
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "macro_f1": "f1_macro",
    }

    model = tfidf_pipeline((1, 2), LinearSVC())
    scores = cross_validate(
        model,
        df["clean_basic_ablated"],
        df["gender"],
        cv=cv,
        scoring=scoring,
        n_jobs=None,
    )

    return pd.DataFrame(
        [
            {
                "model": "tfidf_unigram_bigram_svm_basic_full_text_ablated",
                "representation": "tfidf_unigram_bigram",
                "cleaning": "basic",
                "text_scope": "full_text",
                "classifier": "svm",
                "ablated_features": len(features_to_remove),
                "folds": N_SPLITS,
                "accuracy_mean": scores["test_accuracy"].mean(),
                "accuracy_std": scores["test_accuracy"].std(),
                "precision_macro_mean": scores["test_precision_macro"].mean(),
                "precision_macro_std": scores["test_precision_macro"].std(),
                "recall_macro_mean": scores["test_recall_macro"].mean(),
                "recall_macro_std": scores["test_recall_macro"].std(),
                "macro_f1_mean": scores["test_macro_f1"].mean(),
                "macro_f1_std": scores["test_macro_f1"].std(),
            }
        ]
    )


def run_sbert_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[dict]:
    # Sentence-BERT erzeugt semantische Embeddings.
    # Danach wird ein einfacher Klassifikator auf diesen Embeddings trainiert.
    try:
        from embeddings import encode_with_sentence_bert
    except ImportError as exc:
        return [
            skipped_row(
                "sbert_logreg_strict",
                "sentence_bert",
                "strict",
                "full_text",
                "logistic_regression",
                str(exc),
            )
        ]

    y_train = train_df["gender"]
    y_test = test_df["gender"]

    try:
        x_train = encode_with_sentence_bert(train_df["clean_strict"])
        x_test = encode_with_sentence_bert(test_df["clean_strict"])
    except Exception as exc:
        return [
            skipped_row(
                "sbert_logreg_strict",
                "sentence_bert",
                "strict",
                "full_text",
                "logistic_regression",
                f"Sentence-BERT konnte nicht ausgeführt werden: {exc}",
            )
        ]

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    save_confusion_matrix(y_test, preds, "sbert_logreg_strict")

    return [
        metric_row(
            "sbert_logreg_strict",
            "sentence_bert",
            "strict",
            "full_text",
            "logistic_regression",
            y_test,
            preds,
        )
    ]


def save_test_split_for_error_analysis(test_df: pd.DataFrame) -> None:
    # Speichert den Test-Split, damit später Fehlerbeispiele analysiert werden können.
    test_df[
        [
            "name",
            "gender",
            "field",
            "text",
            "text_first_300",
            "clean_basic",
            "clean_strict",
            "clean_basic_first_300",
            "clean_strict_first_300",
        ]
    ].to_csv(
        OUTPUT_ANALYSIS / "test_split_for_error_analysis.csv",
        index=False,
    )


def main() -> None:
    # Hauptablauf: Daten laden, Experimente ausführen, Ergebnisse speichern.
    ensure_output_dirs()

    df = load_clean_data()
    train_df, test_df = split_data(df)

    rows = []
    rows.extend(run_tfidf_experiments(train_df, test_df))
    rows.extend(run_sbert_experiment(train_df, test_df))

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_METRICS / "model_results.csv", index=False)

    cv_results = run_tfidf_cross_validation(df)
    cv_results.to_csv(OUTPUT_METRICS / "cross_validation_results.csv", index=False)

    ablation_results = run_feature_ablation(df)
    ablation_results.to_csv(OUTPUT_METRICS / "ablation_results.csv", index=False)

    save_test_split_for_error_analysis(test_df)

    print("\nHold-out results:")
    print(results.round(3))
    print("\n5-fold cross-validation results:")
    print(cv_results.round(3))
    print("\nFeature ablation results:")
    print(ablation_results.round(3))


if __name__ == "__main__":
    main()
