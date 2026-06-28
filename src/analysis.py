"""Datenqualitäts- und Leakage-Analyse für Phase 2.

Zustündig: Negar

Diese Datei ist unabhängig von den Modellexperimenten. Sie erstellt Tabellen
und Grafiken, die helfen zu interpretieren, ob die Klassifikationsergebnisse
verlässlich sind.
"""

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "data_math.csv"
OUTPUT_ANALYSIS = PROJECT_ROOT / "outputs" / "analysis"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"

LEAKAGE_WORDS = [
    "she",
    "her",
    "hers",
    "he",
    "his",
    "him",
    "woman",
    "women",
    "female",
    "male",
    "wife",
    "husband",
    "mother",
    "father",
    "daughter",
    "son",
    "married",
]


def ensure_output_dirs() -> None:
    # Erstellt die Ausgabeordner für Tabellen und Grafiken.
    OUTPUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    # Lädt den Datensatz und berechnet die Wortanzahl jeder Biografie.
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].fillna("")
    df["word_count"] = df["text"].str.split().str.len()
    return df


def save_dataset_summary(df: pd.DataFrame) -> None:
    # Speichert eine Zusammenfassung pro Gender:
    # Anzahl der Biografien und Textlängenstatistiken.
    summary = df.groupby("gender").agg(
        biographies=("name", "count"),
        mean_words=("word_count", "mean"),
        median_words=("word_count", "median"),
        min_words=("word_count", "min"),
        max_words=("word_count", "max"),
    )
    summary.round(2).to_csv(OUTPUT_ANALYSIS / "dataset_summary.csv")
    summary.round(2).to_csv(OUTPUT_ANALYSIS / "text_length_by_gender.csv")


def save_leakage_counts(df: pd.DataFrame) -> None:
    # Zühlt, in wie vielen Artikeln bestimmte Gender- oder Familienwörter vorkommen.
    # Das hilft zu prüfen, ob mögliche Leakage-Signale im Datensatz vorhanden sind.
    rows = []
    text = df["text"].str.lower()

    for word in LEAKAGE_WORDS:
        pattern = rf"\b{re.escape(word)}\b"
        contains_word = text.str.contains(pattern, regex=True, na=False)
        counts = contains_word.groupby(df["gender"]).sum()

        rows.append(
            {
                "word": word,
                "female_articles": int(counts.get("female", 0)),
                "male_articles": int(counts.get("male", 0)),
            }
        )

    pd.DataFrame(rows).to_csv(
        OUTPUT_ANALYSIS / "leakage_word_counts.csv",
        index=False,
    )


def save_longest_biographies(df: pd.DataFrame) -> None:
    # Speichert die laengsten Biografien, weil Textlänge ein wichtiger Einflussfaktor ist.
    df.sort_values("word_count", ascending=False)[
        ["name", "gender", "field", "word_count"]
    ].head(20).to_csv(OUTPUT_ANALYSIS / "longest_biographies.csv", index=False)


def save_text_length_plot(df: pd.DataFrame) -> None:
    # Erstellt eine Grafik zur Textlängenverteilung nach Gender.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib ist nicht installiert; Textlängen-Grafik wird übersprungen.")
        return

    plt.figure(figsize=(8, 5))
    for gender, group in df.groupby("gender"):
        plt.hist(group["word_count"], bins=25, alpha=0.6, label=gender)

    plt.xlabel("Wortanzahl")
    plt.ylabel("Anzahl der Biografien")
    plt.title("Textlängenverteilung nach Gender")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "text_length_distribution.png", dpi=200)
    plt.close()


def main() -> None:
    # Hauptablauf der Analyse-Datei.
    ensure_output_dirs()
    df = load_data()

    save_dataset_summary(df)
    save_leakage_counts(df)
    save_longest_biographies(df)
    save_text_length_plot(df)

    print("Phase-2-Analyseoutputs wurden gespeichert.")


if __name__ == "__main__":
    main()
