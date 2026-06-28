# Wikipedia Gender Gap NLP - Phase 2

## Überblick

Dieses Projekt untersucht mögliche sprachliche Unterschiede zwischen weiblichen und männlichen Wikipedia-Biografien aus dem STEM-/Mathematikbereich.

Phase 1 enthielt bereits eine Baseline mit bereinigten Biografietexten, TF-IDF und Logistic Regression. Phase 2 erweitert diese Basis durch zusätzliche Modellvergleiche und kritischere Analysen.

## Projektziel

Das Ziel ist nicht nur eine möglichst hohe Klassifikationsleistung, sondern auch eine vorsichtige Interpretation der Ergebnisse:

- Welche Modelle klassifizieren weibliche vs. männliche Biografien am besten?
- Wie stabil sind die Ergebnisse über mehrere Splits?
- Wie stark hängen die Ergebnisse von Textlänge, Leakage-Wörtern oder einzelnen Features ab?
- Liefert Sentence-BERT eine sinnvolle semantische Vergleichsbasis?

## Wichtige Dateien

- `src/features.py`: Preprocessing, Basic Cleaning, Strict Cleaning und Hilfsfunktionen
- `src/experiments.py`: Phase-2-Modellexperimente und Evaluation
- `src/embeddings.py`: Sentence-BERT-Embeddings
- `src/analysis.py`: Datenqualität, Textlängenanalyse und Leakage-Analyse

## Methoden

- TF-IDF Unigramme
- TF-IDF Unigramme und Bigramme
- Logistic Regression
- Linear SVM
- Sentence-BERT Embeddings mit Logistic Regression
- 5-fold Cross-Validation
- Feature Ablation
- Textlängen-Kontrolle mit den ersten 300 Wörtern

## Outputs

Die wichtigsten Ergebnisse werden in `outputs/` gespeichert:

- `outputs/metrics/model_results.csv`
- `outputs/metrics/cross_validation_results.csv`
- `outputs/metrics/ablation_results.csv`
- `outputs/metrics/phase2_results_presentation_view.xlsx`
- `outputs/analysis/dataset_summary.csv`
- `outputs/analysis/text_length_by_gender.csv`
- `outputs/analysis/leakage_word_counts.csv`
- `outputs/figures/`

Die Excel-Datei `phase2_results_presentation_view.xlsx` fasst die wichtigsten Ergebnisse zusätzlich in einer präsentationsnahen Struktur zusammen:

- Experiment 1: lexikalische TF-IDF-Modelle
- Experiment 2: Sentence-BERT
- Experiment 3: Textlängen-Kontrolle
- Experiment 4: Feature Ablation

## Ausführung

Phase-1-Baseline:

```bash
python src/main.py --mode adj
python src/main.py --mode full
python src/main.py --mode full_nosample
```

Phase-2-Experimente:

```bash
python src/experiments.py
```

Hinweis: Dieser Schritt kann beim ersten Lauf länger dauern, weil Sentence-BERT geladen und Embeddings berechnet werden.

Datenanalyse:

```bash
python src/analysis.py
```

## Interpretation

Die Ergebnisse werden nicht als direkter Beweis für Gender Bias interpretiert. Eine hohe Modellleistung kann auch durch Datenartefakte wie Textlänge, Wikipedia-Coverage, thematische Unterschiede oder verbleibende Leakage-Wörter beeinflusst werden.
