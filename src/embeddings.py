"""Sentence-BERT-Hilfsfunktionen für Phase 2.

Zuständig: Tugba

Diese Datei enthält nur den Sentence-BERT-Teil. experiments.py ruft diese Funktion auf, wenn semantische Embeddings gebraucht werden.
"""


def encode_with_sentence_bert(texts, model_name="all-MiniLM-L6-v2"):
    # Lädt Sentence-BERT erst hier. So bleiben die TF-IDF-Experimente unabhängig von dieser zusätzlichen Bibliothek.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Bitte sentence-transformers installieren, bevor SBERT-Experimente ausgeführt werden."
        ) from exc

    # all-MiniLM-L6-v2 ist ein kleines vortrainiertes Sentence-BERT-Modell.
    model = SentenceTransformer(model_name)

    # Jeder Biografietext wird in einen Bedeutungsvektor umgewandelt. experiments.py nutzt diese Vektoren danach für Logistic Regression.
    return model.encode(
        list(texts),
        show_progress_bar=True,
        convert_to_numpy=True,
    )
