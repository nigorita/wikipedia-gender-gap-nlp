"""Sentence-BERT-Hilfsfunktionen für Phase 2.

Zuständig: Tugba

Diese Datei ist bewusst klein gehalten. Sie kann aus experiments.py verwendet
werden, sobald das Paket sentence-transformers installiert ist.
"""


def encode_with_sentence_bert(texts, model_name="all-MiniLM-L6-v2"):
    # Lädt Sentence-BERT nur dann, wenn diese Funktion wirklich verwendet wird.
    # So bleiben die normalen TF-IDF-Experimente unabhängig von SBERT.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Bitte sentence-transformers installieren, bevor SBERT-Experimente ausgeführt werden."
        ) from exc

    # all-MiniLM-L6-v2 ist ein kleines, schnelles Sentence-BERT-Modell.
    model = SentenceTransformer(model_name)

    # Jeder Biografietext wird in einen numerischen Bedeutungsvektor umgewandelt.
    # Diese Embeddings können danach mit Logistic Regression klassifiziert werden.
    return model.encode(
        list(texts),
        show_progress_bar=True,
        convert_to_numpy=True,
    )
