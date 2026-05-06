# Wikipedia Gender Bias NLP Project

## Overview

Analyze gender bias in Wikipedia biographies using NLP and machine learning.

---

## Data

* Input: `data/manual_math_dataset.csv` (name, gender, field)
* Output: `data/data_math.csv` (biography text)

Text is fetched from Wikipedia API and limited to biography sections.

---

## Preprocessing

* Remove names and gender words
* Optional: sample 300 words
* Clean text → `clean`

---

## Features

* **Adjectives** → extracted + filtered → `adj_features`
* **Full text** → cleaned text → `full_features`
* **Reference features**:

  * `total_refs`, `words`, `ref_ratio`

---

## Model

* TF-IDF + Logistic Regression
* Train/test split: 80/20 (stratified)

---

## Run

```bash
python src/main.py --mode adj
python src/main.py --mode full
python src/main.py --mode full_nosample
```

---

## Output

* Classification report (precision, recall, F1)
* Top words:

  * negative → female
  * positive → male

---

## Goal

Identify linguistic patterns that differ between male and female biographies.
