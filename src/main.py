import argparse
import pandas as pd
from collections import Counter

from features import *
from model import train_model


# ------------------------
# Parse argument
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    choices=["adj", "full", "full_nosample"],
    default="adj"
)

args = parser.parse_args()


# ------------------------
# Load data
# ------------------------
df = pd.read_csv("data/data_math.csv")

print("Samples:", len(df))


# ------------------------
# Preprocessing
# ------------------------
if args.mode != "full_nosample":
    df["text"] = df["text"].apply(lambda x: sample_text(x, 300))

names = list(df["name"])

df["clean"] = df["text"].apply(lambda x: remove_names(x, names))
df["clean"] = df["clean"].apply(remove_gender_words)


# ------------------------
# Adjectives
# ------------------------
df["adj"] = df["clean"].apply(extract_adjectives)
df["adj"] = df["adj"].apply(filter_adjectives)

freq = Counter(" ".join(df["adj"]).split())
common = {w for w, c in freq.items() if c >= 3}

df["adj_features"] = df["adj"].apply(
    lambda x: " ".join([w for w in x.split() if w in common])
)

df["full_features"] = df["clean"]

df = add_reference_features(df)
print("\nReference ratio:")
print(df.groupby("gender")["ref_ratio"].mean())

# ------------------------
# Run experiment
# ------------------------
if args.mode == "adj":
    train_model(df, "adj_features", "Adjectives only")

elif args.mode == "full":
    train_model(df, "full_features", "Full cleaned text")

elif args.mode == "full_nosample":
    train_model(df, "full_features", "Full cleaned text (no sampling)")