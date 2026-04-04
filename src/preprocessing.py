import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from nltk import pos_tag


nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

def load_data():
    df = pd.read_csv("data.csv")
    return df


def lowercase_text(df):
    df["text"] = df["text"].str.lower()
    return df

def tokenize_text(df):
    df["tokens"] = df["text"].apply(nltk.word_tokenize)
    return df

def remove_stopwords(df):
    stop_words = set(stopwords.words('english'))
    df["tokens"] = df["tokens"].apply(lambda tokens: [word for word in tokens if word not in stop_words])
    return df

def remove_punctuation(df):
    df["tokens"] = df["tokens"].apply(lambda tokens: [word for word in tokens if word not in string.punctuation])
    return df

def pos_tagging(df):
    df["pos_tags"] = df["tokens"].apply(pos_tag)
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    df = lowercase_text(df)
    print(df.head())
    df = tokenize_text(df)
    print(df.head())
    df = remove_stopwords(df)
    print(df.head())
    df = remove_punctuation(df)
    print(df.head())
    df = pos_tagging(df)
    print(df.head())