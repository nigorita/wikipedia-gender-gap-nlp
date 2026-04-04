import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download('stopwords')

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

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    df = lowercase_text(df)
    print(df.head())
    df = tokenize_text(df)
    print(df.head())
    df = remove_stopwords(df)
    print(df.head())