import pandas as pd

def load_data():
    df = pd.read_csv("data.csv")
    return df


def lowercase_text(df):
    df["text"] = df["text"].str.lower()
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    df = lowercase_text(df)
    print(df.head())