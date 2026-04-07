import pandas as pd


INPUT_FILE = "data.csv"
OUTPUT_FILE = "cleaned_data.csv"


def load_data(path):
    # Load the csv file.
    return pd.read_csv(path)


def remove_missing_text(df):
    # Remove rows without text.
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    return df


def remove_duplicates(df):
    # Remove duplicate names.
    return df.drop_duplicates(subset=["name"])


def print_stats(df, title):
    # Print simple dataset information.
    print(title)
    print("Rows:", len(df))
    print("Gender counts:")
    print(df["gender"].value_counts())
    print()


def main():
    df = load_data(INPUT_FILE)
    print_stats(df, "Original dataset")

    df = remove_missing_text(df)
    df = remove_duplicates(df)
    print_stats(df, "Cleaned dataset")

    df.to_csv(OUTPUT_FILE, index=False)
    print("Saved cleaned data to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
