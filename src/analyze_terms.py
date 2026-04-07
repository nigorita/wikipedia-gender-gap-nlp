from collections import Counter

import pandas as pd


INPUT_FILE = "features.csv"
OUTPUT_FILE = "term_frequencies.csv"


def load_data():
    # Load the feature file.
    return pd.read_csv(INPUT_FILE)


def count_words(df, gender, column_name):
    # Count words for one gender and one feature column.
    text = " ".join(df[df["gender"] == gender][column_name].fillna(""))
    words = text.split()
    return Counter(words)


def print_top_words(counter, title, top_n=10):
    # Print the most common words.
    print(title)
    for word, count in counter.most_common(top_n):
        print(word, count)
    print()


def build_frequency_rows(counter, gender, feature_type):
    # Convert counts into rows for a csv file.
    rows = []
    for word, count in counter.most_common():
        rows.append(
            {
                "gender": gender,
                "feature_type": feature_type,
                "word": word,
                "count": count,
            }
        )
    return rows


def main():
    df = load_data()

    female_adjectives = count_words(df, "female", "adjective_text")
    male_adjectives = count_words(df, "male", "adjective_text")
    female_nouns = count_words(df, "female", "noun_text")
    male_nouns = count_words(df, "male", "noun_text")

    print_top_words(female_adjectives, "Top female adjectives")
    print_top_words(male_adjectives, "Top male adjectives")
    print_top_words(female_nouns, "Top female nouns")
    print_top_words(male_nouns, "Top male nouns")

    rows = []
    rows.extend(build_frequency_rows(female_adjectives, "female", "adjective"))
    rows.extend(build_frequency_rows(male_adjectives, "male", "adjective"))
    rows.extend(build_frequency_rows(female_nouns, "female", "noun"))
    rows.extend(build_frequency_rows(male_nouns, "male", "noun"))

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print("Saved term frequencies to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
