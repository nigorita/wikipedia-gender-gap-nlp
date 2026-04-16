import os
import requests
import pandas as pd
import time


def get_selected_sections(title):
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "NLP-Project"}
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json"
    }

    try:
        r = requests.get(url, params=params, headers=headers)
        if r.status_code != 200:
            return None

        pages = r.json().get("query", {}).get("pages", {})

        for page in pages.values():
            text = page.get("extract", "")
            lines = text.split("\n")

            selected, keep = [], False

            for line in lines:
                l = line.lower()

                if any(x in l for x in ["early life", "life", "career", "biography"]):
                    keep = True
                    continue

                if any(x in l for x in ["references", "external links"]):
                    keep = False

                if keep:
                    selected.append(line)

            return " ".join(selected)

    except:
        return None


def fetch_and_save():
    df_manual = pd.read_csv("data/manual_math_dataset.csv")

    existing = set()
    if os.path.exists("data/data_math.csv"):
        df_old = pd.read_csv("data/data_math.csv")
        existing = set(df_old["name"])
    else:
        df_old = None

    rows = []

    for _, row in df_manual.iterrows():
        name = row["name"]

        if name in existing:
            continue

        print("Fetching:", name)

        text = get_selected_sections(name)

        if text:
            rows.append({
                "name": name,
                "gender": row["gender"],
                "field": row["field"],
                "text": text
            })

        time.sleep(1)

    df_new = pd.DataFrame(rows)

    if df_old is not None:
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df = df.drop_duplicates(subset=["name"])
    df.to_csv("data/data_math.csv", index=False)

    print("Saved:", len(df))


if __name__ == "__main__":
    fetch_and_save()