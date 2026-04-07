import requests
import pandas as pd

try:
    from generated_names import male_names, female_names
except ImportError:
    from sample_names import male_names, female_names


def test_connection():
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/Albert_Einstein"
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        print("Success")
    else:
        print("Failed", response.status_code)


def get_lead_text(title):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("extract")
    else:
        return None



def fetch_and_save():
    data = []

    for name in male_names:
        text = get_lead_text(name)
        data.append({"name": name, "gender": "male", "text": text})

    for name in female_names:
        text = get_lead_text(name)
        data.append({"name": name, "gender": "female", "text": text})

    df = pd.DataFrame(data)
    df.to_csv("data.csv", index=False)

if __name__ == "__main__":
    fetch_and_save()
