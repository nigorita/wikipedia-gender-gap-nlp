import requests
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



def fetch_samples():
    for name in female_names:
        text = get_lead_text(name)
        print("FEMALE:", name)
        print(text)
        print("-" * 50)

if __name__ == "__main__":
    fetch_samples()
