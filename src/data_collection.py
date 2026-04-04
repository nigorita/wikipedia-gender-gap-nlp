import requests

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

if __name__ == "__main__":
    test_connection()