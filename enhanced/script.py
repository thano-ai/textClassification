import requests

url = "http://127.0.0.1:5000/classify"
data = {"text":'''
'''  }  # <-- Change this text as needed

try:
    response = requests.post(url, json=data)
    print(response.json())  # Raw JSON response
except Exception as e:
    print({"error": str(e)})