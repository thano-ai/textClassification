import requests

url = "http://127.0.0.1:5000/classify"
data = {"text": "with Cybersecurity techniques"}

response = requests.post(url, json=data)
print(response.json())  # Print the classification result
