import requests

res = requests.get("http://localhost:5000/measure")
print(res.json())
