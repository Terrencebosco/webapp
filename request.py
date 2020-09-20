import requests

url = 'http://localhost:5000/api'

r = requests.post(url)
print(r.json())