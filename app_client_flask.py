import requests

# Endpoint URL
url = "http://localhost:5000/query"
API_TOKEN = "secret-token-123"
# Headers including Authorization and Content-Type
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# JSON payload
payload = {
    "query": "What is the H-1B visa process?"
}

# Send POST request
response = requests.post(url, headers=headers, json=payload)

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
