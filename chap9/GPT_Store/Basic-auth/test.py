import requests

# api_key = 'b1b15e88fa797225412429c1c50c122a1' 
api_key = 'My_API_Key'
# URL = 'http://localhost:8080'
URL = 'https://8513-73-68-198-103.ngrok-free.app'
endpoint = '/weather'
headers = {
    'Authorization': f'Basic {api_key}'
}
params = {
    'city': 'London'  # Replace with the desired city
}

response = requests.get(URL + endpoint, headers=headers, params=params)

print(response.status_code)
print(response.json())