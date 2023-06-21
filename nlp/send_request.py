import requests, json
url = 'http://localhost:5005/model/parse'

query = input("Enter query: ")

# data = '{ "text": "Hello!" }'
data = f'{{ "text": "{query}" }}'

print(data)

response = requests.post(url, data=data)

content = json.loads(response.content)
content = json.dumps(content, indent=4)

print(content)
