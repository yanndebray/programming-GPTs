import ollama
response = ollama.chat(model='llama3:8b', messages=[
  {
    'role': 'user',
    'content': 'Hello world',
  },
])
print(response['message']['content'])