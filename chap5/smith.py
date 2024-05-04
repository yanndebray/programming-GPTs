import openai, json, requests, os, tomli
with open("../.streamlit/secrets.toml","rb") as f:
    secrets = tomli.load(f)
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
os.environ["SERPAPI_API_KEY"] = secrets["SERPAPI_API_KEY"]

def agent(prompt, system='', tools=None):
  # Generate a response from the ChatGPT model
  messages = []
  if tools:
    tool_name = pick_tool(prompt, tools=tools)
    if tool_name == 'search':
      system = search_tool(prompt)
      print(f"tool: search")
    elif tool_name == 'python':
      system = python_tool()
      print(f"tool: python")
    else: # Probably "null"
      print('sorry, could not use a tool for this request')
    print("------------------")
  if system:
    messages.append({'role': 'system', 'content': system})
  messages.append({'role': 'user', 'content': prompt })
  completion = openai.chat.completions.create(
      model='gpt-3.5-turbo',
        messages= messages
  )
  return completion.choices[0].message.content

def load_tools():
  # TODO: update calling sequence to take a list of tools
  # load_tools(['search','python'])
  with open("tools.jsonl","r") as f:
    tools_list = json.load(f)
  return tools_list

def pick_tool(prompt,tools):
  # Use ChatGPT to decide which tool to use
  # Arg:
  # prompt : str 
  # tools : List of Dict for each tool
  # Out: 
  # tool_name : str of the tool name

  name = [p['name'] for p in tools]
  desc = {p['name']:p['desc'] for p in tools}
  examples = {p['name']:p['example'] for p in tools}

  promptIntro = '''You are an agent that has access to tools to perform action. 
  For every prompt you receive, you will decide which available tool to use from the following list:
  '''
  promptCore = ''.join([f'- {n} : {desc[n]}\n' for n in name])
  promptExample = '''
  Here's some examples of prompts you'll get and the response you should give:

  ''' + ''.join([f'  USER: {examples[n]}\n  BOT: {n}\n\n' for n in name])
  promptEnd = '''
  Give a single word answer with the tool chosen to answer the prompt.
  If you're very confident that you don't need extra information, respond with the string "null".
  '''

  toolDecisionPrompt = promptIntro+promptCore+promptExample+promptEnd 
  tool_name = agent(prompt,system=toolDecisionPrompt)
  return tool_name

def search_tool(prompt):
  # Set the API endpoint
  api_endpoint = "https://serpapi.com/search"
  # Set your API key
  api_key = os.environ['SERPAPI_API_KEY']
  # Set the search parameters
  params = {
      "q": prompt,
      "api_key": api_key,
  }
  # Send the request
  response = requests.get(api_endpoint, params=params)
  results = json.loads(response.content)
  with open("search_results.json","w") as f:
    json.dump(results,f,indent=4)
  # Build system promt from featured snippet answer if it exists
  if "answer" in results["answer_box"]:
    searchPrompt = results["answer_box"]["answer"]
  else:
    searchPrompt = results["answer_box"]["title"]
  system = f'''Answer the user request given the following information retrieved from an internet search:
  {searchPrompt}
  '''
  return system

def python_tool():
  system = '''Generate python code from the request.
  Give only the code, no explanation in natural language.
  '''
  return system

if __name__ == "__main__":
  prompt = "Who is the CEO of Twitter?"
  print(agent(prompt,tools=load_tools()))