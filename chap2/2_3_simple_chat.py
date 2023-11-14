from openai import OpenAI
import json, os
import streamlit as st

st.set_page_config(page_title='Chat',page_icon='🤖')

avatar = {"assistant": "🤖", "user": "🐱"}

# Set the API key for the openai package
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=st.secrets['OPENAI_API_KEY'],
)

# Debug
# st.sidebar.write(st.session_state.convo)

# Functions
def dumb_chat():
  with open('fake/dummy1.json') as f:
    dummy = json.load(f)
  st.write(dummy[1]['content'])
  return dummy[1]['content']

def chat_stream(messages,model='gpt-3.5-turbo'):
  # Generate a response from the ChatGPT model
  completion = client.chat.completions.create(
        model=model,
        messages= messages,
        stream = True
  )
  report = []
  res_box = st.empty()
  # Looping over the response
  for resp in completion:
      if resp.choices[0].finish_reason is None:
          # join method to concatenate the elements of the list 
          # into a single string, then strip out any empty strings
          report.append(resp.choices[0].delta.content)
          result = ''.join(report).strip()
          result = result.replace('\n', '')        
          res_box.write(result) 
  return result


# Initialization
if 'convo' not in st.session_state:
    st.session_state.convo = []

st.sidebar.title('ChatGPT-like bot 🤖')

models_name = ['gpt-3.5-turbo', 'gpt-4']
selected_model = st.sidebar.selectbox('Select OpenAI model', models_name)

# Display the response in the Streamlit app
for line in st.session_state.convo:
    # st.chat_message(line.role,avatar=avatar[line.role]).write(line.content)
    if line['role'] == 'user':
      st.chat_message('user',avatar=avatar['user']).write(line['content'])
    elif line['role'] == 'assistant':
      st.chat_message('assistant',avatar=avatar['assistant']).write(line['content'])

# Create a text input widget in the Streamlit app
if prompt := st.chat_input():
  # Append the text input to the conversation
  with st.chat_message('user',avatar=avatar['user']):
    st.write(prompt)
  st.session_state.convo.append({'role': 'user', 'content': prompt })
  # Query the chatbot with the complete conversation
  with st.chat_message('assistant',avatar=avatar['assistant']):
     result = chat_stream(st.session_state.convo,selected_model)
    #  result = dumb_chat()
  # Add response to the conversation
  st.session_state.convo.append({'role':'assistant', 'content':result})