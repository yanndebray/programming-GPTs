import os
import streamlit as st
from chat_utils import *

st.set_page_config(page_title='Chat',page_icon="🤖")
st.sidebar.title('ChatGPT-like bot 🤖')

if 'avatar' not in st.session_state:
  st.session_state.avatar = {"assistant": "🤖", "user": "🐱"}

avatar = st.session_state.avatar

# Initialization
if 'convo' not in st.session_state:
    st.session_state.convo = []

n = len(os.listdir('chat'))
if 'id' not in st.session_state:
    st.session_state.id = n

id = st.session_state.id

if 'model' not in st.session_state:
    st.session_state.model = 'gpt-3.5-turbo'
# models_name = ['gpt-3.5-turbo', 'gpt-4o']
# selected_model = st.sidebar.selectbox('Select OpenAI model', models_name)
selected_model = st.session_state.model
st.sidebar.write(f'Selected model: {selected_model}')

if st.sidebar.button(f'New Chat {avatar["user"]}'):
   new_chat()
for file in sorted(os.listdir('chat')):
  filename = file.replace('.json','')
  if st.sidebar.button(f'💬 {filename}'):
     select_chat(file)

# if st.sidebar.toggle('Export chat'):
#   zip_chat()
#   st.sidebar.download_button('Download chat',open('chat.zip', 'rb'),'chat.zip',mime='application/zip')

# Display the response in the Streamlit app
for line in st.session_state.convo:
    # st.chat_message(line.role,avatar=avatar[line.role]).write(line.content)
    if line['role'] == 'user':
      st.chat_message('user',avatar=avatar['user']).write(line['content'])
    elif line['role'] == 'assistant':
      st.chat_message('assistant',avatar=avatar['assistant']).write(line['content'])

# Create a text input widget in the Streamlit app
prompt = st.chat_input(f'convo{st.session_state.id}')

if prompt:
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
  save_chat(id)

# Debug
# st.sidebar.write(st.session_state.convo)
