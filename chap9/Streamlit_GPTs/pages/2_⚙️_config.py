import streamlit as st
from chat_utils import *

if 'avatar' not in st.session_state:
  avatar = {"assistant": "🤖", "user": "🐱"}

# Set the page title and icon
st.set_page_config(page_title='Config',page_icon='⚙️')

# Select model

st.session_state.model = st.selectbox('Select Model', ['gpt-3.5-turbo', 'gpt-4o'])

col1,col2 = st.columns(2)
with col1:
  assistant = st.radio('Select Assistant', ['🤖', '🦜', '🐍', '🙊'])
with col2:
  user = st.radio('Select User', ['🐱', '🙂', '🤓', '🤐'])
st.session_state.avatar = {"assistant": assistant, "user": user}

if st.toggle('Export history'):
  zip_chat()
  st.download_button('Download chat history',open('chat.zip', 'rb'),'chat.zip',mime='application/zip')

if st.button('Delete history'):
  delete_chat()

  