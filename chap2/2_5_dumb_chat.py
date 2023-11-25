import streamlit as st
import json

# Functions
def dumb_chat():
    return "Hello world"

def dumb_chat2():
    with open('chat/convo0.json') as f:
        dummy = json.load(f)
    return dummy[1]['content']

if prompt := st.chat_input():
    with st.chat_message('user'):
        st.write(prompt)
    with st.chat_message('assistant'):
        result = dumb_chat()
        st.write(result)