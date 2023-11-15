import streamlit as st

messages = [{'role': 'user','content': 'hello'},
            {'role': 'assistant','content': 'world'}]

for line in messages:
    st.chat_message(line['role']).write(line['content'])

if prompt := st.chat_input('Enter your message'):
    messages.append({'role': 'user','content': prompt})
