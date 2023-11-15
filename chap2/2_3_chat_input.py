import streamlit as st

messages = [{'role': 'user','content': 'hello'},
            {'role': 'assistant','content': 'world'}]

if prompt := st.chat_input('Enter your message'):
    messages.append({'role': 'user','content': prompt})

for line in messages:
    if line['role'] == 'user':
        with st.chat_message('user'):
            st.write(line['content'])
    elif line['role'] == 'assistant':
        with st.chat_message('assistant'):
            st.write(line['content'])
