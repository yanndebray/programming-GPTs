import openai
import streamlit as st
openai.api_key = st.secrets['OPENAI_API_KEY']
st.title('Button chat 👉🅱️🤖')
m = [{'role': 'system','content': 'If I say hello, say world'}]
prompt = st.text_input('Enter your message')
if st.button('Send'):
    m.append({'role': 'user','content': prompt})
    completion = openai.chat.completions.create(model='gpt-4o-mini',
                                            messages=m)
    response = completion.choices[0].message.content
    st.write(response) # world