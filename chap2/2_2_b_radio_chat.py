import openai
import streamlit as st
openai.api_key = st.secrets['OPENAI_API_KEY']
st.title('Radio chat 📻🤖')
m = [{'role': 'system','content': 'If I say hello, say world'}]
models_name = ['gpt-4o-mini', 'gpt-4o']
selected_model = st.radio('Select OpenAI model', models_name)
# st.write(f'Selected model: {selected_model}')
prompt = st.text_input('Enter your message')
if prompt:
    m.append({'role': 'user','content': prompt})
    completion = openai.chat.completions.create(model=selected_model,
                                            messages=m)
    response = completion.choices[0].message.content
    st.write(response) # world
