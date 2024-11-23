import streamlit as st
from chat_utils import *
from st_audiorec import st_audiorec

st.set_page_config(page_title='Jarvis',page_icon="🎙️")
st.sidebar.title('Jarvis 🎙️')

if 'avatar' not in st.session_state:
  st.session_state.avatar = {"assistant": "🔊", "user": "🎙️"}

avatar = st.session_state.avatar

# Initialization
if 'convo' not in st.session_state:
    st.session_state.convo = []

if 'model' not in st.session_state:
    st.session_state.model = 'gpt-3.5-turbo'
selected_model = st.session_state.model
st.sidebar.write(f'Selected model: {selected_model}')

# Display the conversation
for line in st.session_state.convo:
    # st.chat_message(line.role,avatar=avatar[line.role]).write(line.content)
    if line['role'] == 'user':
      st.chat_message('user',avatar=avatar['user']).write(line['content'])
    elif line['role'] == 'assistant':
      st.chat_message('assistant',avatar=avatar['assistant']).write(line['content'])


with st.sidebar.container():
  wav_audio_data = st_audiorec()

if wav_audio_data:
    with open('audio.wav','wb') as f:
        f.write(wav_audio_data)
    res = stt(open('audio.wav','rb'))
    st.chat_message('user',avatar=avatar['user']).write(res.text)
    st.session_state.convo.append({'role': 'user', 'content': res.text})
    with st.chat_message('assistant',avatar=avatar['assistant']):
      result = chat_stream(st.session_state.convo,selected_model)
    # Add response to the conversation
    st.session_state.convo.append({'role':'assistant', 'content':result})

# # Debug
# st.sidebar.write(st.session_state.convo)
