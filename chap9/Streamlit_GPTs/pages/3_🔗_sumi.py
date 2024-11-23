import streamlit as st
import requests, io
from chat_utils import *

st.set_page_config(page_title='Chains',page_icon='🔗')
st.sidebar.title(f'Sumi 🔗')

if 'avatar' not in st.session_state:
  st.session_state.avatar = {"assistant": "🤖", "user": "🐱"}

avatar = st.session_state.avatar

if 'model' not in st.session_state:
    st.session_state.model = 'gpt-3.5-turbo'
# models_name = ['gpt-3.5-turbo', 'gpt-4o']
# selected_model = st.sidebar.selectbox('Select OpenAI model', models_name)
model = st.session_state.model
st.sidebar.write(f'Selected model: {model}')

chunk_size = st.sidebar.number_input('chunk size',min_value=100, max_value=5000, value=1000, step=100)

# context = st.text_input('Context','summarize the following chapter with bullet points')
context = 'summarize the following chapter'

# file = st.file_uploader('Upload txt file',type='txt')
chapters = [f'chap{i}' for i in range(1,11)]
chap = st.selectbox("Select chapter to summarize",chapters)
maxtokens = {'gpt-3.5-turbo': 16_000,'gpt-4o': 128_000 }
st.sidebar.write("Max tokens:", maxtokens[model])

pdf = {'chap1':'Chap%201%20-%20Intro.pdf',
       'chap2':'Chap%202%20-%20The%20ChatGPT%20API.pdf',
       'chap3':'Chap%203%20-%20Chaining%20%26%20Summarization.pdf',
       'chap4':'Chap%204%20-%20Vector%20search%20%26%20Question%20Answering.pdf',
       'chap5':'Chap%205%20-%20Agents%20%26%20Tools.pdf',
       'chap6':'Chap%206%20-%20Speech-to-Text%20%26%20Text-to-Speech.pdf',
       'chap7':'Chap%207%20-%20Vision.pdf',
       'chap8':'Chap%208%20-%20DALL-E.pdf',
       'chap9':'Chap%209%20-%20Conclusion.pdf',
       'chap10':'Chap%2010%20-%20Appendix.pdf'}

branch = 'bb24362da3358ff9ef95e4ae356d2bfb5095c2ac' # avant la prise de la bastille
# branch = 'main'
url = f'https://raw.githubusercontent.com/yanndebray/programming-GPTs/{branch}/{chap}/{pdf[chap]}'
# st.write(url)
r = requests.get(url)
f = io.BytesIO(r.content)
pages = pdf_to_pages(f)
content = ' '.join(pages)
st.write('pages:',len(pages),'tokens:',num_tokens(content))
# if st.sidebar.checkbox('show pages'):
#     st.write(pages)

# content = st.text_area('Chapter content',content)

if st.button('refine'):
    chunks = my_text_splitter(content,chunk_size=chunk_size)
    st.write('chunks',len(chunks))
    bar = st.progress(0, text='refine')
    summary = summarize(chunks[0],context,model)
    with st.expander('summary 1:'):
        st.write(summary)
    bar.progress(float(1/len(chunks)), text='refine')
    for i in range(1,len(chunks)):
        summary = refine(summary,chunks[i],model)
        with st.expander(f'summary {i+1}:'):
            st.write(summary)
        bar.progress(float((i+1)/len(chunks)), text='refine')
    st.write('**final summary:**')
    st.write(summary)

# download chapter
st.markdown(f'[Download {chap}]({url})')