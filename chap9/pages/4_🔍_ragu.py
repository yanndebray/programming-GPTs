import streamlit as st
from chat_utils import *
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
chat = ChatOpenAI(model_name='gpt-3.5-turbo')
db = Chroma(persist_directory="chap9/chroma", embedding_function=OpenAIEmbeddings())
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})

st.set_page_config(page_title='Ragu',page_icon='🔍')
st.sidebar.title(f'Ragu 🔍')

if 'avatar' not in st.session_state:
  st.session_state.avatar = {"assistant": "🤖", "user": "🐱"}

avatar = st.session_state.avatar

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

query = st.text_input("Ask a question")
if query:
    qa = RetrievalQA.from_chain_type(
        llm=chat, chain_type="stuff", retriever=retriever, return_source_documents=True)
    result = qa.invoke({"query": query})
    # st.write(result)
    st.write(result['result'])
    with st.expander('Source documents'):
        for doc in result['source_documents']:
            st.write('---------------------------------')
            st.write('Page: ',doc.metadata['page'])
            st.write('Source: ',doc.metadata['source'])
            st.write(doc.page_content)

examples = [
            "What are the vector databases mentioned in the book?",
        ]
with st.sidebar.expander('Prompt examples'):
    for ex in examples:
        st.write(ex)