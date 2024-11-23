import streamlit as st
from chat_utils import *
import openai, faiss
import numpy as np

st.set_page_config(page_title='Ragu',page_icon='🔍')
st.sidebar.title(f'Ragu 🔍')

pages = np.load('chap9/pages.npy').tolist()

def get_embedding(text, model="text-embedding-3-small"):
   return openai.embeddings.create(input = [text], model=model).data[0].embedding

# Step 1: Load the embeddings as numpy array from file
embeddings = np.load('chap9/embeddings.npy') # (140 vectors, each 1536-dimensional)
# Step 2: Create a FAISS index (FlatL2 for Euclidean distance)

d = 1536  # Dimensionality of each vector (1536 in this case)
index = faiss.IndexFlatL2(d)  # You can also use other index types (e.g., IndexFlatIP for cosine similarity)

# Step 3: Add the embeddings to the index
index.add(embeddings)  # Now, the index contains 140 vectors

query = st.text_input("Ask a question")
if query:
    query_embedding = np.array([get_embedding(query)]) 
    _,indices = index.search(query_embedding, k=5)
    prompt = f'''
    Answer the question based on the following context: 
    question: {query}
    context: {'/n'.join([pages[i] for i in indices[0]])}
    '''
    m = [{'role':'user','content':prompt}]
    chat_stream(m)
    with st.expander('Source documents'):
        st.write(indices)
        for i in indices[0]:
            st.write('---------------------------------')
            st.write(f'Page: {i}')
            st.write(pages[i])

examples = [
            "What are the vector databases mentioned in the book?",
            "What are agent?",
            "What is the name of the agentic framework introduced in this book?"
        ]
with st.sidebar.expander('Prompt examples'):
    for ex in examples:
        st.write(ex)