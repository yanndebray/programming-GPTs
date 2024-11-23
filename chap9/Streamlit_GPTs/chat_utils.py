import streamlit as st
from openai import OpenAI
import json, os, shutil
from bs4 import BeautifulSoup
import requests
import tiktoken
import requests, pypdf

def num_tokens(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding_name = 'cl100k_base'
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Set the API key for the openai package
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=st.secrets['OPENAI_API_KEY'],
)

# Functions
def new_chat():
   st.session_state.convo = []
   st.session_state.id += 1

def save_chat(n):
  file_path = f'chat/convo{n}.json'
  with open(file_path,'w') as f:
    json.dump(st.session_state.convo, f, indent=4)

def select_chat(file):
  st.session_state.convo = []
  with open(f'chat/{file}') as f:
    st.session_state.convo = json.load(f)
  st.session_state.id = int(file.replace('.json','').replace('convo',''))

def dumb_chat():
  with open('fake/dummy1.json') as f:
    dummy = json.load(f)
  st.write(dummy[1]['content'])
  return dummy[1]['content']

def chat_stream(messages,model='gpt-4o-mini'):
  # Generate a response from the ChatGPT model
  completion = client.chat.completions.create(
        model=model,
        messages= messages,
        stream = True
  )
  report = []
  res_box = st.empty()
  # Looping over the response
  for resp in completion:
      if resp.choices[0].finish_reason is None:
          # join method to concatenate the elements of the list 
          # into a single string, then strip out any empty strings
          report.append(resp.choices[0].delta.content)
          result = ''.join(report).strip()
          result = result.replace('\n', '')        
          res_box.write(result) 
  return result

def zip_chat():
  # Zip the chat folder
  shutil.make_archive('chat', 'zip', 'chat')
  st.toast('Chat zipped!',icon = "🥞")

@st.experimental_dialog("Cast your vote")
def delete_chat():
    st.write("❌ Are you sure you want to delete the chat history?")
    if st.button("Delete chat history"):
        shutil.rmtree('chat')
        os.mkdir('chat')
        st.rerun() 

def my_text_splitter(text,chunk_size=3000):
    # Split text into chunks based on space or newline
    chunks = text.split()

    # Initialize variables
    result = []
    current_chunk = ""

    # Concatenate chunks until the total length is less than 4096 tokens
    for chunk in chunks:
        # if len(current_chunk) + len(chunk) < 4096:
        if num_tokens(current_chunk+chunk) < chunk_size:
            current_chunk += " " + chunk if current_chunk else chunk
        else:
            result.append(current_chunk.strip())
            current_chunk = chunk
    if current_chunk:
        result.append(current_chunk.strip())

    return result

def summarize(text, context = 'summarize the following text:', model = 'gpt-3.5-turbo'):
    """Returns the summary of a text."""
    completion = client.chat.completions.create(
        model = model,
        messages=[
        {'role': 'system','content': context},
        {'role': 'user', 'content': text}
            ]
    )
    return completion.choices[0].message.content

def refine(summary, chunk,  model = 'gpt-3.5-turbo'):
    """Refine the summary with each new chunk of text"""
    context = "Refine the summary with the following context: " + summary
    summary = summarize(chunk, context, model)
    return summary


def pdf_to_pages(file):
	"extract text (pages) from pdf file"
	pages = []
	pdf = pypdf.PdfReader(file)
	for p in range(len(pdf.pages)):
		page = pdf.pages[p]
		text = page.extract_text()
		pages += [text]
	return pages

def search(prompt):
  url = f'https://www.google.com/search?q={prompt}'
  html = requests.get(url).text
  # Get the text of the webpage
  soup = BeautifulSoup(html, "html.parser")
  text = soup.get_text()
  return text

def stt(audio):
  return client.audio.transcriptions.create(model="whisper-1", file=audio)