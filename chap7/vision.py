from openai import OpenAI
import base64, requests, os
import streamlit as st
from PIL import Image

st.set_page_config(page_title='Vision 48',page_icon='👀')

avatar = {"assistant": "👀", "user": "🐱"}

# OpenAI API Key
api_key = st.secrets['OPENAI_API_KEY']

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def chat_vision(messages):
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": messages,  
        "max_tokens": 500
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']

# Initialization
if 'convo' not in st.session_state:
    st.session_state.convo = []

# create the folder vision/img if it doesn't exist
if not os.path.exists('data/image'):
  os.makedirs('data/image')


st.sidebar.title('GPT-4 vision 🤖👀')

with st.sidebar.expander('?'):
    # Example of image download
    st.markdown('### What is in the image?')
    # Download the image from the URL
    img = 'img/funny-corgi-dall-e3.png'
    # Display the downloaded image
    image = Image.open(img)
    st.image(image, caption='Funny corgi in a cartoon style')
    with open(img, 'rb') as file:
        st.download_button(
            label='Download image',
            data=file,
            file_name='funny_corgi.png',
            mime='image/png',
        )
# Add an upload field to the sidebar for images
uploaded_file = st.file_uploader('Upload an image')
if uploaded_file is not None:
    # # Convert the file to an image and display it
    image = Image.open(uploaded_file)
    st.image(image,caption=uploaded_file.name)
    # Save image to disk
    with open('data/image/'+ uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.getvalue())

# Display the response in the Streamlit app
for line in st.session_state.convo:
    # st.chat_message(line.role,avatar=avatar[line.role]).write(line.content)
    if line['role'] == 'user':
      with st.chat_message('user',avatar=avatar['user']):
        st.write(line['content'][0]['text'])
        # st.image(line['content'][1]['image_url']['url'])
    elif line['role'] == 'assistant':
      st.chat_message('assistant',avatar=avatar['assistant']).write(line['content'])

# Create a text input widget in the Streamlit app
prompt = st.chat_input('Ask question about images 👀')

if prompt:
  # Append the text input to the conversation
  with st.chat_message('user',avatar='🐱'):
    st.write(prompt)
  
  # Getting the base64 string
  image_path = 'data/image/'+ uploaded_file.name
  base64_image = encode_image(image_path)  
  st.session_state.convo.append({'role': 'user',
                                'content': [{ "type": "text","text": prompt},
                                            {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
                                })
  # Query the chatbot with the complete conversation
  with st.chat_message('assistant',avatar='🤖'):
     result = chat_vision(st.session_state.convo)
     st.write(result['content'])
  # Add response to the conversation
  st.session_state.convo.append(result)


# Debug
if st.sidebar.toggle('Debug mode',key='debug_mode'):
    st.sidebar.write(st.session_state.convo)