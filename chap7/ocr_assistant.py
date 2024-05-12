import streamlit as st
from streamlit_paste_button import paste_image_button as pbutton
import requests, base64, io

def encode_image_grab(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def chat_vision(prompt, base64_image, api_key= st.secrets["OPENAI_API_KEY"],max_tokens=500):
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = [{"role": "user",
        "content": [{"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
        }]

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": messages,  
        "max_tokens": max_tokens
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

st.title('OCR 👀')
st.write('Paste an image to perform OCR on it.')

paste_result = pbutton("📋 Paste an image")

prompt = st.text_input('prompt', 'Extract the text from the image')
if paste_result.image_data is not None:
    st.sidebar.write('Pasted image:')
    st.sidebar.image(paste_result.image_data)
    base64_image = encode_image_grab(paste_result.image_data)
    st.write(chat_vision(prompt, base64_image))
