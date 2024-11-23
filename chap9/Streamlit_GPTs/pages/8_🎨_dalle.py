from openai import OpenAI
import streamlit as st

st.set_page_config(page_title='dalle',page_icon='🎨')

# Set the API key for the openai client
client = OpenAI(api_key = st.secrets['OPENAI_API_KEY'])

model = ["dall-e-3", "dall-e-2"]

size = {"dall-e-3": ["1024x1024", "1792x1024", "1024x1792"], 
        "dall-e-2": ["256x256", "512x512", "1024x1024"]}

st.sidebar.title("dalle 🎨")

m = st.sidebar.radio("Select Model", model)
s = st.sidebar.radio("Select Size", size[m])
# Create a text input widget in the Streamlit app
prompt = st.text_input(f"Prompt {m}:",
    "a funny corgi")
    
if st.button("Generate Image"):

    # Generate an image from the DALL-E model
    image = client.images.generate(
        model=m,
        prompt=prompt,
        n=1,
        size=s,
        response_format="url"
    )

    # Display the generated image in the Streamlit app
    st.image(image.data[0].url)
    # st.write(image.data[0].url)