import openai
import re
import streamlit as st
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Define a function to generate an image using the OpenAI API
def get_img(prompt):
    with st.spinner('Rendering World 🌎 ...'):
        try:
            response = openai.images.generate(
                prompt=prompt,
                n=1,
                size="512x512"
                )
            img_url = response.data[0].url
        except Exception as e:
            # if it fails (e.g. if the API detects an unsafe image), use a default image
            img_url = "https://pythonprogramming.net/static/images/imgfailure.png"
        return img_url

# Define a function to generate a chat response using the OpenAI API
def chat(messages):
    # Generate a chat response using the OpenAI API
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    # Grab just the text from the API completion response
    response = completion.choices[0].message.content
    # Return the generated response
    return response

def follow_up(next):
    st.session_state.messages.append({"role": "user", "content": next})
    response = chat(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})


st.title("Interactive Story Game 🪄")

# Initialize the message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": """
         You are an interactive story game bot that proposes some hypothetical fantastical situation where the user needs to pick from 2-4 options that you provide. 
         Once the user picks one of those options, you will then state what happens next and present new options, and this then repeats. 
         If you understand, say, OK, and begin when I say "begin." When you present the story and options, present just the story and start immediately with the story, no further commentary, and then options like "Option 1:" "Option 2:" ...etc."""},
        {"role": "assistant", "content": f"""OK, I understand. Begin when you're ready."""}]

# If the user has started the game, display the story and options
if len(st.session_state.messages) <= 2:
    # if st.button("Begin"):
    with st.spinner('Initializing World 🌎 ...'):
        # Generate a chat response with an initial message ("Begin")
        st.session_state.messages.append({"role": "user", "content": "Begin"})
        response = chat(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
        text = response.split("Option 1")[0]
        # Using regex, grab the natural language options from the initial response
        options = re.findall(r"Option \d:.*", response)
        img_url = get_img(text)

        # Display the story text and image
        st.write(text)
        st.image(img_url)
        # st.image("https://pythonprogramming.net/static/images/imgfailure.png")

        for option in options:
            st.button(option, on_click=follow_up, args=(option,))
else:
    # Extract the text from the previous response
    response = st.session_state.messages[-1]["content"]
    text = response.split("Option 1")[0]
    # Using regex, grab the natural language options from the initial response
    options = re.findall(r"Option \d:.*", response)
    img_url = get_img(text)

    # Display the story text and image
    st.write(text)
    st.image(img_url)
    # st.image("https://pythonprogramming.net/static/images/imgfailure.png")

    for option in options:
        st.button(option, on_click=follow_up, args=(option,))

if st.checkbox("debug"):
    st.sidebar.write(st.session_state.messages)