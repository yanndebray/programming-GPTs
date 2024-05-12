import streamlit as st
import pandas as pd
import numpy as np
import os, requests, cv2
import xml.etree.ElementTree as ET

# This function loads an image from Streamlit public repo on S3. We use st.cache on this
# function as well, so we can reuse the images across runs.
@st.cache_data(show_spinner=False)
def load_image(url):
    with requests.get(url) as response:
        image = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]] # BGR -> RGB
    return image

# Path to the public S3 bucket
DATA_URL_ROOT = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"
bucket_list = ET.fromstring(requests.get(DATA_URL_ROOT).content)
xpath = './/{http://s3.amazonaws.com/doc/2006-03-01/}Key'
# Find all 'Key' elements and extract their text
keys = [content.text for content in bucket_list.findall(xpath)]

st.sidebar.title("Self driving dataset")
id = st.sidebar.slider("Frame",0,len(keys),0)
selected_frame = keys[id]

image_url = os.path.join(DATA_URL_ROOT, selected_frame)
image = load_image(image_url)
st.image(image)