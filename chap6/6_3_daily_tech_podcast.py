import requests, datetime, openai
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import tomli, os
with open("../.streamlit/secrets.toml","rb") as f:
    secrets = tomli.load(f)
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
url = "https://techcrunch.com/feed"
date = datetime.datetime.now().strftime("%Y-%m-%d") # date in format YYYY-MM-DD
response = requests.get(url)
with open(f"../data/rss/techcrunch_{date}.xml", "wb") as f:
    f.write(response.content)
# Parse the XML document
tree = ET.fromstring(response.content)
channel = tree.find('channel')
item = channel.find('item')
title = item.find('title').text
link = item.find('link').text
html = requests.get(link).text
soup = BeautifulSoup(html, "html.parser")
# extract only the text from the class article-content
text = soup.find(class_="article-content").get_text()
# Save the text to a file
with open(f"../data/txt/{title}.txt", "w", encoding="utf-8") as f:
    f.write(text)
# OpenAI TTS
speech_file_path = f"../data/audio/{title}.mp3"
response = openai.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input=text
)
response.stream_to_file(speech_file_path)