from bs4 import BeautifulSoup
import requests, openai, os, tomli

with open(".streamlit/secrets.toml", "rb") as f:
    secrets = tomli.load(f)
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

url = "https://www.wired.com/story/rss-readers-feedly-inoreader-old-reader/"
html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")
# extract only the text from the article tag
text = soup.find("article").get_text()
# print(text)

# Save the text to a file
with open("data/txt/wired_article.txt", "w", encoding="utf-8") as f:
    f.write(text)

# OpenAI TTS
text = text[:4000]  # limit to 4000 characters
speech_file_path = "data/audio/wired_article.mp3"
response = openai.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=text
)
response.stream_to_file(speech_file_path)
print(f"Audio file saved to {speech_file_path}")