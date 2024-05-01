import requests, datetime, openai, os, re
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from pydub import AudioSegment
# from elevenlabs.client import ElevenLabs
# from elevenlabs import save

# import tomli
# with open(".streamlit/secrets.toml","rb") as f:
#     secrets = tomli.load(f)
# os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
# os.environ["ELEVEN_API_KEY"] = secrets["ELEVEN_API_KEY"]

# client11 = ElevenLabs()
# voices_list = list(client11.voices.get_all())

def rss(url,episode,date):
  response = requests.get(url)    
  # Parse the XML document
  tree = ET.fromstring(response.content)
  channel = tree.find('channel')
  # Save the text and the rss feed to files
  with open(f"podcast/{episode}/rss/techcrunch_{date}.xml", "wb") as f:
    f.write(response.content)
  return channel

def scrape_article(item,episode):
  title = item.find('title').text
  title = re.sub(r'[<>:"/\\|?*]', '-', title)
  link = item.find('link').text
  html = requests.get(link).text
  soup = BeautifulSoup(html, "html.parser")
  # extract only the text from the class article-content
  text = soup.find(class_="article-content").get_text()
  # Save the text to file
  with open(f"podcast/{episode}/text/{title}.txt", "w", encoding="utf-8") as f:
      f.write(text)
  return (title,link,text)

def openai_tts(text,speech_file_path):
  response = openai.audio.speech.create(
    model="tts-1",
    voice=voice,
    input=text
  )
  response.stream_to_file(speech_file_path)

def elevenlabs_tts(text,speech_file_path):
   # Generate audio for the content
    audio = client11.generate(
        text=text,
        voice=voice,
        model="eleven_multilingual_v2"
    )
    save(audio,speech_file_path)

def text_splitter(text):
  # Split text into chunks of 4000 characters 
  chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
  return chunks

def split_tts(text,speech_file_path):
  chunks = text_splitter(text)
  for i,chunk in enumerate(chunks):
      chunk_file_path = speech_file_path.replace(".mp3",f" - chunk {i+1}.mp3")
      openai_tts(chunk, chunk_file_path)
      print(f"Chunk {i+1} - Characters: {len(chunk)} - File: {chunk_file_path}")
  files = [speech_file_path.replace(".mp3",f" - chunk {i+1}.mp3") for i in range(len(chunks))]
  merge_audio_files(files, speech_file_path)

def merge_audio_files(files, output_file):
    combined = AudioSegment.empty()
    for file in files:
        sound = AudioSegment.from_file(file)
        combined += sound
        os.remove(file)
    combined.export(output_file, format="mp3")

if __name__ == "__main__":
   
  # voice = 'enYann'
  voice = "alloy"
  url = "https://techcrunch.com/feed"
  date = datetime.datetime.now().strftime("%Y-%m-%d") # date in format YYYY-MM-DD

  episode_number = 42
  # episode_number = int(episode_number)
  episode = 'tech' + '{:03d}'.format(episode_number)
  # Set the directory for the episode
  directory = 'podcast/' + episode 
  if not os.path.exists('podcast'):
    os.mkdir('podcast')
  if not os.path.exists(directory):
    os.mkdir(directory)
    os.mkdir(directory + '/audio')
    os.mkdir(directory + '/text')
    os.mkdir(directory + '/rss')

  # retrieve the rss feed
  channel = rss(url,episode,date)
  # Print the channel title
  print(f"Channel Title: {channel.find('title').text}")
  # Print the channel description
  print(f"Channel Description: {channel.find('description').text}")
  # Print the channel link
  print(f"Channel Link: {channel.find('link').text}")
  # Print the channel last build date
  print(f"Last Build Date: {channel.find('lastBuildDate').text}")

  items = channel.findall('item')
  for item in items[0:5]:
    # scrape the article
    (title,link,text) = scrape_article(item,episode)
    # TTS
    speech_file_path = f"podcast/{episode}/audio/{title}.mp3"
    split_tts(text,speech_file_path)
    print(f"New episode: {speech_file_path}")