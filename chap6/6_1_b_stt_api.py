from pathlib import Path
import openai,tomli, os
with open("../.streamlit/secrets.toml","rb") as f:
    secrets = tomli.load(f)
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
file = '../data/audio/enYann-tale_of_two_cities.mp3'
file_path = Path(file)
transcription = openai.audio.transcriptions.create(model="whisper-1", file=file_path)
print(transcription.text)