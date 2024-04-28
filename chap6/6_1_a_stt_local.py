import whisper
file = '../data/audio/enYann-tale_of_two_cities.mp3'
model = whisper.load_model("base")
result = model.transcribe(file) # fp16=False to use FP32 instead
print(result["text"])