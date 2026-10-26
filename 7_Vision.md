# 7. Vision

## 7.1. From traditional computer vision to multi-modal language models

Computer vision is the field of computer science that deals with enabling machines to understand and process visual information, such as images and videos. Computer vision has many applications, such as autonomous driving, medical imaging, objects or humans detection and augmented reality. However, computer vision alone cannot capture the full meaning and context of visual information, especially when it comes to natural language tasks, such as captioning, summarizing, or answering questions about images. For example, a computer vision system may be able to identify objects and faces in an image, but it may not be able to explain their relationships, emotions, or intentions.

This is where “multi-modality” comes in. With regard to LLMs, modality refers to data types. Multimodal language models are systems that can process and generate both text and images and learn from their interactions. By combining computer vision and natural language processing, multimodal language models can achieve a deeper and richer understanding of visual information and produce more natural and coherent outputs. Multimodal language models can also leverage the large amount of available text and image data on the web and learn from their correlations and alignments.

In March 2023 during a GPT-4 developer demo livestream[^1], we got to witness the new vision skills. GPT-4 Turbo with Vision can process images and answer questions about them. Language model systems used to only take one type of input, text. But what if you could provide the model with the ability to “see”? This is now possible with GPT-4V(ision).

And in May 2024, OpenAI did it again and brought significant upgrades with GPT-4o (“o” for “omni”).

Let’s start with a basic example of the capabilities of GPT-4o. To pass local images along the request, you will need to encode them in base64. The alternative approach is to pass a url link to the image online.

``` python
import os, openai, requests, base64
def chat_vision(prompt, base64_image, model="gpt-4o", response_format="text", max_tokens=500):
    response = openai.chat.completions.create(
    model=model,
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ],
        }
    ],
    response_format={ "type": response_format },
    max_tokens=max_tokens,
    )
    return response.choices[0].message.content

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
from PIL import Image

image_path = "../img/dwight-internet.jpg"

base64_image = encode_image(image_path)
Image.open(image_path)
```

![Figure 7‑1 Image from the TV show The Office](media/image67.png)

*Figure 7‑1 Image from the TV show The Office*

``` python

response = chat_vision(prompt,base64_image)
response
```

```
'This image features two characters from a television show. They appear to be talking while walking outside past a parked car. The character on the left is wearing a mustard-colored shirt with a patterned tie and glasses, and the character on the right is wearing a dark suit with a blue tie. There is also a subtitle overlay that reads, "They\'re gonna be screwed once this whole internet fad is over." This subtitle suggests that the scene might be humorous or ironic, especially since the "internet fad" has proven to be a fundamental part of modern society.'
```

## 7.2. Object detection

Object detection[^2] is the task of identifying and locating objects of interest in an image or a video. Object detection can be useful for various applications, such as recognizing road signs to assist in driving cars. There are different AI approaches to object detection, such as:

\- *Template matching*: This approach involves comparing a template image of an object with the input image and finding the best match. This method is simple and fast, but it can be sensitive to variations in scale, orientation, illumination, or occlusion.

\- *Feature-based*: This approach involves extracting distinctive features from the input image and the template image, such as edges, corners, or keypoints, and matching them based on their descriptors. This method can handle some variations in scale and orientation, but it may fail if the features are not distinctive enough or if there are too many background features.

\- *Region-based*: This approach involves dividing the input image into regions and classifying each region as an object or a background. This method can handle complex scenes with multiple objects and backgrounds, but it may require a large amount of training data and computational resources.

\- *Deep learning-based*: This approach involves using neural networks to learn high-level features and representations from the input image and output bounding boxes and labels for the detected objects. This method can achieve state-of-the-art performance on various object detection benchmarks, but it may require a lot of data and compute, and it may be difficult to interpret or explain.

\- *LLM-based*: This approach involves using a pretrained language and vision model, such as GPT-4o, to input the image and the text as queries and generate an answer based on both. This method can leverage the large-scale knowledge and generalization ability of the LLM, but it may require fine-tuning or adaptation for specific domains or tasks.

### Application: detecting cars

Let’s take an example from a self-driving dataset from Udacity[^3]. This project starts with 223GB of open-source Driving Data. The repo[^4] has been archived, but you can still access it. Streamlit developed an associated app[^5] and hosted data on AWS[^6].

The AWS S3 bucket is publicly available, so we can access the content list as XML:

```
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
    <Name>streamlit-self-driving</Name>
    <MaxKeys>1000</MaxKeys>
    <IsTruncated>true</IsTruncated>
    <Contents>
        <Key>1478019952686311006.jpg</Key>
        <LastModified>2019-09-03T22:56:13.000Z</LastModified>
        <ETag>"17593334a87be9a26a6caa1080d32137"</ETag>
        <Size>27406</Size>
        <StorageClass>STANDARD</StorageClass>
    </Contents>
```

Let’s us the XML Path Language (XPath)[^7] to navigate the list:\
'.//{http://s3.amazonaws.com/doc/2006-03-01/}Key'

- . refers to the current node.

- // is used to select nodes in the document from the current node that match the selection no matter where they are.

- {http://s3.amazonaws.com/doc/2006-03-01/} is the namespace. XML namespaces are used for providing uniquely named elements and attributes in an XML document. They are defined with a URI. In this case, the URI is http://s3.amazonaws.com/doc/2006-03-01/.

- Key is the name of the element we are looking for.

``` python
import requests
import xml.etree.ElementTree as ET
bucket = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"
bucket_list = ET.fromstring(requests.get(bucket).content)
xpath = './/{http://s3.amazonaws.com/doc/2006-03-01/}Key'

keys = [content.text for content in bucket_list.findall(xpath)]
keys[48]
```

```
'1478019976687231684.jpg'
```

``` python
from PIL import Image
import io
image = Image.open(io.BytesIO(requests.get(bucket+keys[48]).content))

image_path = "../img/"+keys[48]
image.save(image_path)

image
```

![Figure 7‑2 Image from the Udacity self driving car dataset](media/image68.png)

*Figure 7‑2 Image from the Udacity self driving car dataset*

### LLM-based object detection

``` python
import base64
base64_image = base64.b64encode(requests.get(bucket+keys[48]).content).decode('utf-8')

chat_vision(prompt,base64_image,model="gpt-4-vision-preview")
```

```
'There is a car visible on the left side of the image, moving away from the viewpoint and through the intersection. Another car is visible across the street, making a left turn.'
```

``` python

chat_vision(prompt,base64_image,model="gpt-4-vision-preview")
```

```
"I'm sorry, I can't assist with that request."
```

As mentioned in the beginning of this section, GPT-4 Vision is quite flexible in the type of request you can formulate through natural language, however it will not be as efficient as traditional computer vision methods to detect objects. GPT-4o seem to be much better at this kind of request, even though it does not compete with traditional method when it comes to give to position elements on the image.

``` python

jason = chat_vision(prompt,base64_image,model="gpt-4o", response_format="json_object")
print(jason)
```

```
{'cars' = [
    {'x_min': 225, 'y_min': 145, 'x_max': 265, 'y_max': 175}
]}
```

``` python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

image_path = "../img/1478019976687231684.jpg"
image = plt.imread(image_path)

fig, ax = plt.subplots()

ax.imshow(image)

car_coordinates = [
    {'x_min': 225, 'y_min': 145, 'x_max': 265, 'y_max': 175}
]

for coord in car_coordinates:
    x_min = coord["x_min"]
    y_min = coord["y_min"]
    width = coord["x_max"] - coord["x_min"]
    height = coord["y_max"] - coord["y_min"]
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()
```

![Figure 7‑3 Bounding box attempting to catch a car](media/image69.png)

*Figure 7‑3 Bounding box attempting to catch a car*

As you can see, this clearly isn’t a success yet for object detection and localization.

### Traditional Computer Vision

*YOLO* (You Look Only Once)[^8] is a state-of-the-art, real-time object detection system. It is fast and accurate. You can retrieve the config file for YOLO in the cfg/ subdirectory of the repo[^9]. You will have to download the pre-trained weights file[^10] (237 MB).

``` python


import cv2
import numpy as np

net = cv2.dnn.readNet("../yolo/yolov3.weights", "../yolo/yolov3.cfg")

with open("../yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

image = cv2.imread(image_path)
height, width, _ = image.shape

blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

net.setInput(blob)

outs = net.forward(net.getUnconnectedOutLayersNames())

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and class_id == 2:  # Class ID for car
            # Get bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
Image.fromarray(image_rgb)
```

![Figure 7‑4 Better bounding boxes for cars](media/image70.png)

*Figure 7‑4 Better bounding boxes for cars*

Another project to add to my TODO list is setting up my Raspberry Pi with a camera pointed at a parking spot to notify me when the spot is available.

### Optical Character Recognition

Optical character recognition (OCR) is a computer vision task that involves extracting text from images, such as scanned documents, receipts, or signs. OCR can enable various applications, such as digitizing books, processing invoices, or translating text in images. However, traditional OCR methods have some limitations, such as:

- They may not handle noisy, distorted, or handwritten text well, especially if the text is in different fonts, sizes, or orientations.

- They may not capture the semantic and contextual information of the text, such as the meaning, tone, or intent of the words or sentences.

- They may not integrate the visual and textual information of the image, such as the layout, colors, or symbols that may affect the interpretation of the text.

Let’s try out GPT-4o with an easy example of code copied with a simple printscreen (win+shift+s). You can also use the Windows snipping tool to grab images on your screen:

``` python

from PIL import ImageGrab

img
```

![](media/image71.png)

``` python

import io
import base64
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
text = chat_vision("Extract the text from the image, return only the text.",encode_image_grab(img))
print(text)
```

````
```python

from PIL import ImageGrab

img
```
````

``` python
code = chat_vision("Extract the code from the image, return only the code without markdown formating.",encode_image_grab(img))
print(code)
```

```

from PIL import ImageGrab

img
```

Let’s implement a simple Streamlit app that will provide us with an OCR assistant, with a paste button[^11] getting images from the clipboard:

```
```

This is what the ocr_assistant.py looks like:

![Figure 7‑5 Optical Character Recognition app](media/image72.png)

*Figure 7‑5 Optical Character Recognition app*

## 7.3. From mock to web UI

One really exciting application of GPT-4o is to generate code from a hand-drawn mock-up of a webpage. This was part of the GPT-4 unveiling demo. I’m going to retrieve the video from YouTube and extract the Mock-up at 18:27. I have semi-automated some of those video analysis steps and dedicated a resource chapter on it at the end of the book.

``` python
from pytube import YouTube
def youtube_download(video_id, quality="lowest",path="../data/video"):
    # Define the URL of the YouTube video
    url = f'https://www.youtube.com/watch?v={video_id}'  
    # Create a YouTube object
    yt = YouTube(url)
    if quality == "highest":
        # Download the video in the highest quality 1080p
        # (does not necessarily come with audio)
        video_path = yt.streams.get_highest_resolution().download(path)
    else:    
        # Download the video in the lowest quality 360p
        # (does not necessarily come with audio)
        video_path = yt.streams.get_lowest_resolution().download(path)
    return video_path
video_id = "outcGtbnMuQ"
video_path = youtube_download(video_id, quality="highest",path="../data/video")

import cv2
def extract_video_frame(video_path, time):
    # Load video
    cap = cv2.VideoCapture(video_path)
    # Get the frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Get the total number of frames
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # Set the frame to extract
    minutes, seconds = map(int, time.split(':'))
    total_seconds = minutes * 60 + seconds
    frame_to_extract = total_seconds * fps
    # Extract the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_extract)
    ret, frame = cap.read()
    # Save the frame
    frame_path = "../img/frame.jpg"
    cv2.imwrite(frame_path, frame)
    # Release the video capture
    cap.release()
    return frame_path
frame_path = extract_video_frame(video_path, time="18:27")

Image.open(frame_path)
```

![Figure 7‑6 Picture of a notebook displaying the mock of a basic website](media/image73.png)

*Figure 7‑6 Picture of a notebook displaying the mock of a basic website*

Let’s open the frame in Paint.net, and grab the area to crop:

![](media/image74.png)

We can leverage our OCR assistant to write the python code to extract the selection.

![Figure 7‑7 OCR assistant extracting python code](media/image75.png)

*Figure 7‑7 OCR assistant extracting python code*

``` python
from PIL import Image

original_image = Image.open(frame_path)

selection = (449, 95, 383, 552)

x1, y1, width, height = selection
x2 = x1 + width
y2 = y1 + height

cropped_image = original_image.crop((x1, y1, x2, y2))

cropped_image_path = '../img/cropped_image.jpg'
cropped_image.save(cropped_image_path)
cropped_image
```

![Figure 7‑8 Cropped view of the notebook](media/image76.png)

*Figure 7‑8 Cropped view of the notebook*

Let’s reuse the prompt from the demo:

``` python
original_prompt = "Write brief HTML/JS to turn this mock-up into a colorful website, where the jokes are replaced by two real jokes."

base64_image = encode_image(cropped_image_path)
code = chat_vision(prompt,base64_image)
with open("joke_website.html","w") as f:
    f.write(code)
```

And Voila! We have our website coded for us (the resulted html is served up as a GitHub page):

<https://yanndebray.github.io/programming-GPTs/chap7/joke_website>

![Figure 7‑9 Generated website based on the mock](media/image77.png)

*Figure 7‑9 Generated website based on the mock*

## 7.4. Video understanding

Let’s look at another use case: Processing and narrating a video with GPT's visual capabilities and the TTS API[^12]. For this example, I will retrieve a video from Youtube that I created, and add a different voiceover.

``` python
from IPython.display import display, Image, Audio
from pytube import YouTube
import cv2, base64, time, openai, os, requests
video_id = "lBaEO836ECY"
video_path = youtube_download(video_id, quality="lowest",path="../data/video")
video = cv2.VideoCapture(video_path)
base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
video.release()
print(len(base64Frames), "frames read.")
```

```
2134 frames read.
```

``` python
display_handle = display(None, display_id=True)
for img in base64Frames:
    display_handle.update(Image(data=base64.b64decode(img.encode("utf-8"))))
    time.sleep(0.025)
```

![Figure 7‑10 Last frame from the video displayed at the end of the loop](media/image78.jpeg)

*Figure 7‑10 Last frame from the video displayed at the end of the loop*

``` python
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames of a movie trailer. Create a short voiceover script in the style of Wes Anderson. Only include the narration.",
            *map(lambda x: {"image": x, "resize": 512}, base64Frames[0::60]),
        ],
    },
]
params = {
    "model": "gpt-4-o",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 500,
}
result = openai.chat.completions.create(**params)
script = result.choices[0].message.content
print(script)
```

```
"In a world where equations come to life... there exists an extraordinary young man, Max.
Max, played by Timothée Chalamet, is no ordinary scientist. His experiments defy the ordinary, boldly wander into the strange, and occasionally, become magical.
With analog computers and whimsical algorithms, Max teams up with the peculiar yet brilliant Professor Anderson, portrayed by Bill Murray.
Their journey through the enigmatic realms of MATLAB and Simulink unfolds in a symphony of colorful chaos, eclectic technologies, and serendipitous discoveries.
A journey where each computation sparks a revelation, every calculation a wonder.
From the visionary mind of Wes Anderson comes 'MATLAB and Simulink: The Symphony of Imagination'— pulling the strings of reason, and plucking the notes of creativity."
```

``` python
response = requests.post(
    "https://api.openai.com/v1/audio/speech",
    headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    },
    json={
        "model": "tts-1-1106",
        "input": script,
        "voice": "onyx",
    },
)
audio = b""
for chunk in response.iter_content(chunk_size=1024 * 1024):
    audio += chunk
Audio(audio)
```

![](media/image79.png)

(Don’t try to click play on a book, it’s not going to work… go online instead)

``` python
with open("../data/audio/voiceover.mp3", "wb") as f:
    f.write(audio)
```

Let’s use MusicGen[^13] from Meta to add some music to the background. The initial video is lasting 71 sec, let’s speed it up by a factor of 1.4, and save the result with voiceover.

``` python
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.video.fx.speedx import speedx
video = VideoFileClip(video_path)

speed_up_factor = 1.4
video = speedx(video, speed_up_factor)
audio = AudioFileClip("../data/audio/voiceover.mp3")
final_video = video.set_audio(audio)

final_video.write_videofile('../data/video/symphony_voiced_over.mp4', codec="libx264")
```

```
Moviepy - Done !
Moviepy - video ready ../data/video/symphony_voiced_over.mp4
```

[^1]:  GPT-4 Developer Livestream <https://www.youtube.com/watch?v=outcGtbnMuQ>

[^2]:  <https://www.mathworks.com/discovery/object-detection.html>

[^3]:  <https://medium.com/udacity/open-sourcing-223gb-of-mountain-view-driving-data-f6b5593fbfa5>

[^4]:  Udacity repo: <https://github.com/udacity/self-driving-car>

[^5]:  Streamlit app: <https://github.com/streamlit/demo-self-driving/blob/master/streamlit_app.py>

[^6]:  AWS S3 bucket: <https://streamlit-self-driving.s3-us-west-2.amazonaws.com/>

[^7]:  <https://en.wikipedia.org/wiki/XPath>

[^8]:  YOLO website: <https://pjreddie.com/darknet/yolo/>

[^9]:  YOLO repo: <https://github.com/pjreddie/darknet>

[^10]:  YOLO weights file: <https://pjreddie.com/media/files/yolov3.weights>

[^11]:  <https://github.com/olucaslopes/streamlit-paste-button>

[^12]:  <https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding>

[^13]:  [audiocraft.metademolab.com/musicgen.html](file:///C:\Users\ydebray\Downloads\programming-GPTs\audiocraft.metademolab.com\musicgen.html) - [huggingface.co/facebook/musicgen-large](https://huggingface.co/facebook/musicgen-large)

---

[⬅️ Speech-to-Text and Text-to-Speech](6_Speech_to_Text_and_Text_to_Speech.md) | [🏠 Home](README.md) | [DALL-E image generation ➡️](8_DALL_E_image_generation.md)
