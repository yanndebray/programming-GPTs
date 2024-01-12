# 🥳🎂🎉 One year of Programming GPTs 🤖

It has been a year since [ChatGPT got released](https://openai.com/blog/chatgpt) (November 30th, 2022). What a year! Some were skeptical at first. As Dwight Schrute said himself: “the internet is a fad”. But with 100 million weekly active users a year later, I’m confident stating that this is the beginning of a new internet.

![](img/dwight-internet.jpg)

For the better part of 2023, my hobby has been programming and experimenting with it. What I mean by programming GPTs does not mean that I am recreating the AI behind GPT 3.5 or 4 from scratch. I have tried to fine tune an open-source pretrained AI model like LLaMa2 (from Meta), or start from scratch with much smaller models, but the results you get with such an approach are not as good as what OpenAI provides. And it takes way more skill to reinvent the service that OpenAI offers at a price that is very competitive. So instead, I am focusing here on building on top of the OpenAI giants.

In this blog, you will learn how to program GPTs primarily leveraging OpenAI’s APIs. If you don’t know what an API is (Application Programming Interface), then this is not the blog you’re looking for. Go look it up on the internet, and come back after learning some basics of programming, preferably in Python, as this will be the language used for the tutorials in each chapter.
Generative Pre-Trained Transformers are quite complicated general-purpose machines that can do a lot of different things, like handling sequences of text, images, and sounds. You will learn the basic concepts and principles behind GPTs, and how they work under the hood. But more importantly, you will learn how to integrate them inside of your applications.

The blog is divided into 10 chapters, each covering a different topic and a different aspect of programming GPTs. The chapters are:
- **Chapter 1: 🤖 Introduction to GPTs**. How they work and their evolution. 

- [**Chapter 2: 🐱💬 The Chat API.**](chap2/README.md) In this chapter, you will learn how to use the Chat API, a simple way to create conversational agents with GPTs. You will learn how to create your own chatbot.

    [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chat48.streamlit.app/)

<video src="https://github.com/slevin48/openai/assets/12418115/91b2c4c6-7338-4f51-9440-16e59626378d" controls="controls" style="max-width: 730px;"></video>

- **Chapter 3: 🔗 Chaining & Summarization.** In this chapter, you will learn how to chain calls to a Large Language Model and use it to summarize texts, such as articles, books, or transcripts. You will learn how to use the Chat API together with the LangChain package to enhance GPTs.

    [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://teams-summarizer.streamlit.app/)

<video src="https://user-images.githubusercontent.com/12418115/229315555-c9d2077a-a2ed-4538-816c-ce0b9cece761.webm" controls="controls" style="max-width: 730px;"></video>

- **Chapter 4: 🔎❓ Vector search & Question Answering.** In this chapter, you will learn how to use embeddings and vector search as a way to retrieve informative answers to answer questions while quoting sources. 

    [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ask-impromptu.streamlit.app/)

<video src="https://github.com/slevin48/openai/assets/12418115/b37ec63c-1a7d-4c81-a4b0-5f10f7ceb428" controls="controls" style="max-width: 730px;"></video>

- **Chapter 5: 🕵️‍♀️🛠️ Agent & Tools.** In this chapter, you will learn to build an Agent, called Smith, that has access to tools, such as getting the current weather.

    [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ask-smith.streamlit.app/)

<video src="https://github.com/slevin48/openai/assets/12418115/f8e0a3e6-50c9-4eb9-961f-177000a2d7f4" controls="controls" style="max-width: 730px;"></video>

- **Chapter 6: 🧩 Assistants & Plugins.** In this chapter, you will learn how to use the Assistant API provided by OpenAI, and to extend their capabilities with Plugins to integrate GPTs with external services. You will also learn how to create your own Code Interpreter, that can help you write and run code with GPTs. 

    [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chat-with-python.streamlit.app/)

<video src="https://github.com/slevin48/openai/assets/12418115/624d549a-f819-4929-bba5-a58d8589b396" controls="controls" style="max-width: 730px;"></video>


- **Chapter 7: 🗣️📢 Text to Speech & Synthesis.** In this chapter, you will learn how to use GPTs to transcript text from speech (such as Youtube videos), and synthetize speech from text (such as articles). 

    [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://assistant48.streamlit.app/)

<video src="https://github.com/slevin48/openai/assets/12418115/6f31226b-50d7-436e-b0ec-7132a6ddfd7e" controls="controls" style="max-width: 730px;"></video>


- **Chapter 8: 👀 Vision API.** In this chapter, you will learn how to use GPTs to process and analyze images, such as mock-ups or drawings. You will learn how to use the Vision API, to perform various tasks with GPTs, such as text recognition, or video captioning.

    [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chat-vision.streamlit.app/)

<video src="https://github.com/slevin48/openai/assets/12418115/1a83c37c-b105-42cf-b2d7-b25656824e13" controls="controls" style="max-width: 730px;"></video>

- **Chapter 9: 🎨🖌️ Dall-E image generation.** In this chapter, you will learn how to use Dall-E 2 & 3, which can create stunning and creative images from any text input. You will also learn how to use the outpainting, inpainting and variations APIs, which can complete or modify existing images.

    [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://openai-image.streamlit.app/)

<video src="https://github.com/slevin48/openai/assets/12418115/3726aa2c-67e0-474c-909f-5cb54eaa39db" controls="controls" style="max-width: 730px;"></video>

- **Chapter 10: 📌 Conclusion.** In this chapter, you will review what you have learned, and reflect on the potential and challenges of programming GPTs. You will also learn how to keep up with the latest developments and innovations in the field of GPTs and OpenAI with additional resources.
