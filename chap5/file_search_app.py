import streamlit as st
from openai import OpenAI
import time
import datetime

st.sidebar.title('File search 🔎')
avatar = {"assistant": "🤖", "user": "🐱"}
tools = {"code_interpreter": "🐍", "file_search": "🔎"}
client = OpenAI(
    api_key=st.secrets['OPENAI_API_KEY'],
)

def retrieve_citations(message_content):
    # message_content = messages.data[0].content[0].text
    annotations = message_content.annotations
    citations = []
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")
    return message_content,citations

def store_run_steps(thread_id,run_id):
    run_steps = client.beta.threads.runs.steps.list(
        thread_id=thread_id,
        run_id=run_id
    )
    # append new line to file
    with open(f'runs/{run_id}.txt', 'a') as f:
        f.write(f'{run_steps.data}\n')

if 'thread' not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state.thread = thread
thread = st.session_state.thread
if 'messages' not in st.session_state:
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    st.session_state.messages = messages
messages = st.session_state.messages
# if 'run' not in st.session_state:
#     st.session_state.run = []
    
assistant = client.beta.assistants.retrieve("asst_lKH0wC1rCQN1ZnyzGinbYVs3")
# st.sidebar.write(assistant)
assistant_tools = [tool.type for tool in assistant.tools]
assistant_tools_emojis = [tools[tool.type] for tool in assistant.tools]

# st.sidebar.write(f'# {assistant.name}')
st.sidebar.write(f'*({assistant.id})*')
st.sidebar.write(f'**Instructions**:\n{assistant.instructions}')
st.sidebar.write(f'**Model**:\n{assistant.model}')
# st.sidebar.write(f'**Tools**:\n{list(zip(assistant_tools, assistant_tools_emojis))}')

## 1 - Create Thread
# if st.sidebar.button('Create Thread'):
#     thread = client.beta.threads.create()
#     st.session_state.thread = thread
#     st.sidebar.write(thread.id)

st.sidebar.write('## Thread')
# st.sidebar.write(thread)
st.sidebar.write(f'*{thread.id}*')
st.sidebar.write(f'Created at {datetime.datetime.fromtimestamp(thread.created_at)}')

## 2 - Add a message
if prompt := st.chat_input():
    # with st.sidebar.status('Processing...', expanded=True) as status:
    with st.spinner('Wait for it...'):
        
        messages = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt,
        )
        st.session_state.messages = messages
        # st.write(messages)
        st.toast('Adding message...')

## 3 - Run the thread
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            # instructions="Please address the user as Jane Doe. The user has a premium account."
        )
        # st.session_state.run = run
        st.toast(run.status)

## 4 - Get run's status and steps
        while run.status != 'completed':
            run_steps = client.beta.threads.runs.steps.list(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_steps.data:
                if run_steps.data[0].step_details.type == 'tool_calls':
                    st.toast(run_steps.data[0].step_details.tool_calls[0].type)
                elif run_steps.data[0].step_details.type == 'message_creation':
                    st.toast(run_steps.data[0].step_details.type)
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            store_run_steps(thread.id,run.id)
            st.toast(run.status)

## 5 - Get messages
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        st.session_state.messages = messages
        # st.write(messages.data)
        # status.update(label="Processed", state="complete", expanded=False)
    st.toast('Dairy🥛!')

    for line in messages.data[::-1]:
        # st.chat_message(line.role,avatar=avatar[line.role]).write(line.content[0].text.value)
        if line.role == 'user':
            st.chat_message('user',avatar=avatar['user']).write(line.content[0].text.value)
        elif line.role == 'assistant':
            message_content,citations = retrieve_citations(line.content[0].text)
            with st.chat_message('assistant',avatar=avatar['assistant']):
                st.write(message_content.value)
                if citations:
                    st.write('*Citations:*')
                    for citation in citations:
                        st.write(citation)

            

## X - More

# st.sidebar.expander('Messages').write(st.session_state.messages)
# st.sidebar.expander('Run').write(st.session_state.run)

st.sidebar.write('## Prompt examples')
st.sidebar.write("What are the vector databases mentioned in the book?")
st.sidebar.write("What is said about ChromaDB?")
st.sidebar.write("What about FAISS?")