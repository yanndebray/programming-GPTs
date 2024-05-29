import streamlit as st
from openai import OpenAI
import time
import datetime
from assistant import *

st.sidebar.title('Code Interpreter 🐍')
avatar = {"assistant": "🤖", "user": "🐱"}
tools = {"code_interpreter": "🐍", "file_search": "🔎"}
client = OpenAI(
    api_key=st.secrets['OPENAI_API_KEY'],
)

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
if 'code' not in st.session_state:
    st.session_state.code = ''
    
assistant = load_assistant("asst_5zjj3Cp5W2DOT6sRLeT6Cf23")
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
        # st.chat_message('user',avatar=avatar['user']).write(messages.content[0].text.value)
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
        previous_step = None
        st.session_state.code = ''
        while run.status != 'completed':
            run_steps = client.beta.threads.runs.steps.list(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_steps.data:
                # check if the step is a tool call
                if run_steps.data[0].step_details.type == 'tool_calls':
                # check if the code interpreter is selected
                    if run_steps.data[0].step_details.tool_calls != []:
                    # check if the step is different
                        if run_steps.data[0].id != previous_step:
                            previous_step = run_steps.data[0].id
                            # st.toast(run_steps.data[0])
                            st.toast(f'code\n```python\n{run_steps.data[0].step_details.tool_calls[0].code_interpreter.input}\n```')
                            st.session_state.code = run_steps.data[0].step_details.tool_calls[0].code_interpreter.input
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            # store_run_steps(thread.id,run.id)
            st.toast(run.status)

## 5 - Get messages
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        st.session_state.messages = messages
        # st.sidebar.write(messages.data) #[-1]
        # status.update(label="Processed", state="complete", expanded=False)
    st.toast('Dairy🥛!')

    for line in messages.data[::-1]:
        # st.chat_message(line.role,avatar=avatar[line.role]).write(line.content[0].text.value)
        if line.role == 'user':
            st.chat_message('user',avatar=avatar['user']).write(line.content[0].text.value)
        elif line.role == 'assistant':
            with st.chat_message('assistant',avatar=avatar['assistant']):
                for c in line.content:
                    if c.type == 'text':
                        st.write(c.text.value)
                    elif c.type == 'image_file':
                        image_data = client.files.content(c.image_file.file_id)
                        image_data_bytes = image_data.read()
                        st.image(image_data_bytes)
    if st.session_state.code:
        st.expander('code').write('```python\n'+st.session_state.code+'\n```')

## X - More

# st.sidebar.expander('Messages').write(st.session_state.messages)
# st.sidebar.expander('Run').write(st.session_state.run)

st.sidebar.write('## Prompt examples')
examples = [
    "1+1",
    "How to solve the equation `3x + 11 = 14`?",
    "What is the 42nd element of Fibonacci?",
    "What is the 10th element?",
    "plot function 1/sin(x)",
    "zoom in to range of x values between 0 and 1",
    "plot a tangent line to the graph at x=0.3",
    "zoom in to the point of tangency"
]
for ex in examples:
    st.sidebar.write(ex)