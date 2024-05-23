from openai import OpenAI
client = OpenAI()
def create_assistant():
    assistant = client.beta.assistants.create(
    name="Data Analyst",
    instructions="You are a data analyst. When asked a question, write and run code to answer the question.",
    model="gpt-4-turbo",
    tools=[{"type": "code_interpreter"}]
    )
    return assistant

def load_assistant(id):
    assistant = client.beta.assistants.retrieve(id)
    return assistant

def upload_file(file):
    # Upload a file with an "assistants" purpose
    file = client.files.create(
    file = open(file, "rb"),
    purpose='assistants'
    )
    return file.id

def create_thread():
    thread = client.beta.threads.create()
    return thread.id

def add_message_thread(prompt,thread_id,file_id = None):
    message = client.beta.threads.messages.create(
    thread_id = thread_id,
    role="user",
    content=prompt,
    attachments=[
        {
        "file_id": file_id,
        "tools": [{"type": "code_interpreter"}]
        }
    ]
    )
    return message.id

def create_run(thread_id,assistant_id):
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id= assistant_id
    )
    return run.id

def store_run_steps(thread_id,run_id):
    run_steps = client.beta.threads.runs.steps.list(
        thread_id=thread_id,
        run_id=run_id
    )
    # append new line to file
    with open(f'runs/{run_id}.txt', 'a') as f:
        f.write(f'{run_steps.data}\n')

def retrieve_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    return messages.data[0].content[0].text.value

def retrieve_code(thread_id,run_id):
    # Inspect the chain of thoughts
    run_steps = client.beta.threads.runs.steps.list(
        thread_id=thread_id,
        run_id=run_id
    )
    # Extract only the code interpreter input
    for d in run_steps.data[::-1]:
        if d.step_details.type == 'tool_calls':
            print(d.step_details.tool_calls[0].code_interpreter.input)
    code = [d.step_details.tool_calls[0].code_interpreter.input for d in run_steps.data[::-1] if d.step_details.type == 'tool_calls']
    return code