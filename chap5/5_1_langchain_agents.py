from langchain.agents import load_tools, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import tomli, os
with open("../.streamlit/secrets.toml","rb") as f:
    secrets = tomli.load(f)
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
os.environ["SERPAPI_API_KEY"] = secrets["SERPAPI_API_KEY"]
# Load the model
llm = ChatOpenAI(temperature=0)
# Load in some tools to use
tools = load_tools(["serpapi"])
# Finally, let's initialize an agent with:
# 1. The tools
# 2. The language model
# 3. The type of agent we want to use.
agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
# Now let's test it out!
res = agent("who is the ceo of twitter?")
print(res)