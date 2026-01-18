import os 
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
os.environ["GEMINI_API_KEY"]=os.getenv("GEMINI_API_KEY")


from tavily import TavilyClient
from typing import Literal

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_query(query:str,max_results:int=3,
topic: Literal["general","news","medical","finance","technology"]="general"):
    """Search the web for information on a given topic."""
    response = tavily.search(query=query,max_results=max_results,include_raw_content=True,topic=topic)
    return response

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("groq:qwen/qwen3-32b")

agent = create_deep_agent(
    model=model,
    tools=[web_query],
    system_prompt="Act as a researcher and provide detailed information on the topic.",
    debug=True
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [("user", "search for world record of 100m sprint")]})
    print(result["messages"][-1].content)