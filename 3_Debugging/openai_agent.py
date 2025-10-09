from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("Langchain_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("Groq_API_Key")
class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

#model=ChatOpenAI(temperature=0)
from langchain_groq import ChatGroq

groq_llm=ChatGroq(model="qwen/qwen3-32b")

def make_default_graph():
    graph_workflow=StateGraph(State)

    def call_model(state):
        return {"message":[groq_llm.invoke(state['messages'])]}
    
    graph_workflow.add_node("agent",call_model)

    graph_workflow.add_edge(START,"agent")
    graph_workflow.add_edge("agent",END)

    agent=graph_workflow.compile()

    return agent

agent=make_default_graph()


