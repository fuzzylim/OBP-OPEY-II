import json
import uuid

from pprint import pprint

from langchain_core.messages import ToolMessage

from agent.components.chains import opey_agent, query_formulator_chain
from agent.components.sub_graphs.endpoint_retrieval.endpoint_retrieval_graph import endpoint_retrieval_graph
from agent.components.sub_graphs.glossary_retrieval.glossary_retrieval_graph import glossary_retrieval_graph
from agent.components.states import OpeyGraphState
from agent.components.chains import retrieval_decider_chain

def run_retrieval_decider(state: OpeyGraphState):
    state["current_state"] = "retrieval_decider"
    messages = state["messages"]
    output = retrieval_decider_chain.invoke({"messages": messages})
    print(f"Retrieval decider: {output.tool_calls}")
    return {"messages": output}

    
def run_opey(state):
    messages = state["messages"]
    response = opey_agent.invoke({"messages": messages})
    print(response)
    return {"messages": response}

def human_review_node(state):
    state["current_state"] = "human_review"
    print("Awaiting human approval for tool call...")
    pass
