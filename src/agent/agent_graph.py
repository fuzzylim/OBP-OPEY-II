from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

from agent.components.states import OpeyGraphState
from agent.components.nodes import run_endpoint_retrieval, run_glossary_retrieval, run_opey
from agent.components.conditional_edges import run_retrieval_decider
from agent.components.tools import obp_requests


memory = MemorySaver()

opey_workflow = StateGraph(OpeyGraphState)

#opey_workflow.add_node("decide_to_retrieve", run_retrieval_decider)
opey_workflow.add_node("retrieve_endpoints", run_endpoint_retrieval)
opey_workflow.add_node("retrieve_glossary", run_glossary_retrieval)
opey_workflow.add_node("opey", run_opey)
opey_workflow.add_node("obp_requests", ToolNode([obp_requests]))

opey_workflow.add_conditional_edges(
    START,
    run_retrieval_decider,
    ["retrieve_endpoints", "retrieve_glossary", "opey"]
)

opey_workflow.add_conditional_edges(
    "opey",
    tools_condition,
    {
        "tools": "obp_requests",
        "__end__": END
    }
)

opey_workflow.add_edge("obp_requests", "opey")
opey_workflow.add_edge("retrieve_endpoints", "opey")
opey_workflow.add_edge("retrieve_glossary", "opey")
#opey_workflow.add_edge("opey", END)

opey_graph = opey_workflow.compile(checkpointer=memory)