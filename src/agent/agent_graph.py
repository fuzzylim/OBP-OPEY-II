from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

from agent.components.states import OpeyGraphState
from agent.components.nodes import run_opey, run_retrieval_decider, human_review_node, run_summary_chain
from agent.components.edges import should_summarize
from agent.components.tools import obp_requests, glossary_retrieval_tool, endpoint_retrieval_tool


memory = MemorySaver()

opey_workflow = StateGraph(OpeyGraphState)

# Define tool nodes
# Define retrieval tools
retrieval_tools = [glossary_retrieval_tool, endpoint_retrieval_tool]
retrieval_tool_node = ToolNode(retrieval_tools)

# Define requests tools
obp_requests_tool_node = ToolNode([obp_requests])

# Add Nodes to graph
opey_workflow.add_node("retrieval_decider", run_retrieval_decider)
opey_workflow.add_node("retrieval_tools", retrieval_tool_node)
opey_workflow.add_node("opey", run_opey)
opey_workflow.add_node("human_review", human_review_node)
opey_workflow.add_node("obp_requests_tools", obp_requests_tool_node)
opey_workflow.add_node("summarize_conversation", run_summary_chain)

opey_workflow.add_conditional_edges(
    "retrieval_decider",
    tools_condition,
    {
        "tools": "retrieval_tools",
        "__end__": "opey"
    }
)
opey_workflow.add_conditional_edges(
    "opey",
    tools_condition,
    {
        "tools": "human_review",
        "__end__": END
    }
)

opey_workflow.add_conditional_edges(
    "opey",
    should_summarize,
    {
        "summarize_conversation": "summarize_conversation",
        END: END
    }

)

opey_workflow.add_edge("human_review", "obp_requests_tools")
opey_workflow.add_edge(START, "retrieval_decider")
opey_workflow.add_edge("obp_requests_tools", "opey")
opey_workflow.add_edge("retrieval_tools", "opey")
opey_workflow.add_edge("summarize_conversation", END)

opey_graph = opey_workflow.compile(checkpointer=memory, interrupt_before=["human_review"])