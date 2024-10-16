import json

from pprint import pprint

from agent.components.chains import opey_agent, query_formulator_chain
from agent.components.sub_graphs.endpoint_retrieval.endpoint_retrieval_graph import endpoint_retrieval_graph
from agent.components.sub_graphs.glossary_retrieval.glossary_retrieval_graph import glossary_retrieval_graph

def run_endpoint_retrieval(state):
    """
    Run vector db retrieval workflow for endpoints.
    Args:
        state: The current state.
    Returns:
        The state with the updated aggregated context.
    """
    state["current_state"] = "retrieve_endpoints"
    messages = state["messages"]
    aggregated_context = state.get("aggregated_context", "")
    
    print("Running the retrieval workflow...")
    # Come up with a query from the messages
    output = query_formulator_chain.invoke({"messages": messages, "retrieval_mode": "endpoint_retrieval"})
    inputs = {"question": output.query}
    for output in endpoint_retrieval_graph.stream(inputs):
        for _, _ in output.items():
            pass 
        pprint("--------------------")
    if not aggregated_context:
        state["aggregated_context"] = ""
        
    relevant_context = ""
    for doc in output['relevant_documents']:
        context_string = f"Swagger schema for the endpoint {doc.metadata["method"]} {doc.metadata["path"]}:\n"
        context_string += json.dumps(doc.page_content, indent=2) + "-------------------\n\n"
        relevant_context += context_string

    state["aggregated_context"] += relevant_context
    return state

def run_glossary_retrieval(state):
    """
    Run vector db retrieval workflow for glossary.
    Args:
        state: The current state.
    Returns:
        The state with the updated aggregated context.
    """
    state["current_state"] = "retrieve_glossary"
    messages = state["messages"]
    aggregated_context = state.get("aggregated_context", "")
    
    print("Running the retrieval workflow...")
    # Come up with a query from the messages
    output = query_formulator_chain.invoke({"messages": messages, "retrieval_mode": "glossary_retrieval"})
    inputs = {"question": output.query}
    for output in glossary_retrieval_graph.stream(inputs):
        for _, _ in output.items():
            pass 
        pprint("--------------------")
    if not aggregated_context:
        state["aggregated_context"] = ""
        
    relevant_context = ""
    for doc in output['relevant_documents']:
        context_string = f"Glossary entry for {doc.metadata["title"]}:\n"
        context_string += doc.page_content + "-------------------\n\n"
        relevant_context += context_string

    state["aggregated_context"] += relevant_context
    return state
    
def run_opey(state):
    messages = state["messages"]
    aggregated_context = state.get("aggregated_context", "")
    response = opey_agent.invoke({"messages": messages, "aggregated_context": aggregated_context})
    print(response)
    return {"messages": response}