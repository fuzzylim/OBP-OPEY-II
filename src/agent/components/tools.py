import json
import requests

from langchain_core.tools import tool

from agent.utils.config import obp_base_url, get_headers
from agent.components.sub_graphs.endpoint_retrieval.endpoint_retrieval_graph import endpoint_retrieval_graph
from agent.components.sub_graphs.glossary_retrieval.glossary_retrieval_graph import glossary_retrieval_graph

@tool
def obp_requests(method: str, path: str, body: str):
    # TODO: Add more descriptive docstring, I think this is required for the llm to know when to call this tool
    """Executes a request to the OpenBankProject (OBP) API"""
    url = f"{obp_base_url}{path}"
    headers = get_headers()
    
    if body == '':
        json_body = None
    else:
        json_body = json.loads(body)
        
    if method == "GET":
        response = requests.get(url, headers=headers)
        return response.json()
    else:
        return {"error": f"dangerous method {method} not allowed"}

# Define endpoint retrieval tool nodes

endpoint_retrieval_tool = endpoint_retrieval_graph.as_tool(name="retrieve_endpoints")
glossary_retrieval_tool = glossary_retrieval_graph.as_tool(name="retrieve_glossary")