import json
import requests

from langchain_core.tools import tool

from agent.utils.config import obp_base_url, get_headers


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
