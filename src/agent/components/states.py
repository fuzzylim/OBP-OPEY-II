from langgraph.graph import MessagesState
from pprint import pprint

### States
class OpeyGraphState(MessagesState):
    conversation_summary: str
    current_state: str
    aggregated_context: str