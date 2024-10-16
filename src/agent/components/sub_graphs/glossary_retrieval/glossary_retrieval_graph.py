from langgraph.graph import StateGraph, START, END
from agent.components.sub_graphs.endpoint_retrieval.components.states import SelfRAGGraphState
from agent.components.sub_graphs.glossary_retrieval.components.nodes import retrieve_glossary, grade_documents_glossary

# Glossary retrieval graph definition
# NOTE: Some components are shared with the endpoint retrieval graph

glossary_retrieval_workflow = StateGraph(SelfRAGGraphState)

glossary_retrieval_workflow.add_node("retrieve_items", retrieve_glossary)
glossary_retrieval_workflow.add_node("grade_documents", grade_documents_glossary)

glossary_retrieval_workflow.add_edge(START, "retrieve_items")
glossary_retrieval_workflow.add_edge("retrieve_items", "grade_documents")
glossary_retrieval_workflow.add_edge("grade_documents", END)

glossary_retrieval_graph = glossary_retrieval_workflow.compile()