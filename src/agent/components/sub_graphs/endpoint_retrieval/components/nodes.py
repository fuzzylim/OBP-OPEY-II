import os

from agent.components.sub_graphs.endpoint_retrieval.components.states import OutputState
from agent.components.sub_graphs.retriever_config import setup_chroma_vector_store, setup_retriever
from agent.components.sub_graphs.endpoint_retrieval.components.chains import retrieval_grader, endpoint_question_rewriter
from dotenv import load_dotenv

load_dotenv()

# Setup vector store and retriever
retriever_batch_size = os.getenv("ENDPOINT_RETRIEVER_BATCH_SIZE", 5)

endpoint_vector_store = setup_chroma_vector_store("obp_endpoints")
endpoint_retriever = setup_retriever(k=8, vector_store=endpoint_vector_store)

def retrieve_endpoints(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE ENDPOINTS---")
    rewritten_question = state.get("rewritten_question", "")
    total_retries = state.get("total_retries", 0)
    
    if rewritten_question:
        question = state["rewritten_question"]
        total_retries += 1
    else:
        question = state["question"]
    # Retrieval
    documents = endpoint_retriever.invoke(question)
    return {"documents": documents, "total_retries": total_retries}


def return_documents(state) -> OutputState:
    """Return the relevant documents"""
    print("---RETRUN RELEVANT DOCUMENTS---")
    relevant_documents = state["relevant_documents"]
    return {"relevant_documents": relevant_documents}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    # web_search = False
    # glossary_search = False
    retry_query = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print(f"{d.metadata["method"]} - {d.metadata["path"]}", " [RELEVANT]")
            #print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print(f"{d.metadata["method"]} - {d.metadata["path"]}", " [NOT RELEVANT]")
            #print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
        
    # If there are three or less relevant endpoints then retry query after rewriting question
    retry_threshold = 2
    
    if len(filtered_docs) <= retry_threshold:
        retry_query=True
        
    #print("Documents: \n", "\n".join(f"{doc.metadata["method"]} - {doc.metadata["path"]}" for doc in filtered_docs))
    return {"documents": documents, "relevant_documents": filtered_docs, "question": question, "retry_query": retry_query}
              

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    total_retries = state.get("total_retries", 0)
    # Re-write question
    better_question = endpoint_question_rewriter.invoke({"question": question})
    print(f"New query: \n{better_question}\n")
    return {"documents": documents, "rewritten_question": better_question}
