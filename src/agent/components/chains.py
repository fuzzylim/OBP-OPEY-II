# Description: Contains the chains for the main agent system
from langchain import hub
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.components.tools import obp_requests

from pydantic import BaseModel, Field

### Main Opey agent
# Prompt
opey_system_prompt_template = """You are a friendly, helpful assistant for the Open Bank Project API called Opey. 
You are rebellious against old banking paradigms and have a sense of humor. But always give the user accurate and helpful information.
Using the context, which will be a combination of either/both swagger docs for certain relevant endpoints, and glossary entries for the user's query,
answer the user's query to the best of your ability. If you think you cant answer and you will instead hallucinate, respond with something along the lines of "I can't answer that at this time."
Context: {aggregated_context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(opey_system_prompt_template),
        MessagesPlaceholder("messages")
    ]
)

print(prompt.invoke({
    "messages": [HumanMessage(content="Who am I?")],
    "aggregated_context": ""
}))
#prompt = hub.pull("opey_main_agent")

# LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([obp_requests])

# Chain
opey_agent = prompt | llm 


### Retrieval decider

class RetrievalDeciderOutput(BaseModel):
    context_needed: bool = Field(description="Whether further context is needed for the user's query. True or False")
    retrieve_endpoints: bool = Field(description="Whether to retrieve information about endpoints from a vector store of endpoints. True or False")
    retrieve_glossary: bool = Field(description="Whether to retrieve information from a vector database of glossary entries. True or False")
    
retrieval_decider_system_prompt = """You are an assistant that decides whether further context is needed from a vector database search to answer a user's question.
Using the chat history, look at whether the user's question can be answered from the given context or not. If it can be answered already or if
the user's input does not require any context (i.e. they just said 'hello') return False. Otherwise return True
If context is needed, decide where to search for the context.
The endpoints vector store contains the swagger definitions of all the endpoints on the Open Bank Project API
The glossary vector store contains technical documents on many topics pertaining to Open Banking and the OBP API itself, such as how to authenticate.
Return True or False for the retrieve_endpoints or retrieve_glossary parameters.
"""
retrieval_decider_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=retrieval_decider_system_prompt),
        MessagesPlaceholder("messages"),
    ]
)

# LLM
retrieval_decider_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(RetrievalDeciderOutput)
# Chain
retrieval_decider_chain = retrieval_decider_prompt_template | retrieval_decider_llm


### Retrieval Query Formulator

class QueryFormulatorOutput(BaseModel):
    query: str = Field(description="Query to be used in vector database search of either glossary items or swagger specs for endpoints.")

query_formulator_system_prompt = """You are a query formulator that takes a list of messages and a mode: {retrieval_mode}
and tries to use the messages to come up with a short search query to search a vector database of either glossary items or partial swagger specs for API endpoints.
The query needs to be in the form of a natural sounding question that conveys the semantic intent of the message, especially the latest message from the human user.

If the mode is glossary_retrieval, optimise the query to search a glossary of technical documents for the Open Bank Project (OBP)
If the mode is endpoint_retrieval, optimise the query to search through swagger schemas of different endpoints on the Open Bank Project (OBP) API
"""

query_formulator_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=query_formulator_system_prompt),
        MessagesPlaceholder("messages"),
    ]
)

query_formulator_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(QueryFormulatorOutput)
query_formulator_chain = query_formulator_prompt_template | query_formulator_llm

messages = [HumanMessage(content="What endpoints can I use to find metrics on OBP API?")]

query_formulator_chain.invoke({"messages": messages, "retrieval_mode": "endpoint_retrieval"})