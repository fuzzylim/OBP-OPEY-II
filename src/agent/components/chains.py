# Description: Contains the chains for the main agent system
from langchain import hub
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.utils.model_factory import get_llm
from agent.components.tools import obp_requests, glossary_retrieval_tool, endpoint_retrieval_tool

from pydantic import BaseModel, Field

### Main Opey agent
# Prompt
opey_system_prompt_template = """You are a friendly, helpful assistant for the Open Bank Project API called Opey. 
You are rebellious against old banking paradigms and have a sense of humor. But always give the user accurate and helpful information.
Using the context, which will be a combination of either/both swagger docs for certain relevant endpoints, and glossary entries for the user's query,
answer the user's query to the best of your ability. If you think you cant answer and you will instead hallucinate, respond with something along the lines of "I can't answer that at this time."
"""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(opey_system_prompt_template),
        MessagesPlaceholder("messages")
    ]
)

#prompt = hub.pull("opey_main_agent")

# LLM
llm = get_llm(size='medium', temperature=0.7).bind_tools([obp_requests])

# Chain
opey_agent = prompt | llm 


### Retrieval decider

from langchain_core.prompts import SystemMessagePromptTemplate, MessagesPlaceholder, ChatPromptTemplate

retrieval_decider_system_prompt = """You are an assistant that decides whether further context is needed from a vector database search to answer a user's question.
Using the chat history, look at whether the user's question can be answered from the given context or not. If it can be answered already or if
the user's input does not require any context (i.e. they just said 'hello') just answer the question.
If context is needed, decide to use either or both of the glossary retrieval tool or the endpoint retreival tool.

The endpoints vector store contains the swagger definitions of all the endpoints on the Open Bank Project API
The glossary vector store contains technical documents on many topics pertaining to Open Banking and the OBP API itself, such as how to authenticate.

When calling the endpoint or glossary retrievers, formulate a relevant query. use the messages to come up with a short search query to search the vector database of either glossary items or partial swagger specs for API endpoints.
The query needs to be in the form of a natural sounding question that conveys the semantic intent of the message, taking into account the message history

If the mode is glossary_retrieval, optimise the query to search a glossary of technical documents for the Open Bank Project (OBP)
If the mode is endpoint_retrieval, optimise the query to search through swagger schemas of different endpoints on the Open Bank Project (OBP) API
"""

retrieval_decider_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=retrieval_decider_system_prompt),
        MessagesPlaceholder("messages"),
    ]
)

retrieval_decider_llm = get_llm(size='medium', temperature=0).bind_tools([glossary_retrieval_tool, endpoint_retrieval_tool])
retrieval_decider_chain = retrieval_decider_prompt_template | retrieval_decider_llm


### Retrieval Query Formulator

class QueryFormulatorOutput(BaseModel):
    query: str = Field(description="Query to be used in vector database search of either glossary items or swagger specs for endpoints.")

query_formulator_system_prompt = """You are a query formulator that takes a list of messages and a mode: {retrieval_mode}
and tries to use the messages to come up with a short search query to search a vector database of either glossary items or partial swagger specs for API endpoints.
The query needs to be in the form of a natural sounding question that conveys the semantic intent of the message, especially the latest message from the human user.

If the mode is glossary_retrieval, optimise the query to search a glossary of technical documents for the Open Bank Project (OBP)
If the mode is endpoint_retrieval, optimise the query to search through swagger schemas of different endpoints on the Open Bank Project (OBP) API

Here are a list of API endpoint tags that you can use to help you write the query. Each tag is a keyword that is associated with a group of endpoints.
Identify the most relevant tags and use them in the query to help the vector search find the most relevant endpoints.
    
    - Old-Style
    - Transaction-Request
    - API
    - Bank
    - Account
    - Account-Access
    - Direct-Debit
    - Standing-Order
    - Account-Metadata
    - Account-Application
    - Account-Public
    - Account-Firehose
    - FirehoseData
    - PublicData
    - PrivateData
    - Transaction
    - Transaction-Firehose
    - Counterparty-Metadata
    - Transaction-Metadata
    - View-Custom
    - View-System
    - Entitlement
    - Role
    - Scope
    - OwnerViewRequired
    - Counterparty
    - KYC
    - Customer
    - Onboarding
    - User
    - User-Invitation
    - Customer-Meeting
    - Experimental
    - Person
    - Card
    - Sandbox
    - Branch
    - ATM
    - Product
    - Product-Collection
    - Open-Data
    - Consumer
    - Data-Warehouse
    - FX
    - Customer-Message
    - Metric
    - Documentation
    - Berlin-Group
    - Signing Baskets
    - UKOpenBanking
    - MXOpenFinance
    - Aggregate-Metrics
    - System-Integrity
    - Webhook
    - Mocked-Data
    - Consent
    - Method-Routing
    - WebUi-Props
    - Endpoint-Mapping
    - Rate-Limits
    - Counterparty-Limits
    - Api-Collection
    - Dynamic-Resource-Doc
    - Dynamic-Message-Doc
    - DAuth
    - Dynamic
    - Dynamic-Entity
    - Dynamic-Entity-Manage
    - Dynamic-Endpoint
    - Dynamic-Endpoint-Manage
    - JSON-Schema-Validation
    - Authentication-Type-Validation
    - Connector-Method
    - Berlin-Group-M
    - PSD2
    - Account Information Service (AIS)
    - Confirmation of Funds Service (PIIS)
    - Payment Initiation Service (PIS)
    - Directory
    - UK-AccountAccess
    - UK-Accounts
    - UK-Balances
    - UK-Beneficiaries
    - UK-DirectDebits
    - UK-DomesticPayments
    - UK-DomesticScheduledPayments
    - UK-DomesticStandingOrders
    - UK-FilePayments
    - UK-FundsConfirmations
    - UK-InternationalPayments
    - UK-InternationalScheduledPayments
    - UK-InternationalStandingOrders
    - UK-Offers
    - UK-Partys
    - UK-Products
    - UK-ScheduledPayments
    - UK-StandingOrders
    - UK-Statements
    - UK-Transactions
    - AU-Banking
"""

query_formulator_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=query_formulator_system_prompt),
        MessagesPlaceholder("messages"),
    ]
)

query_formulator_llm = get_llm(size='medium', temperature=0).with_structured_output(QueryFormulatorOutput)
query_formulator_chain = query_formulator_prompt_template | query_formulator_llm