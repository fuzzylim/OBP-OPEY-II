from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

### Document Grader chain
# For a given document, assesses whether it is relevant to the user's query
# TODO: might have to chage this to receive at least some of the messages

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
        
llm = ChatOpenAI(model="gpt-4o-mini")

llm_grader = llm.with_structured_output(GradeDocuments)

grader_system_prompt = """You are a grader assessing the relevance of a retrieved API endpoint or glossary entry to a user question. \n
    if the document contains keywords(s) or semantic meaning relevant to the user's query or if it is an endpoint that would help them acheive a specified task, grade it as relevant. 
    Give a binary score 'yes' or 'no', to indicate whether the document is relevant to the question.
"""

grader_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", grader_system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ]
)

retrieval_grader = grader_prompt_template | llm_grader


### Question Re-writer
# NOTE: we may not end up using this in subsequent versions. Have to run retrieval graph against evals (also have to make evals)
# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Prompt
system = """You are a question re-writer that converts an input question to a better version that is designed to find endpoints\n 
     in a vector index of API endpoints. Look at the input and try to reason about the underlying semantic intent / meaning.\n
     Reword the question in such a way so that vector search could find endpoints that would help the user acheive their task.
     Try different keywords/synonyms and API jargon if needed."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

endpoint_question_rewriter = re_write_prompt | llm | StrOutputParser()


