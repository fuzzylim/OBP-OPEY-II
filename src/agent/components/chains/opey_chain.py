### Main Opey agent

from langchain import hub
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import MessagesPlaceholder

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
llm = ChatOpenAI(model_name="gpt-4o", temperature=0).bind_tools([obp_requests])

# Chain
opey_agent = prompt | llm 