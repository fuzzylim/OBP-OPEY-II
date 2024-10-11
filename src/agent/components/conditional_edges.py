def run_retrieval_decider(state: OpeyGraphState):
    state["current_state"] = "retrieval_decider"
    messages = state["messages"]
    output = retrieval_decider_chain.invoke({"messages": messages})
    
    print(f"Retrieval decider: {output}")
    
    if output.context_needed:
        print("Further context needed")
        if output.retrieve_endpoints and output.retrieve_glossary:
            return ["retrieve_endpoints", "retreive_glossary"]
        elif output.retrieve_endpoints:
            return "retrieve_endpoints"
        elif output.retrieve_glossary:
            return "retrieve_glossary"
        else:
            print("error - context needed but retrieval not recognized")
            return "opey"
    else:
        print("No further context needed - answer question")
        return "opey"