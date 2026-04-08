from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from state import GraphState

def check_scope(state: GraphState):
    print("---NODE: GUARDRAIL CHECK---")
    question = state["question"]


    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    system_prompt = """You are a strict security routing assistant for a UPSC learning app.
    If the user's topic is related to the Indian Constitution, Indian Polity, or Fundamental Rights, reply ONLY with 'YES'.
    If the topic is completely unrelated (e.g., coding, cooking, history of other countries, math, chit-chat), reply ONLY with 'NO'."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    response = (prompt | llm).invoke({"question": question})
    
    is_valid = "YES" in response.content.upper()
    
    rejection_message = ""
    if not is_valid:
        rejection_message = "Out of Scope: Kalam is currently focusing exclusively on Indian Constitutional Law and Fundamental Rights. Please ask a related topic."
        print("Guardrail Blocked Query:", question)
    else:
        print("Guardrail Passed")

    return {"is_valid": is_valid, "rejection_message": rejection_message}