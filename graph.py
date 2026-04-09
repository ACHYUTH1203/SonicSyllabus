import logging
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from state import GraphState
from nodes.retrieve_node import retrieve
from nodes.grade_node import grade_documents
from nodes.generate_node import generate_english_script
from nodes.translate_node import translate_script
from nodes.guardrail_node import check_scope

# Set up the logger
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

load_dotenv()

def route_after_guardrail(state: GraphState):
    """If the query is invalid, end the graph immediately."""
    is_valid = state.get("is_valid")
    logger.info(f"[ROUTING] Guardrail check complete. is_valid={is_valid}")
    
    if is_valid:
        logger.info("[ROUTING] Proceeding to -> 'retrieve' node")
        return "retrieve"
        
    logger.warning("[ROUTING] Query invalid. Proceeding to -> END")
    return END

def route_translation(state: GraphState):
    """If the language is English, skip translation and finish."""
    language = state.get("target_language", "English").strip().lower()
    logger.info(f"[ROUTING] Checking target language: '{language}'")
    
    if language == "english":
        logger.info("[ROUTING] English selected. Skipping translation. Proceeding to -> END")
        return END
        
    logger.info("[ROUTING] Translation needed. Proceeding to -> 'translate_dynamic' node")
    return "translate_dynamic"

def build_and_run_graph(user_query: str, language: str):
    """
    Builds the LangGraph pipeline, compiles it, and runs the user query.
    Returns the final state containing the generated scripts.
    """
    logger.info("[GRAPH] Initializing StateGraph...")
    workflow = StateGraph(GraphState)
    
    workflow.add_node("guardrail", check_scope)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_english", generate_english_script)
    workflow.add_node("translate_dynamic", translate_script)
    
    logger.info("[GRAPH] Setting entry point to 'guardrail'")
    workflow.set_entry_point("guardrail")
    
    workflow.add_conditional_edges(
        "guardrail", 
        route_after_guardrail, 
        {
            "retrieve": "retrieve", 
            END: END
        }
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "generate_english")
    
    workflow.add_conditional_edges(
        "generate_english", 
        route_translation, 
        {
            "translate_dynamic": "translate_dynamic", 
            END: END
        }
    )

    workflow.add_edge("translate_dynamic", END)

    logger.info("[GRAPH] Compiling workflow...")
    app = workflow.compile()

    initial_state = {
        "question": user_query,
        "target_language": language,
        "is_valid": True 
    }
    
    logger.info(f"[EXECUTION] Invoking graph with query: '{user_query}' | Language: '{language}'")
    final_state = app.invoke(initial_state)
    logger.info("[EXECUTION] Graph execution completed successfully.")
    
    return final_state