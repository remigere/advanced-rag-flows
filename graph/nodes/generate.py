from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate(state: GraphState, ) -> Dict[str, Any]:
    """Generates an answer to the user question based on the retrieved documents.
    
    Args:
        state (dict): The current state of the graph.
    
    Returns:
        state (dict): Updated state with a generated answer.
    """
    
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    
    generation = generation_chain.invoke({"context": documents, "question": question})
    
    return {"generation": generation, "question": question}