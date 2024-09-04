from typing import List, TypedDict


class GraphState(TypedDict):
    """Represents the state of the graph.

    Args:
        question: question
        generation: LLM generation
        web_search: boolean flag whether web search is enabled
        documents: list of documents
    """
    
    question: str
    generation: str
    web_search: bool
    documents: List[str]
