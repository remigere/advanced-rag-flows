from typing import Any, Dict

from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from graph.state import GraphState

web_search_tool = TavilySearchResults(max_results=3)

from dotenv import load_dotenv
load_dotenv()

def web_search(state: GraphState, ) -> Dict[str, Any]:
    """Searches the web for relevant documents based on the user question.
    
    Args:
        state (dict): The current state of the graph.
    
    Returns:
        state (dict): Updated state with new documents from web search.
    """
    
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    
    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_results = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    return {"documents": documents, "question": question}

if __name__ == "__main__":
    web_search(
        state = {
        "question": "agent memory",
        "documents": None
    }
    )
