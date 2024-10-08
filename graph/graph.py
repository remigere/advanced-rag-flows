from dotenv import load_dotenv

from langgraph.graph import END, StateGraph
from regex import P

from graph.consts import RETRIEVE, GRADE_DOCUMENTS, WEBSEARCH, GENERATE
from graph.nodes import generate, web_search, grade_documents, retrieve
from graph.state import GraphState

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery

load_dotenv()

def decide_to_generate(state: GraphState):
    print("---ASSESS GRADED DOCUMENTS---")
    
    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT, NEED TO SEARCH THE WEB---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "not supported"
    
def route_question(state: GraphState):
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO VECTOR STORE---")
        return RETRIEVE
    elif source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH

workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(GENERATE, generate)

# workflow.set_entry_point(RETRIEVE)
workflow.set_conditional_entry_point(
    route_question,
    {
        RETRIEVE: RETRIEVE,
        WEBSEARCH: WEBSEARCH,
    }
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS, 
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    })
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "useful": END,
        "not useful": WEBSEARCH,
        "not supported": WEBSEARCH,
    }
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)


app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
