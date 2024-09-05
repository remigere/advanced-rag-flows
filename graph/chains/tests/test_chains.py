from pprint import pprint


from typing import List
from dotenv import load_dotenv
from langchain_core.documents.base import Document

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from ingestion import retriever
    
def test_retrival_grade_answer_yes():
    question = "agent memory"
    docs: List[Document] = retriever.invoke(question)
    doc_txt = docs[1].page_content
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )
    assert res.binary_score == "yes"
    
    
def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs: List[Document] = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    assert res.binary_score == "no"
    
def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)
 
def test_hallucination_grader_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke({"documents": docs, "generation": generation})
    print(res.binary_score)
    assert res.binary_score
    
def test_hallucination_grader_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke({"documents": docs, "generation": "in order to make pizza, you need to have a dough"})
    print(res.binary_score)
    assert not res.binary_score