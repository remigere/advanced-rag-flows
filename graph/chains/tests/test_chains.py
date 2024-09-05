from pprint import pprint


from typing import List
from dotenv import load_dotenv
from langchain_core.documents.base import Document

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader

from ingestion import retriever

# def test_generation_chain() -> None:
#     question = "agent memory"
#     docs = retriever.invoke(question)
#     generation = generation_chain.invoke({"context": docs, "question": question})
#     pprint(generation)
    
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
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    assert res.binary_score == "no"
    
