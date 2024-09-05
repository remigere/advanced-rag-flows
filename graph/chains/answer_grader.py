import struct
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GradeAnswer(BaseModel):
    """Binary score for whether the generated answer resolves the question."""
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether a generated answer addresses / resolves a user question. \n
Give a binary score 'yes' or 'no' score to indicate whether the answer addresses the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
