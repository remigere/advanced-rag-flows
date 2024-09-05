from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GradeHallucinations(BaseModel):
    """Binary score for hallucination presence in generated answer."""
    binary_score: bool = Field(
        description="Answer is grounded in facts, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether a generated answer is grounded in or supported by a set of documents. \n 
Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
