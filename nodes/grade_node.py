from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from state import GraphState


class RelevanceGrade(BaseModel):
    relevant: bool = Field(description="True if the document contains information relevant to answering the question")


def grade_documents(state: GraphState):
    print("---NODE: GRADING RETRIEVED DOCUMENTS---")
    question = state["question"]
    documents = state["context"]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(RelevanceGrade)

    filtered_docs = []
    for doc in documents:
        grade = llm.invoke([
            {
                "role": "system",
                "content": "You are a relevance grader. Given a document and a question, determine if the document contains information that would help answer the question. Be lenient — partial relevance counts.",
            },
            {"role": "user", "content": f"Document:\n{doc[:500]}\n\nQuestion: {question}"},
        ])
        if grade.relevant:
            filtered_docs.append(doc)

    print(f"---GRADER: {len(filtered_docs)}/{len(documents)} documents passed---")

    if not filtered_docs:
        print("---GRADER: All docs filtered — falling back to full context---")
        filtered_docs = documents

    return {"context": filtered_docs}
