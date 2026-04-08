
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from graph import build_and_run_graph

load_dotenv()


class GroundedGrade(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score")
    grounded: bool = Field(description="True if the answer is grounded in facts, False if it hallucinates")

class RelevanceGrade(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score")
    relevant: bool = Field(description="True if the answer addresses the question, False otherwise")

grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.
Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct."""

relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION
Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct."""

def evaluate_pipeline():
    print("\n" + "="*50)
    print("Starting Custom LangChain RAG Evaluation")
    print("="*50)

    grounded_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(GroundedGrade)
    relevance_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(RelevanceGrade)

    test_questions = [
        "Explain Article 21 and the protection of life and personal liberty.",
        "What are the exceptions to the Right to Equality under Article 14?",
        "How is the Right to Freedom of Religion protected?"
    ]

    results = []

    for query in test_questions:
        print(f"\nRunning pipeline for: '{query}'")
        
        final_state = build_and_run_graph(query, "English")
        
        answer = final_state["english_script"]
        contexts = final_state["context"]
        
        doc_string = "\n\n".join(contexts)

        print("Judging Groundedness (Faithfulness)...")
        grounded_prompt = f"FACTS: {doc_string}\nSTUDENT ANSWER: {answer}"
        grounded_grade = grounded_llm.invoke([
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": grounded_prompt}
        ])

        print("Judging Relevance (Answer Relevancy)...")
        relevance_prompt = f"QUESTION: {query}\nSTUDENT ANSWER: {answer}"
        relevance_grade = relevance_llm.invoke([
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": relevance_prompt}
        ])

        results.append({
            "question": query,
            "grounded_score": grounded_grade.grounded,
            "relevance_score": relevance_grade.relevant,
            "grounded_explanation": grounded_grade.explanation,
            "relevance_explanation": relevance_grade.explanation
        })

    df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(df[["question", "grounded_score", "relevance_score"]])
    
    csv_filename = "rag_evaluation_report.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nFull detailed report saved to: {csv_filename}")

if __name__ == "__main__":
    evaluate_pipeline()