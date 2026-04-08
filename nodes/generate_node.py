from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from state import GraphState

def generate_english_script(state: GraphState):
    print("---NODE: GENERATING ENGLISH SCRIPT---")
    question = state["question"]
    context = state["context"]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    

    system_prompt = """You are the lead UPSC mentor for SuperKalam.
Your objective is to write a highly engaging, 1-minute audio revision script based strictly on the provided context.

<role_and_tone>
- Speak directly to the student. Use an encouraging, authoritative, and accessible tone.
- Do not sound like a textbook. Sound like a passionate teacher explaining a concept.
</role_and_tone>

<rag_constraints>
- You MUST base your explanation ONLY on the provided context.
- If the context mentions how a topic was tested in a Previous Year Question (PYQ) or mentions specific Supreme Court cases, you MUST highlight that explicitly.
- If the context does not contain enough information, clearly state: "I don't have enough verified information in our current syllabus to cover that fully."
</rag_constraints>

<audio_formatting_constraints>
CRITICAL: This text will be fed directly into a Text-to-Speech (TTS) engine.
1. Keep the script strictly under 150 words to ensure rapid synthesis. Brevity is key.
2. DO NOT use emojis, asterisks, hashtags, bullet points, or markdown formatting.
3. Spell out numbers, percentages, and dates naturally.
4. Use commas and periods strategically to force natural breathing pauses.
</audio_formatting_constraints>

Context Documents:
{context}
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Write an audio script explaining: {question}")
    ])
    
    chain = prompt | llm
    
    joined_context = "\n\n".join(context)
    
    response = chain.invoke({
        "context": joined_context, 
        "question": question
    })
    
    return {"english_script": response.content}