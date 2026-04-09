from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from state import GraphState


def retrieve(state: GraphState):
    print("---NODE: RETRIEVING CONTEXT---")
    question = state["question"]


    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    hyde_prompt = (
        f"Write a short passage (2-3 sentences) from an Indian Constitutional Law "
        f"textbook explaining: {question}"
    )
    hypothetical_doc = llm.invoke(hyde_prompt).content
    print(f"---HyDE QUERY: {hypothetical_doc[:100]}---")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.7},
    )
    docs = retriever.invoke(hypothetical_doc)

    context = [doc.page_content for doc in docs]

    return {"context": context}