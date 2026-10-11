import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import GEMINI_API_KEY
from src.logger import logger

# Set environment variable expected by langchain-google-genai
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

def build_langchain_rag_chain():
    """
    Constructs a RAG pipeline using LangChain Expression Language (LCEL)
    """

    logger.info("Initializing LangChain RAG pipeline...")

    # 1. Embeddings Wrapper (Maps to our src/embeddings.py)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
    )

    # 2. VectoreStore Wrapper (Connects to our existing chroma_db directory from Day 3)
    vector_store = Chroma(
        collection_name="finsight_financial_docs",
        embedding_function=embeddings,
        persist_directory="chroma_db",
    )

    # Convert vectorstore into a LangChain Retriever (top_k = 2)
    retriever = vector_store.as_retriever(search_kwargs={"k":2})

    # 3. Prompt Template (Maps to our augmented prompt in src/rag.py)
    template = """You are FinSight AI, a rigorous finanacial research assistant.
    Answer the question using ONLY the provided document excerpts.
    Stict rules:
    1. Only state facts directly supported by the context.
    2. If the cotext does not contain the answer, state: 'The provided document does not contain information to answer this question.'
    3. Always cite the page number for any figures or facts (e.g., [Page 1]).
    4. Do not speculate or use outside knowledge.

    FINANCIAL CONTEXT:
    {context}

    USER QUESTION: {question}

    ANSWER:"""

    prompt = ChatPromptTemplate.from_template(template)

    # 4. LLM Wrapper (Maps to our src/llm.py eith temperature=0.0)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
    )

    # Helper function to format retrieved documents and their page metadata
    def format_docs(docs):
        formatted = []
        for d in docs:
            page = d.metadata.get("page_number", "Unknown")
            formatted.append(f"--- (Source: Page {page}) ---\n{d.page_content}")
        return "\n\n".join(formatted)

    # 5. LCEL Chain: (Input => Context/Question -> Prompt -> LLM -> String Output)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    chain = build_langchain_rag_chain()

    test_query = "What was Apple's total net sales, and how much was from Services for the three months ended June 27, 2026?"

    print("\n============================================================")
    print("        FinSight AI - LangChain LCEL RAG Demo               ")
    print("==============================================================\n")
    print(f"Query: '{test_query}'\n")
    answer = chain.invoke(test_query)

    print("--- FinSight AI (via LangChain) Answer ---")
    print(answer)