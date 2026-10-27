import time
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY
from src.vector_store import query_vector_store
from src.logger import logger

def format_retrieved_context(retrieved_chunks: list[dict]) -> str:
    """
    Formats retrieved chunks and their page numbers into a clean context block.
    """
    context_blocks = []
    for idx, ch in enumerate(retrieved_chunks, 1):
        file_info = f" | File: {ch['filename']}" if ch.get('filename') else ""
        block = f"--- EXCERPT {idx} (Source: Page {ch['page_number']}{file_info}) ---\n{ch['text']}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)

def answer_financial_query(
        question: str,
        top_k: int = 2,
        model_name: str = "gemini-3.5-flash"
) -> dict:
    """
    Executes the full RAG pipeline:
    1. Retrieve relevant chunks from ChromaDB
    2. Augment prompt with retrieved context
    3. Generate grounded answer via Gemini
    """
    logger.info(f"Executing RAG pipeline for: '{question}'")

    # 1. RETRIEVE
    retrieved_chunks = query_vector_store(question, top_k=top_k)
    context_str = format_retrieved_context(retrieved_chunks)

    # 2. AUGMENT: Strict financial analyst prompt template
    system_instruction = (
        "You are FinSight AI, a rigorous financial research assistant. "
        "Your task is to answer the user's question using ONLY the provided document excerpts.\n"
        "Strict rules:\n"
        "1. Only state facts directly supported by the context.\n"
        "2. If the context does not contain the answer, state: "
        "'The provided document does not contain information to answer this question.'\n"
        "3. Always cite the page number (and document name if available) for any figures or facts (e.g., [Page 1]).\n"
        "4. Do not speculate or use outside knowledge."
    )

    user_prompt = (
        f"FINANCIAL CONTEXT:\n{context_str}\n\n"
        f"USER QUESTION: {question}\n\n"
        "ANSWER:"
    )

    # 3. GENERATE
    client = genai.Client(api_key=GEMINI_API_KEY)

    response = None
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0,
                ),
            )
            break
        except Exception as e:
            if attempt == 3:
                raise e
            wait_s = (attempt + 1) * 6
            logger.warning(f"RAG transient error ({e}). Retrying in {wait_s}s...")
            time.sleep(wait_s)

    return {
        "question": question,
        "answer": response.text if response else "No response generated.",
        "context": context_str,
        "sources": [
            {
                "page": ch["page_number"],
                "filename": ch.get("filename", ""),
                "similarity": ch["similarity_score"]
            }
            for ch in retrieved_chunks
        ]
    }

if __name__ == "__main__":
    print("\n===============================================")
    print("      FinSight AI - Live RAG Demonstration     ")
    print("=================================================")

    query_1 = "What was Apple's total net sales for the three months ended June 27, 2026?"
    print(f"Testing Query: '{query_1}'\n")
    result_1 = answer_financial_query(query_1, top_k=2)

    print("\n--- FinSight AI Answer ---")
    print(result_1["answer"])
    print(f"\nSources Cited: {result_1['sources']}")
