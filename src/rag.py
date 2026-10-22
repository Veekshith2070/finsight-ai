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
        block = f"--- EXCEPT {idx} (Source: Page {ch['page_number']}) ---\n{ch['text']}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)

def answer_financial_query(
        question: str,
        top_k: int = 2,
        model_name: str = "gemini-3.5-flash"
) -> dict:
    """
    Executes the full RAG pipeline:
    1.Retriev relevant chunks from ChromaDB
    2. Augument prompt with retrieved context
    3. Generate grounded answer via Gemini
    """
    logger.info(f"Executing RAG pipeline for: '{question}'")

    # 1. RETRIEVE
    retrieved_chunks = query_vector_store(question, top_k=top_k)
    context_str = format_retrieved_context(retrieved_chunks)

    # 2. AUGMENT: Strict finanacial analyst prompt template
    system_instruction = (
        "You are FinSight AI, a rigorous financial research assistant. "
        "Your task is to answer the user's question using ONLY the provided document excerpts."
        "Strict rules: \n"
        "1. only state facts directly supported by the context. \n"
        "2. If the context does not contain the answer, state: "
        "'The provided document does not contain information to answer this question.'\n"
        "3. Always cite the page number fr any figures or facts (e.r., [Page 1]). \n"
        "4. Do not speculate or use outside knowledge."
    )

    user_prompt = (
        f"FINANCIAL CONTEXT:\n{context_str}\n\n"
        f"USER QUESTION: {question}\n\n"
        "ANSEWR:"
    )

    # 3. GENERATE
    client = genai.Client(api_key=GEMINI_API_KEY)

    import time
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
        "answer": response.text,
        "context": context_str,
        "sources": [
            {"page": ch["page_number"], "similarity": ch["similarity_score"]}
            for ch in retrieved_chunks
        ]
    }

if __name__ == "__main__":
    print("\n===============================================")
    print("      FinSigh AI - Live RG Demonstration       ")
    print("=================================================")

    # TEST 1: Question contained in the apple report
    query_1 = "What was Aplle's toal net sales, and how much was from Services for the three months ended June 27, 2026?"

    print(f"Testing Query 1: '{query_1}'\n")
    result_1 = answer_financial_query(query_1, top_k=2)

    print("\n--- FinSight AI Answer ---")
    print(result_1["answer"])
    print(f"\nSources Cited: {result_1['sources']}")

    print("\n"+"="*55+"\n")

    # TEST 2: Out-of-domain question to test grounding and anti-hallucination
    query_2 = "What was MIcrosoft's Azure cloud revenue last quarter?"

    print(f"testing Query 2 (hallucination Test): '{query_2}'\n")
    result_2 = answer_financial_query(query_2, top_k=2)

    print("\n--- FinSight AI Answer ---")

    print(result_2["answer"])