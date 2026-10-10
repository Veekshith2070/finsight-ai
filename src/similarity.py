import numpy as np
from src.embeddings import generate_embedding, embed_chunks
from src.extractor import extracted_text_from_pdf
from src.cleaner import clean_extracted_pages
from src.chunker import chunk_document_pages
from src.logger import logger

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Computes cosine similarity between two high-dimensional vectors:
    cos(theta) = (A.B)/(|A|*|B|)
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)

    # 1. Dot product
    dot_product = np.dot(a, b)

    # 2. Vector magnitudes (L2 norms)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # 3. Handle zero vectors safely
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(dot_product / (norm_a*norm_b))

def find_most_relevant_chunks(
        query: str,
        embedded_chunks: list[dict],
        top_k: int = 3
) -> list[dict]:
    """
    Ranks all document chunks against the user query using cosine similarity
    and returns the top_k best matches.
    """
    logger.info(f"Generating query embedding for: '{query}'")
    query_vector = generate_embedding(query)

    scored_chunks = []

    for ch in embedded_chunks:
        score = cosine_similarity(query_vector, ch["embedding"])
        scored_chunks.append({
            "chunk_id": ch["chunk_id"],
            "page_number": ch["page_number"],
            "score":score,
            "text": ch["text"]
        })

    # Sort descending by similarity score (highest first)
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    return scored_chunks[:top_k]

if __name__ == "__main__":
    # 1. Ingest, clean, chunk, and embed our Apple PDF
    raw_pages = extracted_text_from_pdf("data/sample_report.pdf")
    cleaned_pages = clean_extracted_pages(raw_pages)
    chunks = chunk_document_pages(cleaned_pages, chunks_size=500, chunk_overlap=100)

    # Embed all chunks
    embedded_chunks = embed_chunks(chunks)

    # 2. test query (Notice: words 'iPhnoe' and 'sales' are used, not 'Products')
    test_query = "What are the sales for products and services?"

    print("\n==================================")
    print(f"SEARCH QUERY: '{test_query}'")
    print("====================================")

    top_matches = find_most_relevant_chunks(test_query,embedded_chunks, top_k=2)

    for rank, match in enumerate(top_matches, 1):
        print(f"--- MATCH #{rank} (Similarity Score: {match['score']:.4f} | Page: {match['page_number']}) ---")
        print(match["text"])
        print("-"*50)