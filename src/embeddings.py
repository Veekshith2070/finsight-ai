import time
import re
from google import genai
from src.config import GEMINI_API_KEY
from src.logger import logger

def get_embedding_client() -> genai.Client:
    """Returns the authenticated Gemini client."""
    return genai.Client(api_key=GEMINI_API_KEY)

def generate_embedding(
        text: str,
        model_name: str = "gemini-embedding-001"
) -> list[float]:
    """
    Passes a text string to the embedding model and returns
    its 3,072-dimensional coordinate vector.
    """
    if not text.strip():
        raise ValueError("Cannot embed empty text.")

    client = get_embedding_client()

    for attempt in range(6):
        try:
            response = client.models.embed_content(
                model=model_name,
                contents=text,
            )
            embedding_vector = response.embeddings[0].values
            return embedding_vector
        except Exception as e:
            err_msg = str(e)
            if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < 5:
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg)
                if not match:
                    match = re.search(r"retryDelay':\s*'(\d+)s'", err_msg)
                wait_time = float(match.group(1)) + 2.0 if match else 35.0
                logger.warning(f"Embedding rate limit reached. Pausing {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                if attempt == 5:
                    logger.error(f"Failed to generate embedding: {e}")
                    raise e
                time.sleep(3)

def embed_chunks(
        chunks: list[dict],
        model_name: str = "gemini-embedding-001",
        batch_size: int = 50
) -> list[dict]:
    """
    Generates and attaches embedding vectors to each chunk dictionary
    using high-throughput batched API calls with exponential backoff and rate limit recovery.
    """
    if not chunks:
        return []

    logger.info(f"Generating embeddings for {len(chunks)} chunks using [{model_name}] in batches of {batch_size}...")

    client = get_embedding_client()
    embedded_chunks = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [ch["text"] for ch in batch]

        # Retry loop for transient rate limits
        response = None
        for attempt in range(8):
            try:
                response = client.models.embed_content(
                    model=model_name,
                    contents=texts
                )
                break
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg)
                    if not match:
                        match = re.search(r"retryDelay':\s*'(\d+)s'", err_msg)
                    wait_time = float(match.group(1)) + 2.0 if match else 40.0
                    logger.warning(f"Embedding rate limit reached (100 RPM quota). Pausing {wait_time:.1f}s for quota replenishment...")
                    time.sleep(wait_time)
                else:
                    if attempt == 7:
                        logger.error(f"Embedding batch failed after 8 attempts: {e}")
                        raise e
                    wait_time = (attempt + 1) * 3
                    logger.warning(f"Batch embedding error ({e}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        for ch, emb in zip(batch, response.embeddings):
            embedded_chunks.append({
                "chunk_id": ch["chunk_id"],
                "page_number": ch["page_number"],
                "text": ch["text"],
                "filename": ch.get("filename", ""),
                "embedding": emb.values,
                "dimensions": len(emb.values)
            })

        logger.info(f"Embedded batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size} ({len(embedded_chunks)}/{len(chunks)} chunks)")
        time.sleep(0.5)

    logger.info(f"Successfully embedded {len(embedded_chunks)} chunks.")
    return embedded_chunks

if __name__ == "__main__":
    sample_texts = [
        "Apple Inc. iPhone products net sales",
        "Quarterly earnings, profit, and revenue",
        "Microsoft Cloud and Azure services growth"
    ]

    print("\n--- Generating Embeddings ---")
    for text in sample_texts:
        vector = generate_embedding(text)
        print(f"\nText: '{text}'")
        print(f"Vector Dimensions: {len(vector)}")
        print(f"First 5 Numbers: {[round(x, 4) for x in vector[:5]]}")
