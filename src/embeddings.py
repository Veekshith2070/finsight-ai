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
    its 768-dimensional coordinate vector.
    """
    if not text.strip():
        raise ValueError("Cannot embed empty text.")

    client = get_embedding_client()

    try:
        response = client.models.embed_content(
            model=model_name,
            contents=text,
        )

        # Extract the vector (list of floats)
        embedding_vector = response.embeddings[0].values
        return embedding_vector
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise e

def embed_chunks(chunks: list[dict], model_name: str = "gemini-embedding-001") -> list[dict]:
    """
    Generates and attaches embedding vectors to each chunk dictionary.
    """
    logger.info(f"Generating embeddings for {len(chunks)} chunks using [{model_name}]...")

    embedded_chunks = []
    for ch in chunks:
        vector = generate_embedding(ch["text"], model_name=model_name)

        embedded_chunk = {
            "chunk_id": ch["chunk_id"],
            "page_number": ch["page_number"],
            "text": ch["text"],
            "embedding": vector,
            "dimensions": len(vector)
        }
        embedded_chunks.append(embedded_chunk)

    logger.info(f"Successfully embedded {len(embedded_chunks)} chunks (Dimensions: {embedded_chunks[0]['dimensions']}).")
    return embedded_chunks

if __name__ == "__main__":
    # Test with 3 different phrases
    sample_texts = [
        "Apple Inc. iPhone products net sales",
        "Quaterly earnings, profit, and revenue",
        "How to bake a chocolate chip cookie at home"
    ]

    print("\n--- Generating Embeddings ---")
    for text in sample_texts:
        vector = generate_embedding(text)
        print(f"\nText: '{text}'")
        print(f"Vector Dimensions: {len(vector)}")
        print(f"First 5 Numbers: {[round(x, 4) for x in vector[:5]]}")