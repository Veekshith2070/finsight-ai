from pathlib import Path
import chromadb
from src.logger import logger
from src.embeddings import generate_embedding, embed_chunks
from src.extractor import extracted_text_from_pdf
from src.cleaner import clean_extracted_pages
from src.chunker import chunk_document_pages

# Directory on disk where ChromaDB stores vectors permanently
PERSIST_DOCUMENTORY = Path("chroma_db")

def get_chroma_client() -> chromadb.PersistentClient:
    """Returns a persistent ChromaDB client that writes to local disk."""
    return chromadb.PersistentClient(path=str(PERSIST_DOCUMENTORY))

def get_or_create_collection(
        collection_name: str = "finsight_financial_docs"
) -> chromadb.Collection:
    """Gets an existing collection or creates a new one."""
    client  = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name,
        #We tell Chroma we are using cosine distance
        metadata={"hnsm:space": "cosine"}
    )

def index_document_chunks(
        embedded_chunks: list[dict],
        collection_name: str = "finsight_financial_docs"
) -> None:
    """
    Stores documents, vectors, and metadata into ChrmaDB.
    """
    collection = get_or_create_collection(collection_name)

    ids = [str(ch["chunk_id"]) for ch in embedded_chunks]
    documents = [ch["text"] for ch in embedded_chunks]
    embeddings = [ch["embedding"] for ch in embedded_chunks]
    metadatas = [{"page_number": ch["page_number"]} for ch in embedded_chunks]

    logger.info(f"Indexing {len(ids)} chunks into ChromaDB collection [{collection_name}]...")

    # Upsert prevents duplicate errors if re-run
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    logger.info(f"Indexing complete. Total items in collection: {collection.count()}")

def query_vector_store(
        query_text: str,
        top_k: int = 2,
        collection_name: str = "finsight_financial_docs"
) -> list[dict]:
    """
    Embeds the user query, queries ChrmaDb's HNSW, and returns top results.
    """
    collection = get_or_create_collection(collection_name)

    logger.info(f"Querying ChromaDB for: '{query_text}'")
    query_vector = generate_embedding(query_text)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    formatted_results=[]
    #Chroma return lists of lists for batch queries
    for idx in range(len(results["ids"][0])):
        # Convert cosine distance to cosine similarity: similarity = 1 - distance
        cosine_distance = results["distances"][0][idx]
        cosine_similarity = 1.0 - cosine_distance

        formatted_results.append({
            "chunk_id": results["ids"][0][idx],
            "text": results["documents"][0][idx],
            "page_number": results["metadatas"][0][idx]["page_number"],
            "similarity_score": cosine_similarity
        })

    return formatted_results

if __name__ == "__main__":
    collection = get_or_create_collection()

    # Check if vectors are already saved on disk
    if collection.count()==0:
        logger.info("ChromaDb is empty. Ingesting and embedding document...")
        raw_pages = extracted_text_from_pdf("data/sample_report.pdf")
        cleaned_pages = clean_extracted_pages(raw_pages)
        chunks = chunk_document_pages(cleaned_pages, chunks_size=500, chunk_overlap=100)
        embedded_chunks = embed_chunks(chunks)
        index_document_chunks(embedded_chunks)
    else:
        logger.info(f"Found existing index with {collection.count()} chunks. Skipping ingestion!")

    # Test query
    test_query = "What were the sales for the products and services"
    matches = query_vector_store(test_query, top_k=2)

    print("\n=================================")
    print(f"CHROMADB SEARCH: '{test_query}'")
    print("===================================")

    for rank, m in enumerate(matches, 1):
        print(f"--- MATCH #{rank} (Similarity: {m['similarity_score']:.4f} | Page: {m['page_number']})---")
        print(m["text"])
        print("-"*50)
