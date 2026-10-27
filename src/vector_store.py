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
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

def index_document_chunks(
        embedded_chunks: list[dict],
        collection_name: str = "finsight_financial_docs"
) -> None:
    """
    Stores documents, vectors, and metadata into ChromaDB.
    """
    if not embedded_chunks:
        return

    collection = get_or_create_collection(collection_name)

    ids = [str(ch["chunk_id"]) for ch in embedded_chunks]
    documents = [ch["text"] for ch in embedded_chunks]
    embeddings = [ch["embedding"] for ch in embedded_chunks]
    metadatas = [
        {"page_number": ch["page_number"], "filename": ch.get("filename", "")}
        for ch in embedded_chunks
    ]

    logger.info(f"Indexing {len(ids)} chunks into ChromaDB collection [{collection_name}]...")

    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )

    logger.info(f"Indexing complete. Total items in collection: {collection.count()}")

def query_vector_store(
        query_text: str,
        top_k: int = 2,
        collection_name: str = "finsight_financial_docs"
) -> list[dict]:
    """
    Embeds the user query, queries ChromaDB's HNSW, and returns top results.
    """
    collection = get_or_create_collection(collection_name)

    logger.info(f"Querying ChromaDB for: '{query_text}'")
    query_vector = generate_embedding(query_text)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    formatted_results = []
    if not results or not results["ids"] or not results["ids"][0]:
        return formatted_results

    for idx in range(len(results["ids"][0])):
        cosine_distance = results["distances"][0][idx]
        cosine_similarity = 1.0 - cosine_distance

        meta = results["metadatas"][0][idx] if results.get("metadatas") and results["metadatas"][0] else {}
        page_num = meta.get("page_number", 1) if meta else 1
        filename = meta.get("filename", "") if meta else ""

        formatted_results.append({
            "chunk_id": results["ids"][0][idx],
            "text": results["documents"][0][idx],
            "page_number": page_num,
            "filename": filename,
            "similarity_score": cosine_similarity
        })

    return formatted_results

def ingest_pdf_document(
        pdf_path: str | Path,
        collection_name: str = "finsight_financial_docs"
) -> dict:
    """
    End-to-end ingestion pipeline:
    Extracts text -> Cleans accounting formatting -> Chunks by page ->
    Generates embeddings -> Upserts to ChromaDB with collision-safe IDs.
    """
    path = Path(pdf_path)
    logger.info(f"Starting complete ingestion pipeline for: {path.name}")

    raw_pages = extracted_text_from_pdf(path)
    cleaned_pages = clean_extracted_pages(raw_pages)
    # Using 1200 char chunks preserves complete financial table sections & avoids quota bottlenecks
    chunks = chunk_document_pages(cleaned_pages, chunks_size=1200, chunk_overlap=150)

    # Prevent ID collisions across different documents & attach filename
    doc_stem = path.stem
    for ch in chunks:
        ch["chunk_id"] = f"{doc_stem}_chunk_{ch['chunk_id']}"
        ch["filename"] = path.name

    embedded_chunks = embed_chunks(chunks)
    index_document_chunks(embedded_chunks, collection_name=collection_name)

    return {
        "filename": path.name,
        "total_pages": len(raw_pages),
        "total_chunks": len(chunks),
        "status": "successfully_indexed"
    }

if __name__ == "__main__":
    collection = get_or_create_collection()

    if collection.count() == 0:
        logger.info("ChromaDB is empty. Ingesting sample report...")
        ingest_pdf_document("data/sample_report.pdf")
    else:
        logger.info(f"Found existing index with {collection.count()} chunks.")

    test_query = "What were the sales for the products and services"
    matches = query_vector_store(test_query, top_k=2)

    print("\n=================================")
    print(f"CHROMADB SEARCH: '{test_query}'")
    print("===================================")

    for rank, m in enumerate(matches, 1):
        print(f"--- MATCH #{rank} (Similarity: {m['similarity_score']:.4f} | Page: {m['page_number']})---")
        print(m["text"])
        print("-" * 50)
