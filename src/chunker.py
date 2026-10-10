from src.logger import logger

def smart_chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[dict]:
    """
    Splits text into chunks using:
    1. Boundary snapping (snaps to nearest newline or space to pren=vent cutting words/rows in half)
    2. Sliding window overlap (retains context across chunk transitions)
    """

    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size")

    chunks = []
    total_len = len(text)
    start = 0
    chunk_id = 0

    while start < total_len:
        # Initial target end
        target_end = start + chunk_size

        if target_end >= total_len:
            actual_end = total_len
        else:
            # 1. BOUNDARY SNAPPING: Walk backwards to find the nearest clean split
            # Priority A: Doubl newline (paragraph break)
            # Priority B: Single newline (table row break)
            # Priority C: Space (word break)
            window = text[start:target_end]

            snap_pos = -1
            for seperator in ["\n\n","\n"," "]:
                pos = window.rfind(seperator)
                if pos != -1 and pos > (chunk_size // 2):# Snap only if reasonably far into the chunk
                    snap_pos = pos+len(seperator)
                    break

            if snap_pos != -1:
                actual_end = start + snap_pos
            else:
                actual_end = target_end # Fallback: cut at target if no seperator found

                
        chunk_text = text[start:actual_end]

        if chunk_text:
            chunks.append({
                "chunk_id": chunk_id,
                "start_char": start,
                "end_char": actual_end,
                "text": chunk_text,
                "length": len(chunk_text)
            })
            chunk_id += 1

        # 2. SLIDING WINDOW OVERLAP:
        # Move forward, but step back bu chunk_overlap to share boundary context
        if actual_end >= total_len:
            break
        start = actual_end - chunk_overlap

    logger.info(
        f"Smart chunking complete: {total_len} chars -> {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks

def chunk_document_pages(
        cleaned_pages: list[dict],
        chunks_size: int = 500,
        chunk_overlap: int = 100,
) -> list[dict]:
    """
    Chunks a document page-by-page, attaching source metadata (page number)
    to every single chunk for auditability and citations.
    """

    all_chunks = []
    global_chunk_id = 0

    for page in cleaned_pages:
        page_num = page["page_number"]
        page_text = page["cleaned_text"]

        # Run smart chunking on this page's text
        page_chunks = smart_chunk_text(
            text=page_text,
            chunk_size=chunks_size,
            chunk_overlap=chunk_overlap,
        )

        for ch in page_chunks:
            all_chunks.append({
                "chunk_id": global_chunk_id,
                "page_number": page_num,
                "text": ch["text"],
                "char_count": ch["length"]
            })
            global_chunk_id += 1
    logger.info(
        f"Document chunking complete across {len(cleaned_pages)} pages: "
        f"Generated {len(all_chunks)} page-attributed chunks." 
    )
    return all_chunks

if __name__ == "__main__":
    from src.extractor import extracted_text_from_pdf
    from src.cleaner import clean_extracted_pages

    # 1. Extract and clean the report
    raw_pages = extracted_text_from_pdf("data/sample_report.pdf")
    cleaned_pages = clean_extracted_pages(raw_pages)

    # Combine all pages into one continuous document string
    full_document = "\n\n".join(p["cleaned_text"] for p in cleaned_pages)

    # RUn page-aware chunking
    final_chunks = chunk_document_pages(cleaned_pages, chunks_size=500, chunk_overlap=100)

    print("\n--- Sample Attributed Chunk ---")
    sample = final_chunks[0]
    print(f"Chunk ID: {sample['chunk_id']} | Page: {sample['page_number']} | Characters: {sample['char_count']}")
    print(sample["text"][:300])
    print("...")