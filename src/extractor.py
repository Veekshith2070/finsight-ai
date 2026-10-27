from pathlib import Path
from pypdf import PdfReader
from src.logger import logger

# Configure pypdf safety budgets for large SEC financial filings (10-K / 10-Q)
try:
    from pypdf import overwrite_configuration
    overwrite_configuration(
        array_maximum_element_count=50_000_000,
        dictionary_maximum_element_count=50_000_000,
        stream_parsing_work_maximum_count=50_000_000,
        stream_decoding_work_maximum_length=500_000_000,
    )
except ImportError:
    pass

def extracted_text_from_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Extracts text page-by-page from a PDF file.

    Returns a list of dictionaries:
    [
       {"page_number": 1, "text": "...", "char_count": 1200},
       ...
    ]
    """

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    logger.info(f"Opening PDF: {path.name}")
    reader = PdfReader(str(path))
    total_pages = len(reader.pages)
    logger.info(f"Total pages detected: {total_pages}")

    extracted_pages = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        try:
            raw_text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Could not extract text from Page {page_num}: {e}")
            raw_text = ""

        extracted_pages.append({
            "page_number": page_num,
            "text": raw_text,
            "char_count": len(raw_text)
        })

    logger.info(f"Successfully extracted text from {total_pages} pages.")
    return extracted_pages

if __name__ == "__main__":
    test_pdf = Path("data/sample_report.pdf")

    pages = extracted_text_from_pdf(test_pdf)

    print("\n---Extraction Summary ---")
    print(f"Total Pages: {len(pages)}")

    # Inspect Page 1
    if pages:
        print(f"\n--- Page 1 (Characters: {pages[0]['char_count']}) ---")
        # Print first 500 characters of Page 1
        print(pages[0]["text"][:500])
        print("...")
