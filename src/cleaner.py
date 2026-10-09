import re
from src.logger import logger

def clean_financial_text(raw_text: str) -> str:
    """
    Cleans raw extracted PDF text while preserving financial figures,
    decimals, percentages, and accounting parentheses.
    """

    if not raw_text:
        return ""

    text = raw_text

    # 1. Repalce non-breaking spaces (\xa0) and carriage returns (\r) with standard spaces
    text = text.replace("\xa0", " ").replace("\r", "\n")

    # 2. Fix broken hyphenated words across lines (e.g., "state-\nment" -> "statement")
    text = re.sub(r'(\b\w+)-\n(\w+\b)', r'\1\2', text)

    # 3. Collapse 3 or more consecutive newlines into a standard double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 4. Cpllapse multiple horizontal spaces/tabs into a single spce (while keeping linebreaks)
    text = re.sub(r'[ \t]+', ' ', text)

    # 5. Clean up leading/trailing whitespace on each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)

    return text.strip()

def clean_extracted_pages(pages: list[dict]) -> list[dict]:
    """
    Cleans text across all extracted PDF pages and updates character counts.
    """
    logger.info(f"Cleaning text across {len(pages)} extracted pages...")
    cleaned_pages = []

    for page in pages:
        cleaned_text = clean_financial_text(page["text"])
        cleaned_pages.append({
            "page_number": page["page_number"],
            "raw_text": page["text"],
            "cleaned_text": page["char_count"],
            "raw_chars": page["char_count"],
            "cleaned_text": cleaned_text,
            "cleaned_chars": len(cleaned_text),
        })

    total_raw = sum(p["raw_chars"] for p in cleaned_pages)
    total_clean = sum(p["cleaned_chars"] for p in cleaned_pages)
    reduction = ((total_raw - total_clean) / total_raw * 100) if total_raw > 0 else 0

    logger.info(
        f"Cleaning complete: {total_raw} raw chars -> {total_clean} cleaned chars "
        f"({reduction:.1f}% reduction in whitespace/noise)"
    )

    return cleaned_pages

if __name__ == "__main__":
    from src.extractor import extracted_text_from_pdf

    raw_pages = extracted_text_from_pdf("data/sample_report.pdf")
    cleaned = clean_extracted_pages(raw_pages)

    print("\n--- Cleaned Page 1 Sample ---")
    print(cleaned[0]["cleaned_text"][:600])
    print("...")