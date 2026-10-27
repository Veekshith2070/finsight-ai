import time
import re
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY
from src.vector_store import query_vector_store
from src.tools import (
    calculate_percentage_growth,
    calculate_financial_margin,
    calculate_financial_ratio,
    calculate_financial_difference,
)
from src.logger import logger

# 1. Wrap our RAG engine as an Agent Tool
def search_financial_document(query: str) -> str:
    """
    Searches the ingested financial documents (e.g., Apple 10-Q, Microsoft 10-K, etc.)
    for relevant factual context, tables, revenue figures, expenses, and notes.

    Args:
        query: The specific topic, keyword, or question to search for in the document.
    """
    logger.info(f"Agent Tool Invoked: search_financial_document(query='{query}')")
    results = query_vector_store(query, top_k=4)

    if not results:
        return "No relevant excerpts found in the documents."

    formatted = []
    for r in results:
        file_info = f" ({r['filename']})" if r.get('filename') else ""
        formatted.append(f"[Source: Page {r['page_number']}{file_info}]\n{r['text']}")

    return "\n\n".join(formatted)

# 2. Bundle all capabilities into the Agent's toolkit
ALL_AGENT_TOOLS = [
    search_financial_document,
    calculate_percentage_growth,
    calculate_financial_margin,
    calculate_financial_ratio,
    calculate_financial_difference
]

def run_financial_agent(
        user_prompt: str,
        model_name: str = "gemini-3.5-flash-lite"
) -> str:
    """
    Executes the autonomous agent loop using Gemini 3.5 Flash Lite with automatic
    rate-limit retry and quota replenishment backoff.
    """
    logger.info(f"Agent received user prompt: '{user_prompt}'")
    client = genai.Client(api_key=GEMINI_API_KEY)

    system_instruction = (
        "You are FinSight AI's Quantitative Financial Research Agent. "
        "You have access to tools for searching SEC financial reports and executing deterministic mathematical calculations.\n"
        "Guidelines:\n"
        "1. When answering questions about financial figures, ALWAYS use `search_financial_document` to look up accurate data from the filings first.\n"
        "2. Retrieve all necessary financial metrics in a comprehensive search query rather than making repeated redundant searches.\n"
        "3. NEVER calculate financial math (growth %, margins, ratios, variances) in your head! ALWAYS call the appropriate calculation tool.\n"
        "4. Include source page citations (e.g., [Page 1]) in your final response.\n"
        "5. If a calculation is requested, show the formula, the exact numbers used, and the tool-calculated result.\n"
        "6. Be concise, objective, and institutional-grade in your final output."
    )

    response = None
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=ALL_AGENT_TOOLS,
                    temperature=0.0
                ),
            )
            break
        except Exception as e:
            err_msg = str(e)
            if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < 4:
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg)
                if not match:
                    match = re.search(r"retryDelay':\s*'(\d+)s'", err_msg)
                wait_time = float(match.group(1)) + 2.0 if match else 25.0
                logger.warning(
                    f"Agent rate limit encountered (attempt {attempt + 1}/5). "
                    f"Pausing {wait_time:.1f}s for quota replenishment..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Agent execution failed: {e}")
                raise e

    return response.text if response else "No response generated."

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("      FinSight AI - Autonomous Agent Loop")
    print("=" * 50)

    prompt = (
        "Calculate the year-over-year percentage growth for Apple's Services revenue "
        "for the three months ended June 27, 2026 compared to June 28, 2025."
    )

    print(f"\nUser Prompt: {prompt}\n")
    agent_output = run_financial_agent(prompt)
    print("\n--- Final Agent Report ---")
    print(agent_output)
