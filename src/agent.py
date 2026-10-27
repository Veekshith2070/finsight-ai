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
        model_name: str = "gemini-3.5-flash"
) -> str:
    """
    Executes the autonomous agent loop using Gemini 3.5 Flash.
    The agent can dynamically decide to retrieve documents, perform math,
    or directly answer questions.
    """
    logger.info(f"Agent received user prompt: '{user_prompt}'")
    client = genai.Client(api_key=GEMINI_API_KEY)

    system_instruction = (
        "You are FinSight AI's Quantitative Financial Research Agent. "
        "You have access to tools for searching SEC financial reports and executing deterministic mathematical calculations.\n"
        "Guidelines:\n"
        "1. When answering questions about financial figures, ALWAYS use `search_financial_document` to look up accurate data from the filings first.\n"
        "2. NEVER calculate financial math (growth %, margins, ratios, variances) in your head! ALWAYS call the appropriate calculation tool.\n"
        "3. Include source page citations (e.g., [Page 1]) in your final response.\n"
        "4. If a calculation is requested, show the formula, the exact numbers used, and the tool-calculated result.\n"
        "5. Be concise, objective, and institutional-grade in your final output."
    )

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=ALL_AGENT_TOOLS,
            temperature=0.0
        ),
    )

    return response.text

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
