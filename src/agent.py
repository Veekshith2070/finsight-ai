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
    Searches the ingested financial document (Apple 10-Q) for relevant factual context,
    tables, revenue figures, expenses, and notes.


    Args:
        query: The specific topic, keyword, or question to search for in the document.
    """
    logger.info(f"Agent Tool Innvoked: search_financial_document(query='{query}')")
    results = query_vector_store(query, top_k=2)

    if not results:
        return "No relenvnat excerpts found in the document."

    formatted = []
    for r in results:
        formatted.append(f"[Source: Page {r['page_number']}]\n{r['text']}")

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
        model_name: str = "gemini-2.5-flash"
) -> str:
    """
    Executes the autonomous agent loop using Gemini 3.5 Flash.
    The agent can dynamically decide to retrieve documents, perform math,
    or chain both together to answer multi-step questions.
    """

    client = genai.Client(api_key=GEMINI_API_KEY)

    logger.info(f"Agent received user prompt: '{user_prompt}'")

    system_instruction = (
        "You are FinSight AI, an autonomous financial research agent."
        "You have access to tools for document retrieval and deterministic financial calculations.\n"
        "Rules:\n"
        "1. For questions requiring data from the report, use `search_financial_document`.\n"
        "2. For ANY arithmetic, growth rates, margins, or ratios, you MUST usse the appropriate calculation tool. NEVER calculate math on your head.\n"
        "3. If a question requires both finding facts and calculating, FIRST search the document, THEN pass the retrieved numbers into the calculation tool.\n"
        "4. Always cite the page number for any factual data retrieved."
    )

    chat = client.chats.create(
        model = model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=ALL_AGENT_TOOLS, # Full toolkit enabled
            temperature=0.0,
        ),
    )

    response = chat.send_message(user_prompt)
    return response.text

if __name__ == "__main__":
    print("\n====================================================")
    print("       FinSight AI - Autonomous Multi-Step Agent      ")
    print("======================================================\n")

    # The ultimate multi-step test: Requires RAG + Tool execution chained together!
    complex_query = (
        "For the three months ended June 27, 2026, find Apple's Products revenue "
        "and Services revenue in the report, and calculate what percentage Services is of Products."
    )

    print(f"User Query: \n'{complex_query}'")
    print("---Agent Execution Trace ---")
    final_answer = run_financial_agent(complex_query)

    print("\n--- Final Agent Response ---")
    print(final_answer)