from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY
from src.tools import FINANCIAL_TOOLS
from src.logger import logger

def run_financial_calculation_query(
    user_query: str,
    model_name: str = "gemini-3.5-flash"
) -> str:
    """
    Sends a calculation prompt to Gemini with our registered Python tools.
    Gemini automatically selects and executes the appropriate tool deterministically.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    logger.info(f"Processing calculation query: '{user_query}'")
    
    # We initialize a chat session configured with our financial tools
    chat = client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are FinSight AI, a financial computation engine. "
                "Whenever a calculation is requested (growth rate, margins, ratios), "
                "you MUST delegate the math to the available tools. "
                "Never calculate arithmetic in your head. State the exact numbers returned by the tools."
            ),
            tools=FINANCIAL_TOOLS, # Automatic Python Function Calling enabled!
            temperature=0.0,
        ),
    )
    
    response = chat.send_message(user_query)
    return response.text

if __name__ == "__main__":
    print("\n=======================================================")
    print("      FinSight AI - Financial Tool Calling Demo       ")
    print("=======================================================\n")
    
    # Query 1: Percentage Growth
    q1 = "Calculate the percentage growth of Apple Services revenue from $27,423 million to $30,739 million."
    print(f"Query 1: {q1}")
    ans1 = run_financial_calculation_query(q1)
    print(f"Answer: {ans1}\n")
    print("-" * 55)
    
    # Query 2: Operating Margin
    q2 = "What is Apple's Operating Margin if Operating Income is $35,695 million and Net Sales is $109,417 million?"
    print(f"Query 2: {q2}")
    ans2 = run_financial_calculation_query(q2)
    print(f"Answer: {ans2}\n")
    print("-" * 55)
    
    # Query 3: Debt-to-Equity Ratio
    q3 = "Calculate the Debt-to-Equity ratio if Total Debt is $101,234 million and Total Equity is $62,100 million."
    print(f"Query 3: {q3}")
    ans3 = run_financial_calculation_query(q3)
    print(f"Answer: {ans3}\n")