from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY
from src.logger import logger

def get_gemini_client() -> genai.Client:
    """Initializes and returns the authenticated Gemini client."""
    return genai.Client(api_key=GEMINI_API_KEY)

def generate_financial_response(
        prompt: str,
        system_instruction: str = "You are FinSight AI, a precise financial analyst assistant. Keep answers factual, concise, and focused on finanacial principles.",
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0
) -> str:
    """
    Sends a generation request to the Gemini API with structured parameters and error handling.
    """
    client = get_gemini_client()
    logger.info(f"Sending request to model: [{model_name}] (temperature={temperature})")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
            ),
        )

        # Extract token usage metadata
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            candidate_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count
            logger.info(f"Tokens consumed - Input: {prompt_tokens}, Output: {candidate_tokens}, Total: {total_tokens}")

            return response.text

    except Exception as e:
        logger.error(f"LLM API request failed: {e}")
        raise e
if __name__ == "__main__":
    # Test finanacial question

    test_query = "Can you give me an example with numbers for the first one?"

    print("\n---Sending Query ---")
    print(f"User: {test_query}\n")

    answer = generate_financial_response(test_query)

    print("---Model Response ---")
    print(answer)