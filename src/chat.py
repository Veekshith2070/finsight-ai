from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY
from src.logger import logger

def run_chat_session():
    client = genai.Client(api_key=GEMINI_API_KEY)

    system_instruction = (
        "You are FinSight AI, an intelligent finanacial research assistant."
        "Provde clear, professional answers to financial questions."

    )

    # 1. Conversation State: An in-memory list tracking the entire conversation
    conversation_history = []

    print("\n==========================================")
    print(" FinSight AI - Interactive Termainal Chat")
    print(" (Types 'exit' or 'quit' to end session) ")
    print("============================================")

    while True:
        user_input = input("User: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ["exit","quit"]:
            logger.info("Ending chat session.")
            break

        # 2. Append the new user message to history
        conversation_history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)]
            )
        )

        try:
            # 3. Send the ENTIRE history to the model
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=conversation_history,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0,
                ),
            )

            # Log token growth
            if response.usage_metadata:
                logger.info(
                    f"[Tokens] Input: {response.usage_metadata.prompt_token_count} | "
                    f"Output: {response.usage_metadata.candidates_token_count}"
                )

            model_reply = response.text
            print(f"\nFinSight: {model_reply}\n")

            # 4. Append the model's reply to the history so next turn has full context
            conversation_history.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=model_reply)]
                )
            )

        except Exception as e:
            logger.error(f"Chat turn failed: {e}")
            print(f"Error: {e}\n")

if __name__ == "__main__":
    run_chat_session()