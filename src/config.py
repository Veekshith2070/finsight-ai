import os
from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.getenv("APP_ENV")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not APP_ENV:
    raise ValueError("Configration Error: 'APP_ENV' environment variable is not set.")
if not GEMINI_API_KEY:
    raise ValueError("Configuation Error: 'GEMINI_API_KEY' environment variable is not set.")


if __name__ == "__main__":
    print(f"FinSight AI initialized in [{APP_ENV}] mode.")