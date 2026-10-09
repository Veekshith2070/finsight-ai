import os
from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.getenv("APP_ENV")

if not APP_ENV:
    raise ValueError("Configration Error: 'APP_ENV' environment variable is not set.")


if __name__ == "__main__":
    print(f"FinSight AI initialized in [{APP_ENV}] mode.")