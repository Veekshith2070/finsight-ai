from src.config import APP_ENV
from src.logger import logger

def main() -> None:
    logger.info(f"Starting FinSight AI in [{APP_ENV}] mode...")
    logger.info("Core application initialized successfully.")

if __name__ == "__main__":
    main()