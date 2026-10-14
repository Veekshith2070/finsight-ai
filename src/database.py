from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from src.config import DATABASE_URL
from src.logger import logger

# Ensure the data directory exists for SQLite
if DATABASE_URL.startswith("sqlite"):
    Path("data").mkdir(exist_ok=True)
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
    FastAPI dependency that yields a database session per request
    and guarantees it closes after the response is sent.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Creates all database tables defined in models."""
    logger.info("Initializing relational database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")
