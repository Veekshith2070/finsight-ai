from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from src.database import Base

class DocumentRecord(Base):
    """
    Catalog of ingested financial reports and SEC filings.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), unique=True, nullable=False, index=True)
    total_pages = Column(Integer, nullable=False)
    total_chunks = Column(Integer, nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class QueryAuditLog(Base):
    """
    Immutable audit trail for every query, agent invocation, and calculation.
    Required for financial compliance, latency tracking, and provenance.
    """
    __tablename__ = "query_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String(50), nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text, nullable=False)
    sources_count = Column(Integer, default=0)
    execution_time_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
