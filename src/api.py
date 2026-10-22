import time
import shutil
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, UploadFile, File, Depends
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from src.config import APP_ENV
from src.database import get_db, init_db
from src.models import DocumentRecord, QueryAuditLog
from src.rag import answer_financial_query
from src.vector_store import ingest_pdf_document
from src.agent import run_financial_agent
from src.logger import logger

# 1. Initialize FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure tables are initialized when the server boots
    init_db()
    yield

app = FastAPI(
    title="FinSight AI API",
    description="Autonomous Financial Research & Quantitative Valuation Backend",
    version="1.0.0",
    lifespan=lifespan
)

# 2. Define Pydantic Request & Response Schemas
class DocumentItem(BaseModel):
    id: int
    filename: str
    total_pages: int
    total_chunks: int
    file_size_bytes: Optional[int] = None
    uploaded_at: datetime

    model_config = ConfigDict(from_attributes=True)

class AuditLogItem(BaseModel):
    id: int
    endpoint: str
    query_text: str
    response_text: str
    sources_count: int
    execution_time_ms: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class DocumentUploadResponse(BaseModel):
    filename: str
    total_pages: int
    total_chunks: int
    status: str

class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str

class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The financial question to answer from the report.")
    top_k: int = Field(default=2, ge=1, le=5, description="Number of document chuks to retrieve.")

class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]

class AgentQueryRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Multi-step research query requiring RAG and/or calculations.")

class AgentQueryResponse(BaseModel):
    prompt: str
    answer: str

# 3. Define REST Endpoints

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint"
)
def health_check():
    """Returns the operational status of the FinSight AI API."""
    return{
        "status": "healthy",
        "environment": APP_ENV,
        "version": "1.0.0"
    }

@app.post(
    "/api/rag",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Financial Intelligence"],
    summary="Direct Grounded RAG Query"
)
def query_rag_endpoint(request: RAGQueryRequest, db: Session = Depends(get_db)):
    """
    Retrieves relevant document excerpts from ChromaDB and genrates
    a gorunded answer with page citations.
    """
    logger.info(f"API [POSR /api/rag] received question: '{request.question}'")
    start_time = time.time()
    try:
        result = answer_financial_query(question=request.question, top_k=request.top_k)
        elapsed_ms = round((time.time() - start_time)*1000, 2)

        # Log to SQL audit trail
        audit = QueryAuditLog(
            endpoint="/api/rag",
            query_text=request.question,
            response_text=result["answer"],
            sources_count=len(result.get("sources", [])),
            execution_time_ms=elapsed_ms
        )
        db.add(audit)
        db.commit()

        return result
    except Exception as e:
        logger.error(f"API RAG query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process RAG query: {str(e)}"
        )

@app.post(
    "/api/agent",
    response_model=AgentQueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Financial Intelligence"],
    summary="Autonomous ReAct Agent Query"
)
def query_agent_endpoint(request: AgentQueryRequest, db: Session = Depends(get_db)):
    """
    Runs the autonomous ReAct agent loop: dynamically searches documents,
    extracts figures, and chains deterministic math calculation.
    """
    logger.info(f"API [POST /api/agent] received prompt: '{request.prompt}'")
    start_time = time.time()
    try:
        answer = run_financial_agent(request.prompt)
        elapsed_ms = round((time.time() - start_time)*1000, 2)

        # Log to SQL audit trail
        audit = QueryAuditLog(
            endpoint="/api/agent",
            query_text=request.prompt,
            response_text=answer,
            sources_count=0,
            execution_time_ms=elapsed_ms
        )
        db.add(audit)
        db.commit()

        return {
            "prompt": request.prompt,
            "answer": answer
        }
    except Exception as e:
        logger.error(f"API execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}"
        )

@app.post(
    "/api/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Document Ingestion"],
    summary="Upload and index an SEC financial PDF report"
)
def upload_pdf_endpoint(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Uploads an SEC financial report (PDF), extracts tables & text,
    cleans accounting formatting, chunks by page, and indexes into ChromaDB.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files (.pdf) are supported."
        )

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / file.filename

    logger.info(f"API [POST /api/upload] receiving file: '{file.filename}'")
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = ingest_pdf_document(destination)
         # Persist in SQL catalog
        doc = db.query(DocumentRecord).filter(DocumentRecord.filename == file.filename).first()
        if not doc:
            doc = DocumentRecord(
                filename=file.filename,
                total_pages=result["total_pages"],
                total_chunks=result["total_chunks"],
                file_size_bytes=destination.stat().st_size
            )
            db.add(doc)
        else:
            doc.total_pages = result["total_pages"]
            doc.total_chunks = result["total_chunks"]
            doc.file_size_bytes = destination.stat().st_size
        db.commit()

        return result
    except Exception as e:
        logger.error(f"API PDF ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )

@app.get(
    "/api/documents",
    response_model=List[DocumentItem],
    tags=["Document Ingestion"],
    summary="List all indexed financial documents"
)
def list_documents_endpoint(db: Session = Depends(get_db)):
    """List all ingested documents stored in the database catalog."""
    return db.query(DocumentRecord).order_by(DocumentRecord.uploaded_at.desc()).all()

@app.get(
    "/api/audit-logs",
    response_model=List[AuditLogItem],
    tags=["Compliance & Auditing"],
    summary="Retrieve query execution autdit trail"
)
def get_audit_logs_endpoint(limit: int = 20, db: Session = Depends(get_db)):
    """
    Returns the most recent query audit logs for regulatory compliance,
    provenance verification, and latency monitoring.
    """
    return db.query(QueryAuditLog).order_by(QueryAuditLog.created_at.desc()).limit(limit).all()

if __name__=="__main__":
    import uvicorn
    logger.info("Starting local Uvicorn development server...")
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)