from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from src.config import APP_ENV
from src.rag import answer_financial_query
from src.agent import run_financial_agent
from src.logger import logger

# 1. Initialize FastAPI Application
app = FastAPI(
    title="FinSight AI API",
    description="Autonomous Financial Research & Quantitative Valuation Backend",
    version="1.0.0",
)

# 2. Define Pydantic Request & Response Schemas
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
def query_rag_endpoint(request: RAGQueryRequest):
    """
    Retrieves relevant document excerpts from ChromaDB and genrates
    a gorunded answer with page citations.
    """
    logger.info(f"API [POSR /api/rag] received question: '{request.question}'")
    try:
        result = answer_financial_query(question=request.question, top_k=request.top_k)
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
def query_agent_endpoint(request: AgentQueryRequest):
    """
    Runs the autonomous ReAct agent loop: dynamically searches documents,
    extracts figures, and chains deterministic math calculation.
    """
    logger.info(f"API [POST /api/agent] received prompt: '{request.prompt}'")
    try:
        answer = run_financial_agent(request.prompt)
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

if __name__=="__main__":
    import uvicorn
    logger.info("Starting local Uvicorn development server...")
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)