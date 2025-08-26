"""
Handles the requests made by the user and sends a response
"""

import uuid
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional, Dict, Any
from datetime import datetime
import asyncio
import uvicorn
from requests import Request
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from answer_questions.db_searching import search_docs

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """
    Function that defines what happens from the moment 
    the app is initialized to the moment it gets shut down
    
    Args:
        _ (FastAPI): throwaway variable because it 
            is required, used when decisions are made 
            based on the app information

    Returns:
        (AsyncGenerator[None, None])
    """
    logging.info("HealthLLM Q&A system starting up")

    yield

    logging.info("HealthLLM Q&A system shutting down...")

class QuestionRequest(BaseModel):
    """
    Verifies the information input by the user
    extending the Pydantic BaseModel class 
    """
    question: str = Field(..., min_length=1, max_length=1000, description="The question to answer")
    context_limit: Optional[int] = Field(5,
                                         ge=1,
                                         le=20,
                                         description="Maximum number of documents to retrieve")
    include_sources: Optional[bool] = Field(True,
                                            description="Whether to include source information")

    class Config: # pylint: disable=too-few-public-methods
        """
        Creates an example on how the output should look like, in order to 
        guide the previous requests
        """
        json_schema_extra = {
            "example": {
                "question": "What is the difference between diploid and haploid nuclei in meiosis?",
                "context_limit": 5,
                "include_sources": True
            }
        }

class DocumentSource(BaseModel):
    """
    Verifies if the information from the document used as 
    source is correctly displayed, extending Pydantic 
    BaseModel class
    """
    pmid: str
    title: Optional[str] = None
    text: str
    similarity_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class AnswerResponse(BaseModel):
    """
    Verifies if the information in the answer is 
    correctly displayed, extending Pydantic 
    BaseModel class
    """
    answer: str
    confidence_score: Optional[float] = None
    sources: Optional[List[DocumentSource]] = None
    processing_time: float
    request_id: str
    timestamp: datetime

class HealthResponse(BaseModel):
    """
    Verifies if the information in the health 
    check is correctly displayed, extending 
    Pydantic BaseModel class
    """
    status: str
    timestamp: datetime
    version: str

app = FastAPI(
    title="HealthLLM",
    description="Answers biomedical questions based on document retrieval",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentRetriever:
    """
    Retriever that gets docs relevant to the query based on the
    from the Qdrant database
    """

    async def retrieve_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the documents by performing cosine similarity search 
        in the Qdrant vector database

        Args:
            self (DocumentRetriever): the DocumentRetriever 
                document itself
            query (str): the input query
            limit (int): maximum number of documents to 
                retrieve

        Returns:
            (List[Dict[str, Any]]): document metadata
        """
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, search_docs, query, limit)

        results = []
        for i, doc in enumerate(docs):
            results.append({
                "pmid": doc.metadata.get("pmid", f"doc_{i}"),
                "title": doc.metadata.get("title", "N/A"),
                "text": doc.page_content,
                "similarity_score": doc.metadata.get("similarity_score"),
                "metadata": doc.metadata
            })

        return results

    async def generate_answer(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Mock answer generation: the answer generation function 
        does not exist yet
        
        Args:
            self (DocumentRetriever): the DocumentRetriever 
                document itself
            query (str): the input query
            documents: retrieved documents
            
        Returns:
            (Dict[str, Any]): dictionary containing answer 
                and confidence score
        """
        context = " ".join([doc["text"] for doc in documents])
        answer = f"Based on the retrieved documents, here's an answer to '{query}': {context}"
        avg_confidence = None
        if documents:
            scores = [doc.get("similarity_score", 0.0) \
                      for doc in documents if doc.get("similarity_score") is not None]
            avg_confidence = sum(scores)/len(scores) if scores else None

        return {
            "answer": answer,
            "confidence_score": avg_confidence
        }

retriever = DocumentRetriever()

@app.get("/", response_model=HealthResponse)
async def root() -> None:
    """
    Root endpoint for basic health check

    Args:
        None
    
    Returns:
        None
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check() -> None:
    """
    More detailed health check endpoint
    
    Args:
        None
    
    Returns:
        None
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest, background_tasks: BackgroundTasks):
    """
    Main endpoint for asking questions and getting answers based on the fetched 
    documents. Gets the relevant articles and generates a natural answer
    
    Args:
        request (QuestionRequest): the question request containing the 
            question and optional parameters, formatted according to the 
            previously defined QuestionRequest class
        background_tasks (BackgroundTasks): async FastAPI background tasks 
        
    Returns:
        (AnswerResponse): formated response, as specified in AnswerResponse, 
            containing the answer, sources, and metadata
    """
    start_time = asyncio.get_event_loop().time()
    request_id = str(uuid.uuid4())

    logging.info("Processing question request %s: %s", request_id, request.question)

    try:
        retrieved_docs = await retriever.retrieve_documents(
            query=request.question,
            limit=request.context_limit
        )

        if not retrieved_docs:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found for the given question"
            )

        answer_result = await retriever.generate_answer(
            query=request.question,
            documents=retrieved_docs
        )

        sources = []
        if request.include_sources:
            sources = [
                DocumentSource(
                    pmid=doc["pmid"],
                    title=doc.get("title"),
                    text=doc["text"],
                    similarity_score=doc.get("similarity_score"),
                    metadata=doc.get("metadata")
                )
                for doc in retrieved_docs
            ]

        processing_time = asyncio.get_event_loop().time() - start_time

        background_tasks.add_task(
            log_request_completion,
            request_id=request_id,
            question=request.question,
            processing_time=processing_time,
            num_sources=len(retrieved_docs)
        )

        return AnswerResponse(
            answer=answer_result["answer"],
            confidence_score=answer_result.get("confidence_score"),
            sources=sources if request.include_sources else None,
            processing_time=processing_time,
            request_id=request_id,
            timestamp=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error("Error processing request %s: %s", request_id, str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e

@app.get("/stats")
async def get_stats() -> None:
    """
    Optional endpoint to get system statistics

    Args:
        None
    
    Returns:
        None
    """
    return {
        "status": "operational",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Root health check"},
            {"path": "/health", "method": "GET", "description": "Detailed health check"},
            {"path": "/ask", "method": "POST", "description": "Ask a question"},
            {"path": "/docs", "method": "GET", "description": "API documentation"},
        ],
        "timestamp": datetime.now()
    }

async def log_request_completion(request_id: str,
                                 question: str,
                                 processing_time: float,
                                 num_sources: int):
    """
    Background task for logging request completion, including information that
    allows the assessment of some bottlenecks
    """
    logging.info(
        "Request %s completed - "
        "Question: '%s' - "
        "Processing time: %.2fs - "
        "Sources: %d",
        request_id,
        question,
        processing_time,
        num_sources
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: Exception) -> Exception:
    """
    HTTP exceptions appear as warnings in logging

    Args:
        _ (Request): throwaway argument
        exc (Exception): catched exception

    Returns:
        exc (Exception): catched exception
    """
    logging.warning("HTTP %d error: %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(_: Request, exc: Exception) -> None:
    """
    General exception handler for unexpected errors, 
    that are not in HTTP, but are raised as such

    Args:
        _ (Request): throwaway argument
        exc (Exception): catched exception

    Returns:
        None
    """
    logging.error("Unexpected error: %s", str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "requests_handler:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
