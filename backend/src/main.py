"""FastAPI application: Fable RAG System API"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv

from src.embeddings import EmbeddingModel
from src.qdrant_manager import QdrantManager

# Load environment variables
load_dotenv()

# Create FastAPI application
app = FastAPI(
    title="Fable RAG API",
    description="Fable Story Retrieval-Augmented Generation System API",
    version="1.0.0"
)

# Setup CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables: model and database connection
embedding_model: Optional[EmbeddingModel] = None
qdrant_manager: Optional[QdrantManager] = None

# Configuration
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "fables")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


# Pydantic models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    limit: int = Field(5, ge=1, le=20, description="Number of results to return (1-20)")
    score_threshold: Optional[float] = Field(None, ge=0, le=1, description="Similarity score threshold (0-1)")


class FableResult(BaseModel):
    id: int
    title: str
    content: str
    moral: str
    score: float
    language: str
    word_count: int


class SearchResponse(BaseModel):
    query: str
    results: List[FableResult]
    total_results: int


class HealthResponse(BaseModel):
    status: str
    message: str
    collection_name: str
    total_fables: int


# Lifecycle events
@app.on_event("startup")
async def startup_event():
    """Initialize model and connections on application startup"""
    global embedding_model, qdrant_manager

    print("🚀 Initializing Fable RAG System...")

    # Initialize embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = EmbeddingModel()

    # Connect to Qdrant
    print(f"Connecting to Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    qdrant_manager = QdrantManager()

    # Check if collection exists
    info = qdrant_manager.get_collection_info(COLLECTION_NAME)
    if info:
        print(f"✓ Collection '{COLLECTION_NAME}' ready with {info['points_count']} fables")
    else:
        print(f"⚠ Collection '{COLLECTION_NAME}' does not exist, please run init_database.py first")

    print("✓ System startup complete!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    print("👋 Shutting down system...")


# API endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Fable RAG API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if embedding_model is None or qdrant_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized yet")

    try:
        info = qdrant_manager.get_collection_info(COLLECTION_NAME)
        if info is None:
            raise HTTPException(
                status_code=503,
                detail=f"Collection '{COLLECTION_NAME}' does not exist, please run init_database.py first"
            )

        return HealthResponse(
            status="healthy",
            message="System running normally",
            collection_name=COLLECTION_NAME,
            total_fables=info['points_count']
        )

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_fables(request: SearchRequest):
    """
    Search for similar fables

    - **query**: Search query text (e.g., "a story about honesty")
    - **limit**: Number of results to return (1-20, default 5)
    - **score_threshold**: Similarity score threshold (0-1, optional)
    """
    if embedding_model is None or qdrant_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized yet")

    try:
        # Vectorize query text
        query_vector = embedding_model.encode_single(request.query)

        # Search for similar vectors
        results = qdrant_manager.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=request.limit,
            score_threshold=request.score_threshold
        )

        # Format results
        fable_results = [
            FableResult(
                id=result['id'],
                title=result['payload']['title'],
                content=result['payload']['content'],
                moral=result['payload']['moral'],
                score=result['score'],
                language=result['payload']['language'],
                word_count=result['payload']['word_count']
            )
            for result in results
        ]

        return SearchResponse(
            query=request.query,
            results=fable_results,
            total_results=len(fable_results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/fables/{fable_id}", tags=["Fables"])
async def get_fable_by_id(fable_id: int):
    """Get specific fable by ID"""
    if qdrant_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized yet")

    try:
        # Use Qdrant's retrieve function
        result = qdrant_manager.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[fable_id]
        )

        if not result:
            raise HTTPException(status_code=404, detail=f"Fable with ID {fable_id} not found")

        point = result[0]
        return {
            "id": point.id,
            "title": point.payload['title'],
            "content": point.payload['content'],
            "moral": point.payload['moral'],
            "language": point.payload['language'],
            "word_count": point.payload['word_count']
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get fable: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )
