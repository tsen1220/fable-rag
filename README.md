# Fable RAG System

A fable story retrieval system using vector database and semantic search technology.

## Overview

This project implements a RAG (Retrieval-Augmented Generation) based fable search system that finds relevant fables based on semantic similarity.

### Key Features

- Semantic story search
- Vectorized text storage
- RESTful API interface
- Multilingual embedding model support

### Tech Stack

- **Backend Framework**: FastAPI
- **Vector Database**: Qdrant
- **Embedding Model**: Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Python Package Manager**: UV
- **Containerization**: Docker Compose

## Quick Start

### Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) (Python package manager)
- Docker & Docker Compose

### 1. Clone the Repository

```bash
git clone <repository-url>
cd story-teller-rag
```

### 2. Configure Environment Variables

Copy the example environment file and adjust as needed:

```bash
cp backend/.env.example backend/.env
```

### 3. Start Vector Database

Start Qdrant vector database using Docker Compose:

```bash
docker compose up qdrant -d
```

### 4. Initialize Database

Run the database initialization script using UV:

```bash
uv run --directory backend python -m src.init_database
```

This step will:
- Load fable story data
- Generate text embeddings
- Insert data into Qdrant database

### 5. Start API Service

#### Option A: Using Docker Compose (Recommended)

```bash
docker compose up -d
```

The API will be available at `http://localhost:8000`.

#### Option B: Using UV (Local Development)

```bash
uv run --directory backend uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Test the API

Visit the following URLs to view API documentation:

- Swagger UI: http://localhost:8000/docs

Or test using curl:

```bash
# Health check
curl http://localhost:8000/health

# Search stories
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "a story about honesty", "limit": 3}'
```

## API Endpoints

### GET /health
Check system health status

### POST /search
Search for similar fables

**Request Example**:
```json
{
  "query": "a story about honesty and lying",
  "limit": 5,
  "score_threshold": 0.7
}
```

**Response Example**:
```json
{
  "query": "a story about honesty and lying",
  "results": [
    {
      "id": 1,
      "title": "The Boy Who Cried Wolf",
      "content": "...",
      "moral": "...",
      "score": 0.85,
      "language": "en",
      "word_count": 150
    }
  ],
  "total_results": 1
}
```

### GET /fables/{fable_id}
Get a specific fable by ID

## Development Guide

### Environment Variables

```env
# Qdrant Database Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=fables

# Embedding Model
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# Data Paths
RAW_DATA_PATH=data/aesop_fables_raw.json
DATA_PATH=data/aesop_fables_processed.json

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### Install Dependencies

Install project dependencies using UV:

```bash
uv pip install --directory backend -r backend/requirements.txt
```

### Stop Services

```bash
# Stop all containers
docker compose down

# Stop and remove volumes
docker compose down -v
```

## Project Structure

```
story-teller-rag/
├── backend/
│   ├── src/
│   │   ├── main.py              # FastAPI application
│   │   ├── init_database.py     # Database initialization script
│   │   ├── embeddings.py        # Embedding model wrapper
│   │   ├── qdrant_manager.py    # Qdrant manager
│   │   └── data_processor.py    # Data processing utilities
│   ├── data/                    # Data directory
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile
│   └── .env.example
├── docker-compose.yml           # Docker Compose configuration
└── README.md
```

## Troubleshooting

### Qdrant Connection Failed

Check if the Qdrant container is running:

```bash
docker compose ps
```

If not running, start it:

```bash
docker compose up qdrant -d
```

### Collection Does Not Exist

If the API reports that the collection doesn't exist, run the initialization script:

```bash
uv run --directory backend python -m src.init_database
```

### Permission Error

Ensure the `qdrant_storage` directory has the correct read/write permissions.

## License

MIT
