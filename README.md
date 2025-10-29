# RAG CRM - Customer Service Assistant

Building a RAG (Retrieval-Augmented Generation) application for CRM Customer Service using Python, FastAPI, MongoDB Atlas Vector Search, and OpenAI.

## Overview

This application helps customer support teams get fast, accurate answers from product docs, past tickets, and knowledge bases using Retrieval-Augmented Generation (RAG). The system stores document embeddings in MongoDB Atlas and retrieves relevant context to generate intelligent responses to customer queries.

## Features

- **Vector Search**: Uses MongoDB Atlas Vector Search for semantic document retrieval
- **OpenAI Integration**: Employs OpenAI's embedding and chat models for text understanding and generation
- **FastAPI Backend**: RESTful API with automatic documentation
- **Smart Chunking**: Sentence-aware text chunking with overlap for better context
- **Metadata Support**: Rich metadata tracking for documents and sources
- **Error Handling**: Comprehensive error handling and logging
- **Health Monitoring**: Built-in health check and statistics endpoints

## Architecture

```
┌─────────────────┐
│   FastAPI App   │
│   (app.py)      │
└────────┬────────┘
         │
         ├───────→ OpenAI (Embeddings + Chat)
         │
         └───────→ MongoDB Atlas Vector Search
                      └────→ Store & Retrieve
```

### Data Flow

1. **Ingestion**: Documents are chunked, embedded, and stored in MongoDB
2. **Query**: User question is embedded and searched against the vector index
3. **Retrieval**: Most relevant chunks are retrieved (KNN search)
4. **Generation**: LLM generates answer using retrieved context

## Setup

### Prerequisites

- Python 3.8+
- MongoDB Atlas account with Vector Search enabled
- OpenAI API key

### Installation

1. **Clone the repository** (if applicable)

```bash
cd RAGCRM
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
DB_NAME=crm_rag
COLLECTION_NAME=kb_chunks
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4
INDEX_NAME=vector_index
CHUNK_SIZE=800
CHUNK_OVERLAP=100
```

### MongoDB Atlas Setup

1. **Create a Database**
   - Log in to MongoDB Atlas
   - Create a new cluster (free tier works)
   - Create a database named `crm_rag`

2. **Create a Collection**
   - Create collection: `kb_chunks`
   - The following schema is used:
     ```
     {
       doc_id: string,
       chunk_id: integer,
       text: string,
       metadata: object,
       embedding: array[float]
     }
     ```

3. **Create Vector Search Index**
   - Go to Atlas Search
   - Click "Create Index"
   - Choose JSON Editor
   - Use this configuration:
     ```json
     {
       "name": "vector_index",
       "type": "vectorSearch",
       "definition": {
         "fields": [
           {
             "type": "knnVector",
             "path": "embedding",
             "numDimensions": 1536,
             "similarity": "cosine"
           }
         ]
       }
     }
     ```
   - Note: Adjust `numDimensions` based on your embedding model (1536 for text-embedding-3-small)

## Usage

### 1. Ingest Documents

Run the ingestion script to populate your knowledge base:

```bash
python ingest.py
```

This will:
- Process example documents (KB articles, support tickets)
- Chunk text intelligently
- Generate embeddings via OpenAI
- Store in MongoDB with metadata

**Custom Ingestion:**

You can modify `ingest.py` to ingest your own documents:

```python
from ingest import ingest_document

ingest_document(
    doc_id="your_doc_id",
    full_text="Your document text here...",
    metadata={
        "source": "knowledge_base",
        "title": "Your Title",
        "category": "account",
        "type": "article"
    }
)
```

### 2. Start the API Server

```bash
python app.py
```

Or with uvicorn:

```bash
uvicorn app:app --reload --port 8000
```

The API will be available at: `http://localhost:8000`

### 3. API Documentation

Access interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Health Check
```bash
GET /health
```

Returns system health status.

### Statistics
```bash
GET /stats
```

Returns collection statistics.

### Query (Main Endpoint)
```bash
POST /query
```

**Request Body:**
```json
{
  "user_id": "user123",
  "query": "How do I reset my password?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "To reset your password...",
  "retrieved_sources": [
    {
      "source": "knowledge_base",
      "text": "How to Reset Your Password...",
      "score": 0.95
    }
  ],
  "query": "How do I reset my password?"
}
```

### Example cURL Request

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "query": "How do I reset my password?",
    "top_k": 5
  }'
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `MONGODB_URI` | MongoDB connection string | Required |
| `DB_NAME` | Database name | `crm_rag` |
| `COLLECTION_NAME` | Collection name | `kb_chunks` |
| `EMBED_MODEL` | Embedding model | `text-embedding-3-small` |
| `CHAT_MODEL` | Chat model | `gpt-4` |
| `INDEX_NAME` | Atlas index name | `vector_index` |
| `CHUNK_SIZE` | Words per chunk | `800` |
| `CHUNK_OVERLAP` | Overlap words | `100` |

## Project Structure

```
RAGCRM/
├── app.py              # FastAPI application
├── ingest.py           # Document ingestion script
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .env               # Environment variables (create this)
```

## How It Works

### Text Chunking Strategy

The system uses intelligent sentence-based chunking:
- Splits text by sentences first
- Creates chunks of ~800 words
- Maintains 100-word overlap between chunks
- Preserves semantic coherence

### RAG Pipeline

1. **Query Embedding**: Convert user question to vector
2. **Vector Search**: Find top-k most similar documents using cosine similarity
3. **Context Assembly**: Combine retrieved chunks with metadata
4. **LLM Generation**: Generate answer using context
5. **Response**: Return answer with source citations

### System Prompt

The LLM is instructed to:
- Answer accurately based on provided context
- Admit when information is unavailable
- Never hallucinate information
- Be professional and concise
- Include source information

## Best Practices

### Ingestion
- Use meaningful `doc_id` values
- Include rich metadata (source, category, title, etc.)
- Chunk appropriately for your document types
- Handle special characters and encoding

### Querying
- Adjust `top_k` based on your use case (3-10 typical)
- Monitor query performance
- Consider caching frequent queries
- Implement rate limiting for production

### Security
- Keep `.env` file secure (add to `.gitignore`)
- Use MongoDB IP whitelist
- Implement API authentication for production
- Monitor API usage and costs

## Troubleshooting

### Common Issues

**Issue**: MongoDB connection failed
- Check `MONGODB_URI` is correct
- Verify network access to MongoDB Atlas
- Check IP whitelist in Atlas

**Issue**: Embedding dimension mismatch
- Ensure embedding model matches index dimension
- text-embedding-3-small: 1536 dimensions
- Update `numDimensions` in Atlas index if needed

**Issue**: No results from vector search
- Verify index is created and active
- Check that documents have `embedding` field
- Ensure index name matches `INDEX_NAME` env var

**Issue**: OpenAI API errors
- Verify API key is valid
- Check API quota and billing
- Handle rate limits appropriately

## Future Enhancements

- [ ] Support for multiple file formats (PDF, DOCX, etc.)
- [ ] Hybrid search (keyword + vector)
- [ ] Conversation history and memory
- [ ] User feedback and learning
- [ ] Multi-language support
- [ ] Analytics dashboard
- [ ] API authentication
- [ ] Docker deployment

## License

MIT

## Contributing

Contributions welcome! Please submit issues and pull requests.

## Support

For issues or questions, please open a GitHub issue.

---

**Built with**: Python • FastAPI • MongoDB Atlas • OpenAI
