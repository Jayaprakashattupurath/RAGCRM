# app.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
DB = os.getenv("DB_NAME", "crm_rag")
COL = os.getenv("COLLECTION_NAME", "kb_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4")
INDEX_NAME = os.getenv("INDEX_NAME", "vector_index")

# Validate required environment variables
if not OPENAI_API_KEY or not MONGODB_URI:
    raise ValueError("OPENAI_API_KEY and MONGODB_URI must be set in .env file")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[DB]
collection = db[COL]

app = FastAPI(
    title="RAG CRM - Customer Service Assistant",
    description="Retrieval-Augmented Generation for Customer Support using MongoDB Atlas Vector Search",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    retrieved_sources: list
    query: str
    
def embed_query(text: str) -> list:
    """Generate embedding for the query text using OpenAI."""
    try:
        response = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")

def vector_search(query_embedding: list, top_k: int = 5) -> list:
    """Search MongoDB Atlas vector index using KNN."""
    try:
        pipeline = [
            {
                "$search": {
                    "index": INDEX_NAME,
                    "knn": {
                        "vector": query_embedding,
                        "path": "embedding",
                        "k": top_k
                    }
                }
            },
            {
                "$project": {
                    "text": 1,
                    "metadata": 1,
                    "_score": {"$meta": "searchScore"}
                }
            }
        ]
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform vector search")

def build_system_prompt() -> str:
    """Build the system prompt for the LLM."""
    return (
        "You are a helpful customer service assistant for our company. "
        "Answer customer queries using ONLY the provided context from our knowledge base. "
        "Important guidelines:\n"
        "1. Answer accurately based on the context provided\n"
        "2. If the context doesn't contain enough information to answer, say so politely\n"
        "3. Never make up or hallucinate information\n"
        "4. If the question is outside the provided context, suggest they contact support directly\n"
        "5. Be professional, friendly, and concise\n"
        "6. Include relevant source information when available"
    )

def generate_answer(query: str, context: str, hits: list) -> str:
    """Generate answer using OpenAI chat completion."""
    try:
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "user",
                "content": f"User question: {query}\n\nContext from knowledge base:\n{context}\n\nProvide a helpful answer to the user's question."
            }
        ]
        
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.0
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer")

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG CRM - Customer Service Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Query the RAG system",
            "GET /health": "Health check",
            "GET /stats": "Database statistics"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        # Check MongoDB connection
        mongo_client.admin.command('ping')
        return {
            "status": "healthy",
            "mongodb": "connected",
            "openai": "configured"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"MongoDB connection failed: {str(e)}")

@app.get("/stats")
def get_stats():
    """Get collection statistics."""
    try:
        total_docs = collection.count_documents({})
        return {
            "collection": COL,
            "database": DB,
            "total_documents": total_docs
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@app.post("/query", response_model=QueryResponse)
def query_rag(q: QueryRequest):
    """
    Query the RAG system with a customer service question.
    
    - **user_id**: Unique identifier for the user
    - **query**: The customer question
    - **top_k**: Number of relevant documents to retrieve (default: 5)
    """
    try:
        logger.info(f"Received query from user {q.user_id}: {q.query}")
        
        # Generate embedding for the query
        query_embedding = embed_query(q.query)
        
        # Search for relevant documents
        hits = vector_search(query_embedding, top_k=q.top_k)
        
        if not hits:
            return QueryResponse(
                answer="I couldn't find relevant information in our knowledge base to answer your question. Please contact our support team directly for assistance.",
                retrieved_sources=[],
                query=q.query
            )
        
        # Build context from retrieved documents
        context_parts = []
        for hit in hits:
            source_info = hit.get('metadata', {}).get('source', 'Unknown source')
            text_content = hit.get('text', '')
            context_parts.append(f"[Source: {source_info}]\n{text_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer using LLM
        answer = generate_answer(q.query, context, hits)
        
        # Format response
        retrieved_sources = [
            {
                "source": hit.get('metadata', {}).get('source', 'Unknown'),
                "text": hit.get('text', '')[:200] + "..." if len(hit.get('text', '')) > 200 else hit.get('text', ''),
                "score": hit.get('_score', 0)
            }
            for hit in hits
        ]
        
        logger.info(f"Successfully generated answer for user {q.user_id}")
        
        return QueryResponse(
            answer=answer,
            retrieved_sources=retrieved_sources,
            query=q.query
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
