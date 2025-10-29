# ingest.py
"""
Document ingestion script for RAG CRM.
This script processes documents, chunks them, generates embeddings, and stores them in MongoDB Atlas.
"""

import os
import re
from typing import List, Dict, Any
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
DB = os.getenv("DB_NAME", "crm_rag")
COL = os.getenv("COLLECTION_NAME", "kb_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Validate required environment variables
if not OPENAI_API_KEY or not MONGODB_URI:
    raise ValueError("OPENAI_API_KEY and MONGODB_URI must be set in .env file")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[DB]
collection = db[COL]

def chunk_text(text: str, max_len: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks by sentences.
    
    Args:
        text: Input text to chunk
        max_len: Maximum words per chunk
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        words = sentence.split()
        sentence_length = len(words)
        
        # If adding this sentence exceeds max_len, save current chunk and start new one
        if current_length + sentence_length > max_len and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Start new chunk with overlap from previous chunk
            words_to_overlap = min(overlap, len(current_chunk))
            current_chunk = current_chunk[-words_to_overlap:] + words
            current_length = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_length += sentence_length
    
    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks if chunks else [text]

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts using OpenAI.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    try:
        # OpenAI API supports batch embedding
        response = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def upsert_chunk(doc_id: str, chunk_id: int, text: str, metadata: Dict[str, Any], embedding: List[float]):
    """
    Insert or update a document chunk in MongoDB.
    
    Args:
        doc_id: Document identifier
        chunk_id: Chunk number within the document
        text: Chunk text content
        metadata: Additional metadata about the chunk
        embedding: Embedding vector
    """
    document = {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": text,
        "metadata": metadata,
        "embedding": embedding
    }
    
    try:
        collection.update_one(
            {"doc_id": doc_id, "chunk_id": chunk_id},
            {"$set": document},
            upsert=True
        )
    except Exception as e:
        logger.error(f"Error upserting chunk {chunk_id} of document {doc_id}: {e}")
        raise

def ingest_document(doc_id: str, full_text: str, metadata: Dict[str, Any]):
    """
    Ingest a complete document by chunking, embedding, and storing in MongoDB.
    
    Args:
        doc_id: Unique identifier for the document
        full_text: Full text content of the document
        metadata: Metadata dictionary (e.g., source, title, category)
    """
    logger.info(f"Processing document {doc_id}...")
    
    # Chunk the text
    chunks = chunk_text(full_text)
    logger.info(f"Split into {len(chunks)} chunks")
    
    if not chunks:
        logger.warning(f"No chunks generated for document {doc_id}")
        return
    
    # Generate embeddings
    try:
        embeddings = embed_texts(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return
    
    # Store chunks in MongoDB
    saved_count = 0
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        try:
            upsert_chunk(doc_id, i, chunk, metadata, emb)
            saved_count += 1
        except Exception as e:
            logger.error(f"Failed to save chunk {i}: {e}")
    
    logger.info(f"Successfully ingested {saved_count}/{len(chunks)} chunks for document {doc_id}")

def ingest_from_file(file_path: str, doc_id: str, metadata: Dict[str, Any]):
    """
    Ingest a document from a file.
    
    Args:
        file_path: Path to the text file
        doc_id: Unique identifier for the document
        metadata: Metadata dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        ingest_document(doc_id, full_text, metadata)
        logger.info(f"Successfully ingested file {file_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")

if __name__ == "__main__":
    # Example usage
    logger.info("Starting document ingestion...")
    
    # Example 1: Ingest a knowledge base article
    kb_text = """
    How to Reset Your Password
    
    If you've forgotten your password or need to reset it, follow these steps:
    
    1. Go to the login page
    2. Click on "Forgot Password" link
    3. Enter your email address
    4. Check your email for the reset link
    5. Click the link and create a new password
    6. Login with your new password
    
    If you continue to have issues, please contact our support team at support@example.com
    """
    
    ingest_document(
        doc_id="kb_article_001",
        full_text=kb_text,
        metadata={
            "source": "knowledge_base",
            "title": "How to Reset Password",
            "category": "account",
            "type": "article"
        }
    )
    
    # Example 2: Ingest a support ticket history
    ticket_text = """
    Support Ticket #12345
    
    Issue: Unable to log in to dashboard
    
    Resolution: The user had multiple browser tabs open with different sessions. 
    The issue was resolved by clearing browser cookies and logging in fresh.
    
    Steps taken:
    1. Asked user to clear browser cache and cookies
    2. User logged out completely and closed all browser windows
    3. User opened new browser window and logged in successfully
    
    Follow-up: User confirmed they can now access the dashboard normally.
    """
    
    ingest_document(
        doc_id="ticket_12345",
        full_text=ticket_text,
        metadata={
            "source": "support_tickets",
            "ticket_id": "12345",
            "category": "login",
            "type": "resolution"
        }
    )
    
    logger.info("Document ingestion complete!")
