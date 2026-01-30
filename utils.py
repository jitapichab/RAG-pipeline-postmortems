"""
Postmortem RAG - Utilities Module
==================================

Shared utility functions used across the RAG pipeline.
Centralizes common operations to avoid code duplication.

Author: Jorge Tapicha
Date: 2026-01-28
"""

import certifi
from pymongo import MongoClient
from langchain_core.documents import Document

from config import MONGO_DB_URL, DB_NAME, COLLECTION_NAME


# =============================================================================
# DATABASE UTILITIES
# =============================================================================

def get_mongo_client() -> MongoClient:
    """
    Create a MongoDB client connection.
    
    Handles TLS certificates for MongoDB Atlas (cloud) connections.
    """
    if not MONGO_DB_URL:
        raise ValueError(
            "MONGO_DB_URL not set. "
            "Add it to your .env file or set the environment variable."
        )
    
    if "mongodb+srv" in MONGO_DB_URL or "mongodb.net" in MONGO_DB_URL:
        return MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    else:
        return MongoClient(MONGO_DB_URL)


def get_collection():
    """Get the MongoDB collection for postmortem documents."""
    client = get_mongo_client()
    db = client[DB_NAME]
    return db[COLLECTION_NAME]


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

def format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a context string for the LLM."""
    if not documents:
        return "No relevant documents found."
    
    formatted_parts = []
    
    for i, doc in enumerate(documents, 1):
        metadata = doc.metadata
        meta_lines = [
            f"--- Document {i} ---",
            f"Incident: {metadata.get('incident_id', 'Unknown')}",
            f"Title: {metadata.get('title', 'Unknown')}",
            f"Severity: {metadata.get('severity', 'Unknown')}",
            f"Date: {metadata.get('date', 'Unknown')}",
            f"Root Cause Category: {metadata.get('root_cause_category', 'Unknown')}",
            f"Services Affected: {metadata.get('services_affected', [])}",
            f"\nContent:\n{doc.page_content}",
        ]
        formatted_parts.append("\n".join(meta_lines))
    
    return "\n\n".join(formatted_parts)


def extract_incident_ids(documents: list[Document]) -> list[str]:
    """Extract unique incident IDs from documents."""
    return list(set([
        doc.metadata.get("incident_id", "Unknown")
        for doc in documents
    ]))


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def check_mongodb_connection() -> dict:
    """Test MongoDB connection and return status."""
    try:
        client = get_mongo_client()
        client.admin.command("ping")
        collection = get_collection()
        doc_count = collection.count_documents({})
        
        return {
            "connected": True,
            "database": DB_NAME,
            "collection": COLLECTION_NAME,
            "document_count": doc_count,
            "error": None
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e)
        }


if __name__ == "__main__":
    print("Testing MongoDB connection...")
    status = check_mongodb_connection()
    if status["connected"]:
        print(f"✅ Connected - {status['document_count']} documents")
    else:
        print(f"❌ Failed: {status['error']}")
