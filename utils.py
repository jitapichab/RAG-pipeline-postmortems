"""
Shared Utilities
================

Common functions extracted to avoid duplication.
"""

import certifi
from pymongo import MongoClient

from config import MONGO_DB_URL, DB_NAME, COLLECTION_NAME


def get_mongo_client() -> MongoClient:
    """Create MongoDB client with TLS support for Atlas."""
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL not set in environment")
    
    if "mongodb+srv" in MONGO_DB_URL or "mongodb.net" in MONGO_DB_URL:
        return MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    return MongoClient(MONGO_DB_URL)


def get_collection():
    """Get the MongoDB collection."""
    client = get_mongo_client()
    return client[DB_NAME][COLLECTION_NAME]
