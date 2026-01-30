"""
Postmortem RAG - Retrieval Module
==================================

This module performs vector similarity search with optional metadata filtering.
It's the core of the Metadata-Filtered RAG pattern.

ARCHITECTURE:
    User Query → Embed → [Pre-Filter by Metadata] → Vector Search → Top-K Results

KEY CONCEPT - Metadata-Filtered RAG:
    Instead of searching ALL documents, we first filter by metadata (severity,
    services, root cause, etc.) and THEN perform vector search on the filtered set.
    This improves precision and reduces noise.

Author: Jorge Tapicha.
Date: 2026-01-28
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# TLS certificates for MongoDB Atlas
import certifi

# MongoDB driver
from pymongo import MongoClient

# LangChain components
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from collections import OrderedDict
from langchain_core.documents import Document

# Load environment variables
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Environment variables
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB configuration - MUST match ingestion.py
DB_NAME = "rag_playbook"
COLLECTION_NAME = "postmortem_rag"
INDEX_NAME = "postmortem_index"

# Retrieval configuration
DEFAULT_TOP_K = 5
# How many documents to retrieve by default
# Trade-off:
#   - Smaller K = More precise, less context
#   - Larger K = More context, might include irrelevant docs
#   - For postmortems, 5 is usually enough

# =============================================================================
# AVAILABLE FILTER VALUES
# =============================================================================
# These are the values that can be used for filtering
# Based on what was extracted during ingestion

AVAILABLE_SEVERITIES = [
    "S1 - Critical Severity",
    "S2 - High Severity", 
    "S3 - Medium Severity",
    "S4 - Low Severity",
    "Unknown"
]

AVAILABLE_ROOT_CAUSE_CATEGORIES = [
    "database",      # MongoDB, Postgres, Aurora issues
    "capacity",      # Load, traffic, scaling issues
    "deployment",    # Rollout, release issues
    "configuration", # Config, settings issues
    "network",       # DNS, timeout, connection issues
    "code_bug",      # Logic errors, null pointers
    "third_party",   # External provider issues
    "unknown"        # Could not determine
]

# Note: services_affected and owner_team vary - query the DB for options


# =============================================================================
# FUNCTION: get_mongo_client
# =============================================================================
def get_mongo_client() -> MongoClient:
    """
    Create a MongoDB client connection.
    
    Same as in ingestion.py - handles Atlas TLS certificates.
    """
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    
    if MONGO_DB_URL.startswith("mongodb+srv"):
        return MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    else:
        return MongoClient(MONGO_DB_URL)


# =============================================================================
# FUNCTION: get_vector_store
# =============================================================================
def get_vector_store():
    """
    Connect to the MongoDB Atlas Vector Store.
    
    WHAT THIS RETURNS:
    - vector_store: The LangChain vector store object for searching
    - client: The MongoDB client (needed to close connection later)
    
    HOW IT WORKS:
    1. Connects to MongoDB Atlas
    2. Gets the collection with our embedded documents
    3. Wraps it in LangChain's MongoDBAtlasVectorSearch
    4. Uses same embedding model as ingestion (important!)
    
    WHY SAME EMBEDDING MODEL:
    - Query embeddings must match document embeddings
    - If you used text-embedding-3-small for ingestion,
      you MUST use it for queries too
    """
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    # Same embedding model as ingestion
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    # Create the vector store object
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    
    return vector_store, client


# =============================================================================
# FUNCTION: build_pre_filter
# =============================================================================
def build_pre_filter(
    severity: Optional[str | list[str]] = None,
    root_cause_category: Optional[str | list[str]] = None,
    services_affected: Optional[str | list[str]] = None,
    owner_team: Optional[str] = None,
    incident_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    """
    Build a MongoDB query filter for pre-filtering before vector search.
    
    THIS IS THE KEY FUNCTION FOR METADATA-FILTERED RAG!
    
    HOW PRE-FILTERING WORKS:
    1. MongoDB first applies these filters (fast, uses indexes)
    2. Then vector search runs ONLY on the filtered documents
    3. Result: Higher precision, less noise
    
    Example:
        Query: "database timeout"
        Filter: severity="S1", root_cause_category="database"
        
        Without filter: Search 400 chunks → might return capacity issues too
        With filter: Search 50 chunks (only S1 + database) → more relevant results
    
    MONGODB FILTER SYNTAX:
    - {"field": {"$eq": "value"}}     - Exact match
    - {"field": {"$in": [...]}}       - Match any in list
    - {"$and": [{...}, {...}]}        - All conditions must match
    - {"$or": [{...}, {...}]}         - Any condition must match
    
    Args:
        severity: Filter by severity level(s)
        root_cause_category: Filter by root cause category
        services_affected: Filter by affected service(s)
        owner_team: Filter by team that owns the incident
        incident_id: Filter by specific incident ID
        date_from: Filter incidents on or after this date (YYYY-MM-DD)
        date_to: Filter incidents on or before this date (YYYY-MM-DD)
    
    Returns:
        MongoDB filter dictionary (empty dict if no filters)
    """
    conditions = []
    
    # --- Severity Filter ---
    if severity is not None:
        if isinstance(severity, list):
            # Match any of the provided severities
            conditions.append({"severity": {"$in": severity}})
        else:
            # Exact match
            conditions.append({"severity": {"$eq": severity}})
    
    # --- Root Cause Category Filter ---
    if root_cause_category is not None:
        if isinstance(root_cause_category, list):
            conditions.append({"root_cause_category": {"$in": root_cause_category}})
        else:
            conditions.append({"root_cause_category": {"$eq": root_cause_category}})
    
    # --- Services Affected Filter ---
    # Note: services_affected is an ARRAY in the document
    # $in on an array field matches if ANY element is in the list
    if services_affected is not None:
        if isinstance(services_affected, str):
            services_affected = [services_affected]
        conditions.append({"services_affected": {"$in": services_affected}})
    
    # --- Owner Team Filter ---
    if owner_team is not None:
        conditions.append({"owner_team": {"$eq": owner_team}})
    
    # --- Incident ID Filter ---
    if incident_id is not None:
        conditions.append({"incident_id": {"$eq": incident_id}})
    
    # --- Date Range Filter ---
    if date_from is not None:
        conditions.append({"date": {"$gte": date_from}})
    
    if date_to is not None:
        conditions.append({"date": {"$lte": date_to}})
    
    # --- Combine Conditions ---
    if not conditions:
        # No filters - return empty dict (search all documents)
        return {}  
    elif len(conditions) == 1:
        # Single condition - no need for $and
        return conditions[0]
    else:
        # Multiple conditions - combine with $and
        return {"$and": conditions}


# =============================================================================
# FUNCTION: retrieve
# =============================================================================
def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    severity: Optional[str | list[str]] = None,
    root_cause_category: Optional[str | list[str]] = None,
    services_affected: Optional[str | list[str]] = None,
    owner_team: Optional[str] = None,
    incident_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    verbose: bool = False,
) -> list:
    """
    Retrieve relevant document chunks using vector search with optional filters.
    
    THIS IS THE MAIN RETRIEVAL FUNCTION!
    
    STEPS:
    1. Build pre-filter from provided metadata constraints
    2. Connect to vector store
    3. Embed the query using OpenAI
    4. Perform filtered vector search
    5. Return top-K most similar documents
    
    Args:
        query: Natural language question/search query
        top_k: Number of documents to retrieve (default: 5)
        severity: Filter by severity (e.g., "S1 - Critical Severity")
        root_cause_category: Filter by category (e.g., "database")
        services_affected: Filter by service (e.g., "payment-rx-orc")
        owner_team: Filter by team (e.g., "core", "infrastructure")
        incident_id: Filter by specific incident ID
        date_from: Filter by start date (YYYY-MM-DD)
        date_to: Filter by end date (YYYY-MM-DD)
        verbose: Print debug information
    
    Returns:
        List of LangChain Document objects with page_content and metadata
    
    Example:
        # Simple search (no filters)
        docs = retrieve("database connection issues")
        
        # Filtered search (Metadata-Filtered RAG)
        docs = retrieve(
            query="database connection issues",
            severity="S1 - Critical Severity",
            root_cause_category="database"
        )
    """
    vector_store, client = get_vector_store()
    
    try:
        # Build the pre-filter
        pre_filter = build_pre_filter(
            severity=severity,
            root_cause_category=root_cause_category,
            services_affected=services_affected,
            owner_team=owner_team,
            incident_id=incident_id,
            date_from=date_from,
            date_to=date_to,
        )
        
        if verbose:
            print(f"🔍 Query: {query}")
            if pre_filter:
                print(f"📁 Pre-filter: {pre_filter}")
            else:
                print(f"📁 Pre-filter: None (searching all documents)")
        
        # Perform the search
        # similarity_search handles:
        # 1. Embedding the query
        # 2. Applying pre-filter (if any)
        # 3. Finding top-K similar vectors
        # 4. Returning documents with metadata
        if pre_filter:
            results = vector_store.similarity_search(
                query=query,
                k=top_k,
                pre_filter=pre_filter
            )
        else:
            results = vector_store.similarity_search(
                query=query,
                k=top_k
            )
        
        if verbose:
            print(f"📄 Retrieved: {len(results)} documents")
        
        return results
        
    finally:
        # Always close the connection
        client.close()


# =============================================================================
# FUNCTION: retrieve_with_scores
# =============================================================================
def retrieve_with_scores(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    severity: Optional[str | list[str]] = None,
    root_cause_category: Optional[str | list[str]] = None,
    services_affected: Optional[str | list[str]] = None,
    owner_team: Optional[str] = None,
    **kwargs
) -> list[tuple]:
    """
    Retrieve documents with similarity scores.
    
    Same as retrieve(), but returns (document, score) tuples.
    
    SCORES:
    - Higher score = More similar
    - Range depends on similarity metric (cosine: 0-1)
    - Useful for:
      - Debugging retrieval quality
      - Setting similarity thresholds
      - Evaluation metrics
    
    Returns:
        List of (Document, score) tuples
    """
    vector_store, client = get_vector_store()
    
    try:
        pre_filter = build_pre_filter(
            severity=severity,
            root_cause_category=root_cause_category,
            services_affected=services_affected,
            owner_team=owner_team,
            **kwargs
        )
        
        if pre_filter:
            results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                pre_filter=pre_filter
            )
        else:
            results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k
            )
        
        return results
        
    finally:
        client.close()


# =============================================================================
# FUNCTION: format_context
# =============================================================================
def format_context(documents: list) -> str:
    """
    Format retrieved documents into a context string for the LLM.
    
    WHY THIS MATTERS:
    - LLMs need text, not Document objects
    - We include metadata for traceability
    - Numbered for clarity in citations
    
    OUTPUT FORMAT:
        [Document 1]
        Source: PFU-191-database-outage.md
        Incident: PFU-191 | Severity: S1 | Category: database
        Content:
        ... the actual text ...
        
        ---
        
        [Document 2]
        ...
    """
    if not documents:
        return "No relevant documents found."
    
    context_parts = []
    
    for i, doc in enumerate(documents, 1):
        # Extract metadata
        source = doc.metadata.get("source_file", "Unknown")
        incident_id = doc.metadata.get("incident_id", "Unknown")
        severity = doc.metadata.get("severity", "Unknown")
        category = doc.metadata.get("root_cause_category", "Unknown")
        services = doc.metadata.get("services_affected", [])
        
        # Format services list
        services_str = ", ".join(services) if services else "N/A"
        
        # Build the context block
        context_block = f"""[Document {i}]
Source: {source}
Incident: {incident_id} | Severity: {severity} | Category: {category}
Services: {services_str}
Content:
{doc.page_content}
"""
        context_parts.append(context_block)
    
    return "\n---\n\n".join(context_parts)


# =============================================================================
# FUNCTION: deduplicate_by_incident
# =============================================================================
def deduplicate_by_incident(documents: list, keep: str = "first") -> list:
    """
    Deduplicate retrieved chunks, keeping one chunk per incident.
    
    WHY DEDUPLICATE:
    - Chunking splits one postmortem into multiple chunks
    - Retrieval may return multiple chunks from the same incident
    - This wastes retrieval slots and inflates precision metrics
    
    Args:
        documents: List of Document objects from retrieval
        keep: Which chunk to keep - "first" (highest score) or "last"
        
    Returns:
        List of Documents, one per unique incident
        
    Example:
        # Before: [PFU-187 chunk1, PFU-187 chunk2, PFU-112 chunk1]
        # After:  [PFU-187 chunk1, PFU-112 chunk1]
    """
    seen_incidents = set()
    unique_docs = []
    
    for doc in documents:
        incident_id = doc.metadata.get("incident_id", "Unknown")
        
        if incident_id not in seen_incidents:
            seen_incidents.add(incident_id)
            unique_docs.append(doc)
    
    return unique_docs


# =============================================================================
# FUNCTION: merge_chunks_by_incident
# =============================================================================
def merge_chunks_by_incident(documents: list) -> list:
    """
    Merge all retrieved chunks that belong to the same incident.
    
    WHY MERGE:
    - Provides complete context for each incident
    - Avoids losing information from discarded chunks
    - Better for structured documents like postmortems
    
    Args:
        documents: List of Document objects from retrieval
        
    Returns:
        List of Documents with merged content per incident
        
    Example:
        # Before: [PFU-187 chunk2 (root cause), PFU-187 chunk0 (summary)]
        # After:  [PFU-187 (summary + root cause)] - in DOCUMENT order
        
    Note:
        - Chunks are sorted by chunk_index (document order) before merging
        - Incidents are ordered by their best chunk's retrieval score
        - Duplicate content from overlap is NOT removed
        - Metadata is taken from the first chunk (by document order)
    """    
    # Group chunks by incident_id, preserving retrieval order for incidents
    # But we'll sort chunks within each incident by document position
    incident_chunks: OrderedDict[str, list] = OrderedDict()
    
    for doc in documents:
        incident_id = doc.metadata.get("incident_id", "Unknown")
        chunk_index = doc.metadata.get("chunk_index", 0)
        
        if incident_id not in incident_chunks:
            incident_chunks[incident_id] = []
        
        # Store tuple of (chunk_index, content, metadata) for sorting
        incident_chunks[incident_id].append((
            chunk_index,
            doc.page_content,
            doc.metadata
        ))
    
    # Create merged documents
    merged_docs = []
    for incident_id, chunks_data in incident_chunks.items():
        # Sort by chunk_index to restore DOCUMENT ORDER
        chunks_data.sort(key=lambda x: x[0])
        
        # Extract content in document order
        ordered_content = [chunk[1] for chunk in chunks_data]
        
        # Use metadata from the FIRST chunk (by document order)
        first_chunk_metadata = chunks_data[0][2]
        
        # Join chunks with separator
        merged_content = "\n\n[...]\n\n".join(ordered_content)
        
        # Create new document with merged content
        merged_doc = Document(
            page_content=merged_content,
            metadata=first_chunk_metadata
        )
        merged_docs.append(merged_doc)
    
    return merged_docs


# =============================================================================
# FUNCTION: retrieve_unique
# =============================================================================
def retrieve_unique(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    merge: bool = True,
    fetch_multiplier: int = 3,
    **filter_kwargs
) -> list:
    """
    Retrieve top_k unique incidents (not chunks).
    
    This is the RECOMMENDED retrieval function for most use cases.
    It addresses the duplicate chunk problem by either:
    - Deduplicating (keeping best chunk per incident)
    - Merging (combining all chunks per incident)
    
    Args:
        query: The search query
        top_k: Number of unique incidents to return
        merge: If True, merge chunks. If False, keep best chunk only.
        fetch_multiplier: Fetch this many times top_k to account for duplicates
        **filter_kwargs: Metadata filters (severity, root_cause_category, etc.)
        
    Returns:
        List of Documents, one per unique incident
        
    Example:
        # Get 5 unique incidents about database issues
        docs = retrieve_unique(
            "database timeout errors",
            top_k=5,
            merge=True,
            root_cause_category="database"
        )
        
        # Result: 5 documents, each representing a complete incident
    """
    # Fetch more chunks to account for duplicates
    chunks = retrieve(
        query=query,
        top_k=top_k * fetch_multiplier,
        **filter_kwargs
    )
    
    if merge:
        # Merge chunks from same incident
        merged = merge_chunks_by_incident(chunks)
        return merged[:top_k]
    else:
        # Keep only the best chunk per incident
        unique = deduplicate_by_incident(chunks)
        return unique[:top_k]


# =============================================================================
# FUNCTION: get_collection_stats
# =============================================================================
def get_collection_stats() -> dict:
    """
    Get statistics about the document collection.
    
    Useful for:
    - Understanding what filters are available
    - Debugging empty results
    - Checking ingestion worked correctly
    """
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        stats = {
            "total_documents": collection.count_documents({}),
            "severities": {},
            "categories": {},
            "services": {},
            "teams": {}
        }
        
        # Count by severity
        pipeline = [
            {"$group": {"_id": "$severity", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        for item in collection.aggregate(pipeline):
            stats["severities"][item["_id"]] = item["count"]
        
        # Count by category
        pipeline = [
            {"$group": {"_id": "$root_cause_category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        for item in collection.aggregate(pipeline):
            stats["categories"][item["_id"]] = item["count"]
        
        # Count by service (top 10)
        pipeline = [
            {"$unwind": "$services_affected"},
            {"$group": {"_id": "$services_affected", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        for item in collection.aggregate(pipeline):
            stats["services"][item["_id"]] = item["count"]
        
        # Count by team
        pipeline = [
            {"$group": {"_id": "$owner_team", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        for item in collection.aggregate(pipeline):
            if item["_id"]:  # Skip empty team names
                stats["teams"][item["_id"]] = item["count"]
        
        return stats
        
    finally:
        client.close()


# =============================================================================
# FUNCTION: debug_collection
# =============================================================================
def debug_collection():
    """
    Print debug information about the collection.
    
    Run this if retrieval isn't working to diagnose issues.
    """
    print("=" * 60)
    print("🔍 Collection Debug Info")
    print("=" * 60)
    
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        # Document count
        doc_count = collection.count_documents({})
        print(f"\n📊 Total documents: {doc_count}")
        
        if doc_count == 0:
            print("❌ No documents found! Run ingestion.py first.")
            return
        
        # Sample document
        sample = collection.find_one()
        print(f"\n📄 Sample document fields:")
        for key in sorted(sample.keys()):
            if key == "embedding":
                print(f"   {key}: [vector with {len(sample[key])} dimensions]")
            elif isinstance(sample[key], str) and len(sample[key]) > 50:
                print(f"   {key}: {sample[key][:50]}...")
            else:
                print(f"   {key}: {sample[key]}")
        
        # Search indexes
        print(f"\n🔎 Search indexes:")
        try:
            indexes = list(collection.list_search_indexes())
            if not indexes:
                print("   ❌ No search indexes found!")
                print("   Vector search won't work until index is created.")
            else:
                for idx in indexes:
                    name = idx.get("name", "unknown")
                    status = idx.get("status", "unknown")
                    status_icon = "✅" if status == "READY" else "⏳"
                    print(f"   {status_icon} {name}: {status}")
        except Exception as e:
            print(f"   ⚠️ Could not list indexes: {e}")
        
        # Stats
        print(f"\n📈 Collection stats:")
        stats = get_collection_stats()
        
        print(f"\n   By Severity:")
        for sev, count in stats["severities"].items():
            print(f"      {sev}: {count}")
        
        print(f"\n   By Category:")
        for cat, count in stats["categories"].items():
            print(f"      {cat}: {count}")
        
        print(f"\n   Top Services:")
        for svc, count in list(stats["services"].items())[:5]:
            print(f"      {svc}: {count}")
        
    finally:
        client.close()


# =============================================================================
# MAIN - Test Retrieval
# =============================================================================
def main():
    """
    Test retrieval with example queries.
    
    Demonstrates:
    1. Basic retrieval (no filters)
    2. Filtered retrieval (Metadata-Filtered RAG)
    3. Comparison of results
    """
    print("=" * 60)
    print("🔍 Postmortem RAG - Retrieval Test")
    print("=" * 60)
    
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL not set")
    
    # First, debug the collection
    debug_collection()
    
    print("\n" + "=" * 60)
    print("🧪 Running Retrieval Tests")
    print("=" * 60)
    
    # --- Test 1: Basic Retrieval (No Filters) ---
    print("\n" + "-" * 40)
    print("Test 1: Basic Retrieval (No Filters)")
    print("-" * 40)
    
    query1 = "database timeout or latency issues"
    print(f"Query: {query1}\n")
    
    results1 = retrieve(query1, top_k=3, verbose=True)
    
    for i, doc in enumerate(results1, 1):
        print(f"\n  [{i}] {doc.metadata.get('incident_id', 'N/A')}")
        print(f"      Severity: {doc.metadata.get('severity', 'N/A')}")
        print(f"      Category: {doc.metadata.get('root_cause_category', 'N/A')}")
        print(f"      Preview: {doc.page_content[:100]}...")
    
    # --- Test 2: Filtered Retrieval (Critical Only) ---
    print("\n" + "-" * 40)
    print("Test 2: Filtered by Severity (S1 Critical)")
    print("-" * 40)
    
    query2 = "database issues"
    print(f"Query: {query2}")
    print(f"Filter: severity='S1 - Critical Severity'\n")
    
    results2 = retrieve(
        query=query2,
        top_k=3,
        severity="S1 - Critical Severity",
        verbose=True
    )
    
    for i, doc in enumerate(results2, 1):
        print(f"\n  [{i}] {doc.metadata.get('incident_id', 'N/A')}")
        print(f"      Severity: {doc.metadata.get('severity', 'N/A')}")
        print(f"      Category: {doc.metadata.get('root_cause_category', 'N/A')}")
        print(f"      Preview: {doc.page_content[:100]}...")
    
    # --- Test 3: Filtered by Category ---
    print("\n" + "-" * 40)
    print("Test 3: Filtered by Category (database)")
    print("-" * 40)
    
    query3 = "how did we fix the issue"
    print(f"Query: {query3}")
    print(f"Filter: root_cause_category='database'\n")
    
    results3 = retrieve(
        query=query3,
        top_k=3,
        root_cause_category="database",
        verbose=True
    )
    
    for i, doc in enumerate(results3, 1):
        print(f"\n  [{i}] {doc.metadata.get('incident_id', 'N/A')}")
        print(f"      Severity: {doc.metadata.get('severity', 'N/A')}")
        print(f"      Category: {doc.metadata.get('root_cause_category', 'N/A')}")
        print(f"      Preview: {doc.page_content[:100]}...")
    
    # --- Test 4: Combined Filters ---
    print("\n" + "-" * 40)
    print("Test 4: Combined Filters")
    print("-" * 40)
    
    query4 = "what went wrong"
    print(f"Query: {query4}")
    print(f"Filters: severity=['S1 - Critical Severity', 'S2 - High Severity']")
    print(f"         root_cause_category='database'\n")
    
    results4 = retrieve(
        query=query4,
        top_k=3,
        severity=["S1 - Critical Severity", "S2 - High Severity"],
        root_cause_category="database",
        verbose=True
    )
    
    for i, doc in enumerate(results4, 1):
        print(f"\n  [{i}] {doc.metadata.get('incident_id', 'N/A')}")
        print(f"      Severity: {doc.metadata.get('severity', 'N/A')}")
        print(f"      Category: {doc.metadata.get('root_cause_category', 'N/A')}")
    
    # --- Show formatted context ---
    print("\n" + "-" * 40)
    print("Formatted Context (for LLM)")
    print("-" * 40)
    
    if results4:
        context = format_context(results4[:2])  # Just first 2 for brevity
        print(context[:1000] + "..." if len(context) > 1000 else context)
    
    print("\n✅ Retrieval tests complete!")


if __name__ == "__main__":
    main()
