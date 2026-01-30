"""
Get Document Chunks - Testing Script
=====================================

This script tests that chunks are retrieved and merged in the correct
document order (by chunk_index), not by retrieval score.

Usage:
    python scripts/get_docs_chunk.py PFU-187
    python scripts/get_docs_chunk.py PFU-191 --query "DNS failure"
    python scripts/get_docs_chunk.py PFU-187 --show-content

Author: Jorge Tapicha
Date: 2026-01-28
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_NAME, COLLECTION_NAME
from utils import get_mongo_client
from retrieval import retrieve, merge_chunks_by_incident


def get_all_chunks_for_incident(incident_id: str) -> list:
    """
    Get ALL chunks for a specific incident directly from MongoDB.
    
    This bypasses vector search to show the ground truth chunk order.
    """
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    # Query all chunks for this incident
    chunks = list(collection.find(
        {"incident_id": incident_id},
        {"text": 1, "chunk_index": 1, "incident_id": 1, "title": 1, "_id": 0}
    ).sort("chunk_index", 1))  # Sort by chunk_index ascending
    
    client.close()
    return chunks


def test_retrieval_order(incident_id: str, query: str = None, show_content: bool = False):
    """
    Test that chunks are retrieved and merged in correct order.
    """
    print("=" * 70)
    print(f"📄 CHUNK ORDER TEST: {incident_id}")
    print("=" * 70)
    
    # --- Step 1: Get ground truth from MongoDB ---
    print("\n1️⃣  GROUND TRUTH (All chunks from MongoDB, sorted by chunk_index):")
    print("-" * 70)
    
    all_chunks = get_all_chunks_for_incident(incident_id)
    
    if not all_chunks:
        print(f"   ❌ No chunks found for {incident_id}")
        print("   Make sure ingestion has been run.")
        return
    
    print(f"   Found {len(all_chunks)} chunks for {incident_id}")
    print()
    
    for chunk in all_chunks:
        idx = chunk.get("chunk_index", "?")
        text_preview = chunk.get("text", "")[:80].replace("\n", " ")
        print(f"   [chunk {idx}] {text_preview}...")
    
    # --- Step 2: Test retrieval with query ---
    if query is None:
        query = f"incident {incident_id}"
    
    print(f"\n2️⃣  RETRIEVAL TEST (Query: '{query}')")
    print("-" * 70)
    
    # Retrieve with higher K to get multiple chunks
    docs = retrieve(
        query=query,
        top_k=20,
        incident_id=incident_id  # Filter to just this incident
    )
    
    if not docs:
        print(f"   ❌ No documents retrieved for query")
        return
    
    print(f"   Retrieved {len(docs)} chunks")
    print("\n   Retrieval order (by similarity score):")
    
    for i, doc in enumerate(docs):
        idx = doc.metadata.get("chunk_index", "?")
        text_preview = doc.page_content[:60].replace("\n", " ")
        print(f"   {i+1}. [chunk {idx}] {text_preview}...")
    
    # --- Step 3: Test merge order ---
    print(f"\n3️⃣  MERGE TEST (Should be in document order)")
    print("-" * 70)
    
    merged = merge_chunks_by_incident(docs)
    
    if merged:
        merged_doc = merged[0]
        
        # Extract chunk indices from the merged content
        # (The content should be in order now)
        print("   Merged document content order:")
        print()
        
        # Split by our separator to see order
        parts = merged_doc.page_content.split("\n\n[...]\n\n")
        
        for i, part in enumerate(parts):
            # Try to find which chunk this is
            preview = part[:60].replace("\n", " ")
            
            # Find matching chunk index
            matching_idx = "?"
            for chunk in all_chunks:
                if chunk.get("text", "").startswith(part[:50]):
                    matching_idx = chunk.get("chunk_index", "?")
                    break
            
            print(f"   Part {i+1} [chunk {matching_idx}]: {preview}...")
        
        # Verify order
        print()
        chunk_indices = []
        for part in parts:
            for chunk in all_chunks:
                if chunk.get("text", "").startswith(part[:50]):
                    chunk_indices.append(chunk.get("chunk_index", -1))
                    break
        
        is_ordered = chunk_indices == sorted(chunk_indices)
        
        if is_ordered:
            print(f"   ✅ PASS: Chunks are in document order: {chunk_indices}")
        else:
            print(f"   ❌ FAIL: Chunks are NOT in order: {chunk_indices}")
            print(f"   Expected: {sorted(chunk_indices)}")
    
    # --- Step 4: Show full content if requested ---
    if show_content and merged:
        print(f"\n4️⃣  FULL MERGED CONTENT")
        print("-" * 70)
        print(merged[0].page_content[:2000])
        if len(merged[0].page_content) > 2000:
            print("\n... [truncated] ...")


def list_available_incidents():
    """List all available incident IDs."""
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    pipeline = [
        {"$group": {"_id": "$incident_id", "chunks": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
        {"$limit": 20}
    ]
    
    incidents = list(collection.aggregate(pipeline))
    client.close()
    
    print("\n📋 Available incidents (first 20):")
    for inc in incidents:
        print(f"   {inc['_id']} ({inc['chunks']} chunks)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test chunk retrieval and merge order for a postmortem"
    )
    parser.add_argument(
        "incident_id",
        nargs="?",
        help="Incident ID (e.g., PFU-187)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Custom query for retrieval (default: 'incident {incident_id}')"
    )
    parser.add_argument(
        "--show-content", "-c",
        action="store_true",
        help="Show full merged content"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available incident IDs"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_incidents()
        return
    
    if not args.incident_id:
        parser.print_help()
        print("\n💡 Examples:")
        print("   python scripts/get_docs_chunk.py PFU-187")
        print("   python scripts/get_docs_chunk.py PFU-191 --query 'DNS failure'")
        print("   python scripts/get_docs_chunk.py --list")
        return
    
    test_retrieval_order(
        incident_id=args.incident_id,
        query=args.query,
        show_content=args.show_content
    )


if __name__ == "__main__":
    main()
