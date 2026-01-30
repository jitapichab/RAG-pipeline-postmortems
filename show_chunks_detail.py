"""
Show Chunks Detail - Visualize how documents are chunked
=========================================================

This script shows exactly how a postmortem is chunked and what
the LLM receives, including overlap regions.

Usage:
    python show_chunks_detail.py PFU-187
    python show_chunks_detail.py PFU-191 --merged

Author: Jorge Tapicha
Date: 2026-01-28
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import DB_NAME, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from utils import get_mongo_client


def get_chunks_for_incident(incident_id: str) -> list:
    """Get all chunks for an incident, sorted by chunk_index."""
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    chunks = list(collection.find(
        {"incident_id": incident_id}
    ).sort("chunk_index", 1))
    
    client.close()
    return chunks


def show_individual_chunks(chunks: list):
    """Show each chunk separately with clear boundaries."""
    print("\n" + "=" * 80)
    print("📄 INDIVIDUAL CHUNKS (What vector search indexes)")
    print("=" * 80)
    print(f"\nChunk settings: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print()
    
    for i, chunk in enumerate(chunks):
        idx = chunk.get("chunk_index", i)
        text = chunk.get("text", "")
        
        print("┌" + "─" * 78 + "┐")
        print(f"│ CHUNK {idx}  (length: {len(text)} chars)")
        print("├" + "─" * 78 + "┤")
        
        # Show text with line wrapping
        lines = text.split('\n')
        for line in lines:
            # Wrap long lines
            while len(line) > 76:
                print(f"│ {line[:76]} │")
                line = line[76:]
            print(f"│ {line:<76} │")
        
        print("└" + "─" * 78 + "┘")
        print()


def show_overlap_analysis(chunks: list):
    """Analyze and show overlap between consecutive chunks."""
    print("\n" + "=" * 80)
    print("🔗 OVERLAP ANALYSIS (Text shared between chunks)")
    print("=" * 80)
    
    for i in range(len(chunks) - 1):
        chunk1 = chunks[i].get("text", "")
        chunk2 = chunks[i + 1].get("text", "")
        idx1 = chunks[i].get("chunk_index", i)
        idx2 = chunks[i + 1].get("chunk_index", i + 1)
        
        # Find overlap by checking end of chunk1 vs start of chunk2
        overlap_text = ""
        for overlap_len in range(min(len(chunk1), len(chunk2), CHUNK_OVERLAP + 100), 0, -1):
            if chunk1[-overlap_len:] == chunk2[:overlap_len]:
                overlap_text = chunk1[-overlap_len:]
                break
        
        print(f"\n📎 Overlap between chunk {idx1} → chunk {idx2}:")
        print("-" * 80)
        
        if overlap_text:
            print(f"   Overlap length: {len(overlap_text)} chars")
            print(f"   Overlap text:")
            print("   ┌" + "─" * 74 + "┐")
            
            lines = overlap_text.split('\n')
            for line in lines[:5]:  # Show first 5 lines
                if len(line) > 72:
                    line = line[:69] + "..."
                print(f"   │ {line:<72} │")
            
            if len(lines) > 5:
                print(f"   │ {'... (' + str(len(lines) - 5) + ' more lines)':<72} │")
            
            print("   └" + "─" * 74 + "┘")
        else:
            print("   ⚠️  No exact overlap found (chunks may not be consecutive)")


def show_merged_content(chunks: list):
    """Show what the LLM receives after merging."""
    print("\n" + "=" * 80)
    print("🤖 MERGED CONTENT (What the LLM receives)")
    print("=" * 80)
    print("\nNote: Chunks are joined with '[...]' separator.")
    print("      Overlap text appears in BOTH chunks (duplicated).\n")
    
    # Merge chunks in order
    texts = [chunk.get("text", "") for chunk in chunks]
    merged = "\n\n[...]\n\n".join(texts)
    
    print("┌" + "─" * 78 + "┐")
    print(f"│ MERGED DOCUMENT  (total length: {len(merged)} chars)")
    print("├" + "─" * 78 + "┤")
    
    lines = merged.split('\n')
    for line in lines:
        while len(line) > 76:
            print(f"│ {line[:76]} │")
            line = line[76:]
        print(f"│ {line:<76} │")
    
    print("└" + "─" * 78 + "┘")


def show_raw_text(chunks: list):
    """Show raw text without formatting for copy/paste."""
    print("\n" + "=" * 80)
    print("📝 RAW MERGED TEXT (Copy/paste friendly)")
    print("=" * 80)
    print()
    
    texts = [chunk.get("text", "") for chunk in chunks]
    merged = "\n\n[...]\n\n".join(texts)
    
    print(merged)
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Show detailed chunk information for a postmortem"
    )
    parser.add_argument(
        "incident_id",
        help="Incident ID (e.g., PFU-187)"
    )
    parser.add_argument(
        "--merged", "-m",
        action="store_true",
        help="Show merged content (what LLM receives)"
    )
    parser.add_argument(
        "--raw", "-r",
        action="store_true",
        help="Show raw text (copy/paste friendly)"
    )
    parser.add_argument(
        "--overlap", "-o",
        action="store_true",
        help="Show overlap analysis"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Show everything"
    )
    
    args = parser.parse_args()
    
    # Get chunks
    chunks = get_chunks_for_incident(args.incident_id)
    
    if not chunks:
        print(f"❌ No chunks found for {args.incident_id}")
        return
    
    print("\n" + "=" * 80)
    print(f"📋 POSTMORTEM: {args.incident_id}")
    print(f"   Title: {chunks[0].get('title', 'Unknown')}")
    print(f"   Total chunks: {len(chunks)}")
    print("=" * 80)
    
    # Default: show individual chunks
    if args.all or not (args.merged or args.raw or args.overlap):
        show_individual_chunks(chunks)
    
    if args.all or args.overlap:
        show_overlap_analysis(chunks)
    
    if args.all or args.merged:
        show_merged_content(chunks)
    
    if args.raw:
        show_raw_text(chunks)


if __name__ == "__main__":
    main()
