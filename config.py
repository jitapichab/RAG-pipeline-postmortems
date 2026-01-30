"""
Postmortem RAG - Configuration Module
======================================

Centralized configuration for the RAG pipeline.
All other modules import from here to ensure consistency.

Why centralize configuration?
- Single source of truth for database names, model settings, etc.
- Easier to switch environments (dev/staging/prod)
- Reduces bugs from inconsistent values across modules
- Makes configuration changes a one-line edit

Author: Jorge Tapicha
Date: 2026-01-28
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# PATHS
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Documents directory
DOCS_DIR = PROJECT_ROOT / "docs"


# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# =============================================================================
# MONGODB CONFIGURATION
# =============================================================================

DB_NAME = "rag_playbook"
"""MongoDB database name"""

COLLECTION_NAME = "postmortem_rag"
"""MongoDB collection name for storing document chunks"""

INDEX_NAME = "postmortem_index"
"""MongoDB Atlas Vector Search index name"""


# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

EMBEDDING_MODEL = "text-embedding-3-small"
"""
OpenAI embedding model to use.

Options:
- text-embedding-3-small: 1536 dims, cheap, fast (recommended)
- text-embedding-3-large: 3072 dims, more accurate, 6x cost
- text-embedding-ada-002: 1536 dims, legacy model
"""

EMBEDDING_DIMENSIONS = 1536
"""Vector dimensions - must match the embedding model"""


# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

CHUNK_SIZE = 1000
"""
Maximum characters per chunk.

Trade-offs:
- Smaller chunks (500): More precise retrieval, less context per chunk
- Larger chunks (2000): More context, may include irrelevant content
- 1000 is a good default for postmortems (~250 tokens)
"""

CHUNK_OVERLAP = 200
"""
Character overlap between consecutive chunks.

Why overlap?
- Prevents losing context at chunk boundaries
- If a key sentence is split, both chunks have partial context
- 20% of chunk_size is a common heuristic
"""


# =============================================================================
# LLM CONFIGURATION
# =============================================================================

GENERATION_MODEL = "gpt-4o-mini"
"""
LLM for answer generation.

Options:
- gpt-4o-mini: Cheap, fast, good for most use cases
- gpt-4o: More capable, 30x cost
- gpt-4: Legacy, expensive
"""

GENERATION_TEMPERATURE = 0
"""
LLM temperature for generation.

- 0: Deterministic, consistent outputs
- 0.3-0.7: Some creativity
- 1.0: Maximum randomness

For RAG, use 0 to ensure consistent, factual responses.
"""

JUDGE_MODEL = "gpt-4o-mini"
"""LLM for evaluation judgments (precision, groundedness)"""


# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================

DEFAULT_TOP_K = 5
"""
Default number of documents to retrieve.

Trade-offs:
- K=3: Fast, precise, may miss relevant context
- K=5: Good balance (recommended)
- K=10: More context, higher recall, more noise
"""


# =============================================================================
# AVAILABLE METADATA VALUES
# =============================================================================
# These are the values available for filtering in retrieval

AVAILABLE_SEVERITIES = [
    "S1 - Critical Severity",
    "S2 - High Severity",
    "S3 - Medium Severity",
    "S4 - Low Severity",
    "Unknown"
]

AVAILABLE_ROOT_CAUSE_CATEGORIES = [
    "database",       # MongoDB, Postgres, Aurora issues
    "capacity",       # Load, traffic, scaling issues
    "deployment",     # Rollout, release issues
    "configuration",  # Config, settings issues
    "network",        # DNS, timeout, connection issues
    "code_bug",       # Logic errors, null pointers
    "third_party",    # External provider issues
    "unknown"         # Could not determine
]


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config() -> dict:
    """
    Validate that required configuration is present.
    
    Returns:
        dict with validation status and any errors
    """
    errors = []
    warnings = []
    
    if not MONGO_DB_URL:
        errors.append("MONGO_DB_URL not set in environment")
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY not set in environment")
    
    if not DOCS_DIR.exists():
        warnings.append(f"Docs directory does not exist: {DOCS_DIR}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("RAG PIPELINE CONFIGURATION")
    print("=" * 60)
    print(f"\n📁 Paths:")
    print(f"   Project Root: {PROJECT_ROOT}")
    print(f"   Docs Dir: {DOCS_DIR}")
    
    print(f"\n🗄️ MongoDB:")
    print(f"   Database: {DB_NAME}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Index: {INDEX_NAME}")
    
    print(f"\n🔢 Embeddings:")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   Dimensions: {EMBEDDING_DIMENSIONS}")
    
    print(f"\n📄 Chunking:")
    print(f"   Chunk Size: {CHUNK_SIZE} chars")
    print(f"   Overlap: {CHUNK_OVERLAP} chars")
    
    print(f"\n🤖 LLM:")
    print(f"   Generation: {GENERATION_MODEL}")
    print(f"   Temperature: {GENERATION_TEMPERATURE}")
    print(f"   Judge: {JUDGE_MODEL}")
    
    print(f"\n🔍 Retrieval:")
    print(f"   Default K: {DEFAULT_TOP_K}")
    
    validation = validate_config()
    if validation["errors"]:
        print(f"\n❌ Errors:")
        for e in validation["errors"]:
            print(f"   - {e}")
    if validation["warnings"]:
        print(f"\n⚠️ Warnings:")
        for w in validation["warnings"]:
            print(f"   - {w}")
    
    print()


if __name__ == "__main__":
    print_config()
