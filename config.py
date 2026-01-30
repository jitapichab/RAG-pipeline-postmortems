"""
Configuration
=============

All settings in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DOCS_DIR = PROJECT_ROOT / "docs"

# Environment
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB
DB_NAME = "rag_playbook"
COLLECTION_NAME = "postmortem_rag"
INDEX_NAME = "postmortem_index"

# Embeddings
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM
GENERATION_MODEL = "gpt-4o-mini"
GENERATION_TEMPERATURE = 0
JUDGE_MODEL = "gpt-4o-mini"

# Retrieval
DEFAULT_TOP_K = 5
