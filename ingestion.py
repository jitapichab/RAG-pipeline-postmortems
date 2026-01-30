"""
Postmortem RAG - Ingestion Pipeline
====================================

This script loads postmortem markdown files, chunks them, creates embeddings,
and stores them in MongoDB Atlas for vector search.

ARCHITECTURE:
    Markdown Files → Parse Frontmatter → Chunk Text → Embed → Store in MongoDB

Author: Jorge Tapicha.
Date: 2026-01-28
"""

import os         
from pathlib import Path 
from dotenv import load_dotenv  
import certifi
import frontmatter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# Purpose: Convert text to vector embeddings using OpenAI's API
# Why: Vectors enable semantic similarity search
#   - "database timeout" and "DB connection slow" have similar vectors
#   - Traditional search wouldn't match these!
#
# How it works:
#   text = "MongoDB timeout error"
#   vector = embeddings.embed_query(text)  # Returns [0.02, -0.15, ..., 0.08]
#   # Vector has 1536 dimensions for text-embedding-3-small
# Install: pip install langchain-openai

from langchain_mongodb import MongoDBAtlasVectorSearch
# Purpose: Store and search vectors in MongoDB Atlas
# Why MongoDB Atlas?
#   - Combines document storage + vector search in one database
#   - No need for separate vector DB (Pinecone, Qdrant)
#   - Pre-filtering by metadata before vector search
#
# How it works:
#   1. Stores documents with embeddings in MongoDB collection
#   2. Creates vector index for similarity search
#   3. Query: embed query → find similar vectors → return documents
# Install: pip install langchain-mongodb

from langchain_core.documents import Document
# Purpose: LangChain's standard document representation
# Structure:
#   Document(
#       page_content="The actual text content...",
#       metadata={"severity": "S1", "source": "PFU-191.md", ...}
#   )
# Why: Consistent interface across all LangChain components
# Install: Comes with langchain-core (dependency of others)
from pymongo import MongoClient

load_dotenv()


MONGO_DB_URL = os.getenv("MONGO_DB_URL")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DB_NAME = "rag_playbook"

COLLECTION_NAME = "postmortem_rag"

INDEX_NAME = "postmortem_index"

CHUNK_SIZE = 1000
# Maximum characters per chunk
# Why 1000?
#   - Small enough to fit many in LLM context
#   - Large enough to contain meaningful information
#   - ~250 tokens (4 chars ≈ 1 token)
# Trade-off:
#   - Smaller chunks = more precise retrieval, less context
#   - Larger chunks = more context, might include irrelevant info

CHUNK_OVERLAP = 200
# Characters to overlap between consecutive chunks
# Why overlap?
#   - Prevents losing context at chunk boundaries
#   - If a sentence is split, both chunks have partial context
# Example:
#   Original: "The root cause was X. The fix was Y."
#   Chunk 1: "The root cause was X. The f..."
#   Chunk 2: "...s X. The fix was Y."  (overlap: "s X. The f")


# =============================================================================
# FUNCTION: get_mongo_client
# =============================================================================
def get_mongo_client() -> MongoClient:
    """
    Create a MongoDB client connection.
    
    Why this function exists:
    - Centralizes connection logic
    - Handles TLS certificates for Atlas (cloud) connections
    - Can be reused across different functions
    
    Returns:
        MongoClient: Connected MongoDB client
        
    Raises:
        ValueError: If MONGO_DB_URL is not set
    """
    if not MONGO_DB_URL:
        raise ValueError(
            "MONGO_DB_URL environment variable not set. "
            "Add it to your .env file."
        )
    
    # Check if using MongoDB Atlas (cloud) vs local MongoDB
    if MONGO_DB_URL.startswith("mongodb+srv"):
        # Atlas uses SRV records for load balancing
        # Requires TLS with trusted CA certificates
        return MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
        # certifi.where() returns path to CA bundle
        # Without this: SSL: CERTIFICATE_VERIFY_FAILED
    else:
        # Local MongoDB doesn't need TLS
        return MongoClient(MONGO_DB_URL)


# =============================================================================
# FUNCTION: load_markdown_files
# =============================================================================
def load_markdown_files(docs_dir: Path) -> list[dict]:
    """
    Load all markdown files and parse their frontmatter.
    
    What this function does:
    1. Find all .md files in the docs directory
    2. Parse YAML frontmatter (metadata) from each file
    3. Separate content from metadata
    4. Return structured data for processing
    
    Args:
        docs_dir: Path to the docs directory containing .md files
        
    Returns:
        List of dicts with keys: 'metadata', 'content', 'source_file'
        
    Example output:
        [
            {
                'metadata': {'severity': 'S1', 'services_affected': [...]},
                'content': '# Incident Summary\\n...',
                'source_file': 'PFU-191-database-outage.md'
            },
            ...
        ]
    """
    documents = []
    
    # Path.glob() finds files matching a pattern
    # sorted() ensures consistent ordering (alphabetical by filename)
    md_files = sorted(docs_dir.glob("*.md"))
    
    print(f"Found {len(md_files)} markdown files")
    
    for md_path in md_files:
        try:
            # frontmatter.load() parses the YAML header and content
            # It handles the --- delimiters automatically
            post = frontmatter.load(md_path)
            
            documents.append({
                "metadata": dict(post.metadata),  # YAML frontmatter as dict
                "content": post.content,          # Everything after ---
                "source_file": md_path.name       # Filename for reference
            })
            
        except Exception as e:
            # Don't fail entire ingestion for one bad file
            print(f"  ⚠️ Error loading {md_path.name}: {e}")
            continue
    
    print(f"Successfully loaded {len(documents)} documents")
    return documents


# =============================================================================
# FUNCTION: chunk_documents
# =============================================================================
def chunk_documents(documents: list[dict]) -> list[Document]:
    """
    Split documents into smaller chunks while preserving metadata.
    
    WHY CHUNKING MATTERS:
    1. LLMs have token limits (GPT-4: 128K, but cost increases with size)
    2. Smaller chunks = more precise retrieval
    3. Each chunk gets its own embedding vector
    
    HOW RecursiveCharacterTextSplitter WORKS:
    - Tries to split at paragraph breaks (\\n\\n) first
    - If chunk still too large, splits at line breaks (\\n)
    - Then at spaces, then at any character
    - "Recursive" = tries each separator, falls back if needed
    
    Args:
        documents: List of parsed document dicts from load_markdown_files
        
    Returns:
        List of LangChain Document objects, each representing one chunk
    """
    # Initialize the text splitter with our configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,       # Max chars per chunk
        chunk_overlap=CHUNK_OVERLAP,  # Overlap between chunks
        length_function=len,          # How to measure length (character count)
        separators=["\n\n", "\n", " ", ""]  # Priority order for splitting
        # \n\n = paragraph break (preferred)
        # \n   = line break
        # " "  = space (word boundary)
        # ""   = any character (last resort)
    )
    
    all_chunks = []
    
    for doc in documents:
        # Split the content into chunks
        # Returns list of strings
        content_chunks = text_splitter.split_text(doc["content"])
        
        # Convert each chunk to a LangChain Document with metadata
        for chunk_idx, chunk_text in enumerate(content_chunks):
            # Start with the original frontmatter metadata
            chunk_metadata = doc["metadata"].copy()
            
            # Add chunk-specific metadata
            chunk_metadata["source_file"] = doc["source_file"]
            chunk_metadata["chunk_index"] = chunk_idx
            chunk_metadata["total_chunks"] = len(content_chunks)
            
            # Create LangChain Document object
            # This is the standard format for LangChain vector stores
            chunk_doc = Document(
                page_content=chunk_text,  # The actual text
                metadata=chunk_metadata    # All metadata for filtering
            )
            
            all_chunks.append(chunk_doc)
        
        print(f"  📄 {doc['source_file']}: {len(content_chunks)} chunks")
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks


# =============================================================================
# FUNCTION: setup_mongodb_collection
# =============================================================================
def setup_mongodb_collection():
    """
    Set up MongoDB collection for storing document chunks.
    
    WHAT THIS DOES:
    1. Connect to MongoDB Atlas
    2. Create collection if it doesn't exist
    3. Clear existing documents (for fresh ingestion)
    
    WHY CLEAR EXISTING:
    - Prevents duplicate documents on re-ingestion
    - Ensures clean state for testing
    - In production, you might want incremental updates instead
    
    Returns:
        Tuple of (MongoClient, Collection object)
    """
    client = get_mongo_client()
    db = client[DB_NAME]  # Access database (creates if doesn't exist)
    
    # Check if collection exists
    if COLLECTION_NAME not in db.list_collection_names():
        # Create new collection
        db.create_collection(COLLECTION_NAME)
        print(f"✅ Created collection: {COLLECTION_NAME}")
    else:
        # Collection exists - clear for fresh ingestion
        result = db[COLLECTION_NAME].delete_many({})
        print(f"🗑️ Cleared {result.deleted_count} existing documents from: {COLLECTION_NAME}")
    
    return client, db[COLLECTION_NAME]


# =============================================================================
# FUNCTION: create_embeddings_and_store
# =============================================================================
def create_embeddings_and_store(collection, chunks: list[Document]):
    """
    Create vector embeddings for all chunks and store in MongoDB.
    
    HOW EMBEDDINGS WORK:
    1. Text → OpenAI API → Vector (1536 floats)
    2. Similar texts have similar vectors (close in vector space)
    3. Vector search finds documents with similar meaning
    
    Example:
        "Database timeout error" → [0.02, -0.15, 0.33, ..., 0.08]
        "DB connection failed"  → [0.03, -0.14, 0.31, ..., 0.07]
        These vectors are close → semantic similarity!
    
    WHAT MongoDBAtlasVectorSearch.from_documents DOES:
    1. Calls OpenAI to embed each chunk's page_content
    2. Creates a document in MongoDB with:
       - text: the chunk content
       - embedding: the 1536-dimension vector
       - All metadata fields
    3. Returns a vector store object for querying
    
    Args:
        collection: MongoDB collection object
        chunks: List of LangChain Document objects
        
    Returns:
        MongoDBAtlasVectorSearch object (the vector store)
    """
    # Initialize the embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # Newest, cheapest, good quality
        # Other options:
        # - text-embedding-3-large: Better quality, more expensive
        # - text-embedding-ada-002: Legacy, still works
        openai_api_key=OPENAI_API_KEY
    )
    
    print(f"\n🔄 Creating embeddings for {len(chunks)} chunks...")
    print("   (This calls OpenAI API - may take a minute)")
    
    # This is the magic function that:
    # 1. Embeds all documents
    # 2. Stores them in MongoDB
    # 3. Sets up for vector search
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,           # Our chunked documents
        embedding=embeddings,       # The embedding model to use
        collection=collection,      # MongoDB collection
        index_name=INDEX_NAME       # Name for the vector index
    )
    
    print(f"✅ Stored {len(chunks)} document chunks with embeddings")
    return vector_store


# =============================================================================
# FUNCTION: print_ingestion_summary
# =============================================================================
def print_ingestion_summary(collection):
    """
    Print summary statistics about the ingested data.
    
    WHY THIS MATTERS:
    - Verify ingestion worked correctly
    - Understand data distribution for query design
    - Debug issues (e.g., all docs have "Unknown" severity)
    """
    print("\n" + "=" * 60)
    print("📊 INGESTION SUMMARY")
    print("=" * 60)
    
    # Count total documents
    total = collection.count_documents({})
    print(f"\nTotal chunks stored: {total}")
    
    # Count by severity
    # MongoDB aggregation pipeline - like SQL GROUP BY
    pipeline = [
        {"$group": {"_id": "$severity", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}  # Sort by count descending
    ]
    severity_counts = list(collection.aggregate(pipeline))
    
    print("\nBy Severity:")
    for item in severity_counts:
        print(f"   {item['_id']}: {item['count']} chunks")
    
    # Count by root cause category
    pipeline = [
        {"$group": {"_id": "$root_cause_category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    category_counts = list(collection.aggregate(pipeline))
    
    print("\nBy Root Cause Category:")
    for item in category_counts[:10]:  # Top 10
        print(f"   {item['_id']}: {item['count']} chunks")
    
    # Count by services (services_affected is an array)
    pipeline = [
        {"$unwind": "$services_affected"},  # Flatten array
        {"$group": {"_id": "$services_affected", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    service_counts = list(collection.aggregate(pipeline))
    
    print("\nTop 10 Services Mentioned:")
    for item in service_counts:
        print(f"   {item['_id']}: {item['count']} chunks")


# =============================================================================
# FUNCTION: create_vector_search_index
# =============================================================================
def create_vector_search_index(collection):
    """
    Create the vector search index programmatically.
    
    WHY PROGRAMMATIC INDEX CREATION:
    - Fully automated pipeline (no manual Atlas UI steps)
    - Reproducible for testing/deployment
    - Index definition is version-controlled with code
    
    WHAT THE INDEX DOES:
    - Enables vector similarity search on 'embedding' field
    - Enables pre-filtering on metadata fields (severity, services, etc.)
    
    IMPORTANT:
    - This requires MongoDB Atlas (not local MongoDB)
    - The index takes 1-2 minutes to become "Active"
    - Queries will fail until index is ready
    
    Args:
        collection: MongoDB collection object
        
    Returns:
        bool: True if index created/exists, False on error
    """
    print("\n📋 Step 6: Creating vector search index...")
    
    # Define the vector search index
    # This is the same JSON you would put in Atlas UI
    index_definition = {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1536,  # text-embedding-3-small dimensions
                "similarity": "cosine"  # cosine similarity for text
            },
            # Filter fields - enable pre-filtering before vector search
            {"type": "filter", "path": "severity"},
            {"type": "filter", "path": "services_affected"},
            {"type": "filter", "path": "root_cause_category"},
            {"type": "filter", "path": "owner_team"},
            {"type": "filter", "path": "date"},
            {"type": "filter", "path": "incident_id"},
            {"type": "filter", "path": "source_file"}
        ]
    }
    
    try:
        # Check if index already exists
        existing_indexes = list(collection.list_search_indexes())
        existing_names = [idx.get("name") for idx in existing_indexes]
        
        if INDEX_NAME in existing_names:
            print(f"   ⚠️ Index '{INDEX_NAME}' already exists")
            
            # Check if it's ready
            for idx in existing_indexes:
                if idx.get("name") == INDEX_NAME:
                    status = idx.get("status", "unknown")
                    print(f"   Status: {status}")
                    if status == "READY":
                        print("   ✅ Index is ready for queries")
                    else:
                        print("   ⏳ Index is still building...")
            return True
        
        # Create the index
        # create_search_index is available in pymongo 4.6+
        # For Vector Search, we must specify type="vectorSearch"
        print(f"   Creating index '{INDEX_NAME}'...")
        
        from pymongo.operations import SearchIndexModel
        
        search_index_model = SearchIndexModel(
            definition=index_definition,
            name=INDEX_NAME,
            type="vectorSearch"  # CRITICAL: Must specify this for vector indexes
        )
        
        result = collection.create_search_index(model=search_index_model)
        
        print(f"   ✅ Index creation initiated: {result}")
        print(f"   ⏳ Index will take 1-2 minutes to become active")
        print(f"   💡 You can check status in Atlas UI or run: python ingestion.py --check-index")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        
        # Handle common errors gracefully
        if "already exists" in error_msg.lower():
            print(f"   ⚠️ Index already exists (from previous run)")
            return True
        
        if "not authorized" in error_msg.lower() or "forbidden" in error_msg.lower():
            print(f"   ⚠️ Cannot create index programmatically (permission denied)")
            print(f"   📋 Create index manually in Atlas UI:")
            print_manual_index_instructions()
            return False
        
        if "Atlas" in error_msg or "M0" in error_msg:
            print(f"   ⚠️ Programmatic index creation not supported on this cluster tier")
            print(f"   📋 Create index manually in Atlas UI:")
            print_manual_index_instructions()
            return False
        
        # Unknown error
        print(f"   ❌ Error creating index: {e}")
        print(f"   📋 Create index manually in Atlas UI:")
        print_manual_index_instructions()
        return False


def print_manual_index_instructions():
    """
    Print manual instructions for creating the vector search index.
    Used as fallback when programmatic creation fails.
    """
    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│  MANUAL INDEX CREATION REQUIRED                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Go to: https://cloud.mongodb.com                                    │
│  2. Navigate: Your Cluster → Atlas Search → Create Search Index         │
│  3. Select "Atlas Vector Search" → "JSON Editor"                        │
│  4. Paste this configuration:                                           │
│                                                                         │
│  {{                                                                      │
│    "fields": [                                                          │
│      {{                                                                  │
│        "type": "vector",                                                │
│        "path": "embedding",                                             │
│        "numDimensions": 1536,                                           │
│        "similarity": "cosine"                                           │
│      }},                                                                 │
│      {{"type": "filter", "path": "severity"}},                           │
│      {{"type": "filter", "path": "services_affected"}},                  │
│      {{"type": "filter", "path": "root_cause_category"}},                │
│      {{"type": "filter", "path": "owner_team"}},                         │
│      {{"type": "filter", "path": "date"}},                               │
│      {{"type": "filter", "path": "incident_id"}}                         │
│    ]                                                                    │
│  }}                                                                      │
│                                                                         │
│  5. Index name: {INDEX_NAME:<50} │
│  6. Database: {DB_NAME:<53} │
│  7. Collection: {COLLECTION_NAME:<50} │
│  8. Click "Create Search Index"                                         │
│  9. Wait for status "Active" (1-2 minutes)                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """
    Main ingestion pipeline - orchestrates all steps.
    
    PIPELINE FLOW:
    1. Validate environment (fail fast if config is wrong)
    2. Load markdown files from docs directory
    3. Chunk documents into smaller pieces
    4. Set up MongoDB collection
    5. Create embeddings and store in MongoDB
    6. Print summary and next steps
    """
    print("=" * 60)
    print("🚀 Postmortem RAG - Ingestion Pipeline")
    print("=" * 60)
    
    # --- Step 0: Validate Environment ---
    print("\n📋 Step 0: Validating environment...")
    
    if not MONGO_DB_URL:
        raise ValueError(
            "❌ MONGO_DB_URL not set!\n"
            "   Add to .env: MONGO_DB_URL=mongodb+srv://..."
        )
    
    if not OPENAI_API_KEY:
        raise ValueError(
            "❌ OPENAI_API_KEY not set!\n"
            "   Add to .env: OPENAI_API_KEY=sk-..."
        )
    
    print("   ✅ Environment variables loaded")
    
    # --- Step 1: Locate Documents ---
    print("\n📋 Step 1: Locating documents...")
    
    # Get path relative to this script
    script_dir = Path(__file__).parent
    docs_dir = script_dir / "docs"
    
    if not docs_dir.exists():
        raise FileNotFoundError(
            f"❌ Docs directory not found: {docs_dir}\n"
            f"   Run convert_jira_to_markdown.py first"
        )
    
    print(f"   📁 Docs directory: {docs_dir}")
    
    # --- Step 2: Load Markdown Files ---
    print("\n📋 Step 2: Loading markdown files...")
    documents = load_markdown_files(docs_dir)
    
    if not documents:
        raise ValueError("❌ No documents loaded!")
    
    # --- Step 3: Chunk Documents ---
    print("\n📋 Step 3: Chunking documents...")
    print(f"   Chunk size: {CHUNK_SIZE} chars")
    print(f"   Chunk overlap: {CHUNK_OVERLAP} chars")
    
    chunks = chunk_documents(documents)
    
    # --- Step 4: Setup MongoDB ---
    print("\n📋 Step 4: Setting up MongoDB...")
    client, collection = setup_mongodb_collection()
    
    try:
        # --- Step 5: Create Embeddings and Store ---
        print("\n📋 Step 5: Creating embeddings and storing...")
        vector_store = create_embeddings_and_store(collection, chunks)
        
        # --- Step 6: Create Vector Search Index ---
        # This enables semantic search on the embeddings
        index_created = create_vector_search_index(collection)
        
        # --- Step 7: Print Summary ---
        print_ingestion_summary(collection)
        
        # --- Final Status ---
        print("\n" + "=" * 60)
        print("✅ INGESTION COMPLETE!")
        print("=" * 60)
        print(f"   Database: {DB_NAME}")
        print(f"   Collection: {COLLECTION_NAME}")
        print(f"   Total chunks: {len(chunks)}")
        
        if index_created:
            print(f"   Vector index: {INDEX_NAME} (creating/ready)")
        else:
            print(f"   Vector index: Manual creation required (see above)")
        
        print("\n📋 Next steps:")
        print("   1. Wait 1-2 minutes for index to become active")
        print("   2. Run retrieval.py to test queries")
        print("   3. Run generation.py for full RAG pipeline")
        
    finally:
        # Always close the MongoDB connection
        # "finally" ensures this runs even if an error occurs
        client.close()
        print("\n🔌 MongoDB connection closed")


# =============================================================================
# FUNCTION: check_index_status
# =============================================================================
def check_index_status():
    """
    Check the status of the vector search index.
    
    Useful for debugging and waiting for index to become ready.
    
    Usage: python ingestion.py --check-index
    """
    print("=" * 60)
    print("🔍 Checking Vector Search Index Status")
    print("=" * 60)
    
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        # Check document count
        doc_count = collection.count_documents({})
        print(f"\nCollection: {COLLECTION_NAME}")
        print(f"Documents: {doc_count}")
        
        # List search indexes
        print(f"\nSearch Indexes:")
        indexes = list(collection.list_search_indexes())
        
        if not indexes:
            print("   ❌ No search indexes found")
            print("   Run: python ingestion.py (to create)")
            return
        
        for idx in indexes:
            name = idx.get("name", "unknown")
            status = idx.get("status", "unknown")
            
            # Status indicators
            if status == "READY":
                status_icon = "✅"
            elif status == "PENDING" or status == "BUILDING":
                status_icon = "⏳"
            else:
                status_icon = "❓"
            
            print(f"   {status_icon} {name}: {status}")
            
            # Show index definition
            definition = idx.get("latestDefinition", idx.get("definition", {}))
            fields = definition.get("fields", [])
            
            vector_fields = [f for f in fields if f.get("type") == "vector"]
            filter_fields = [f for f in fields if f.get("type") == "filter"]
            
            print(f"      Vector fields: {len(vector_fields)}")
            print(f"      Filter fields: {len(filter_fields)}")
            
            if filter_fields:
                filter_paths = [f.get("path") for f in filter_fields]
                print(f"      Filterable: {', '.join(filter_paths)}")
        
        print()
        
    finally:
        client.close()


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # This block only runs when script is executed directly
    # Not when imported as a module
    #
    # Usage: 
    #   python ingestion.py              # Run full ingestion
    #   python ingestion.py --check-index # Check index status only
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--check-index":
        check_index_status()
    else:
        main()
