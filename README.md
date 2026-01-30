# RAG Pipeline - Postmortems

A **Metadata-Filtered RAG** system for querying incident postmortems from JIRA.

## Overview

This pipeline implements Retrieval-Augmented Generation (RAG) to answer questions about production incidents using company postmortem documentation.

### RAG Pattern: Metadata-Filtered

Instead of searching ALL documents, we first filter by metadata (severity, services, root cause) and THEN perform vector search.

```
User Query → Metadata Pre-Filter → Vector Search → LLM Generation → Answer
```

## Project Structure

```
RAG-pipeline-postmortems/
├── config.py           # Centralized configuration
├── utils.py            # Shared utility functions
├── __init__.py         # Package definition
├── ingestion.py        # Load, chunk, embed, store documents
├── retrieval.py        # Vector search with metadata filtering
├── generation.py       # RAG pipeline with LLM
├── convert_jira_to_markdown.py  # JIRA → Markdown converter
├── requirements.txt    # Python dependencies
├── docs/               # Postmortem markdown files
└── evals/              # Evaluation metrics
    ├── precision.py    # Precision@K (retrieval quality)
    ├── groundedness.py # Answer faithfulness
    └── recall.py       # Retrieval coverage
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MONGO_DB_URL="mongodb+srv://..."
export OPENAI_API_KEY="sk-..."

# Run ingestion
python ingestion.py

# Run generation
python generation.py

# Run evaluations
python evals/precision.py
python evals/groundedness.py
python evals/recall.py
```

## Key Features

- **Metadata Filtering**: Filter by severity, root cause, services, date
- **Deduplicated Retrieval**: `retrieve_unique()` returns one doc per incident
- **Chunk Merging**: Combines chunks from same incident for complete context
- **Three Evaluations**: Precision, Groundedness, Recall

## Author

Jorge Tapicha - 2026
