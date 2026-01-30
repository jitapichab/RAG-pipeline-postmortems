# Postmortem RAG Pipeline

A **Metadata-Filtered RAG** system for querying incident postmortems.

## Structure

```
RAG-pipeline-postmortems/
├── config.py        # Configuration
├── utils.py         # Shared utilities (MongoDB client)
├── ingestion.py     # Load, chunk, embed, store
├── retrieval.py     # Vector search with filters
├── generation.py    # RAG pipeline
├── evals/           # Evaluations
│   ├── precision.py
│   ├── groundedness.py
│   └── recall.py
├── scripts/         # Utility scripts
└── docs/            # Postmortem documents
```

## Setup

```bash
# Install with PDM
pdm install

# Or with pip
pip install -r requirements.txt

# Set environment variables
export MONGO_DB_URL="mongodb+srv://..."
export OPENAI_API_KEY="sk-..."
```

## Usage

```bash
# 1. Ingest documents
python ingestion.py

# 2. Query
python generation.py "What caused database issues?"

# 3. Evaluate
python evals/precision.py
python evals/groundedness.py
python evals/recall.py
```
