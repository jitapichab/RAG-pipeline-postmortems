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

## Results

### Final Numbers

| Metric | Initial | Final | Improvement |
|--------|---------|-------|-------------|
| Precision@5 | ~50% | 78.2% | +28% |
| Recall@15 | ~30% | 85.1% | +55% |

### How I got there

**Precision: 50% → 78%**

Started with pure semantic search. The problem was obvious - asking for "database incidents" returned anything with similar words, not actual database-related postmortems.

The fix: add metadata filters. During ingestion, each chunk gets `root_cause_category`, `severity`, etc. from the YAML frontmatter. At query time, I pre-filter by category before running vector search.

```python
# Before: semantic only
docs = vector_store.similarity_search("database issues", k=5)

# After: filter first, then semantic
docs = vector_store.similarity_search(
    "database issues",
    k=5,
    pre_filter={"root_cause_category": {"$eq": "database"}}
)
```

Queries with filters now hit 100% precision. The ones that still use semantic-only (like "kafka problems") sit around 60%.

**Recall: 30% → 85%**

Two problems here:

1. **K was too small.** With K=5 and 20 database incidents in the corpus, max possible recall was 25%. Bumped K to 15.

2. **Ground truth was wrong.** I was comparing against a list of incident IDs I thought were relevant, but hadn't verified. Ran a quick query on MongoDB to get the actual IDs per category:

```python
# Get real ground truth
db.find({"root_cause_category": "database"}).distinct("incident_id")
```

After fixing both, recall jumped from 30% to 85%. Configuration and network categories hit 100% - every relevant incident found.

### Tradeoffs

The K value is the main lever:
- K=5 for precision (78%)
- K=15 for recall (85%)

For postmortem analysis, I prioritize recall. Missing a past incident means missing context that could prevent the next outage. The LLM can handle a few extra docs in the prompt.

### Filters indexed

```python
{"type": "filter", "path": "severity"}
{"type": "filter", "path": "root_cause_category"}
{"type": "filter", "path": "services_affected"}
{"type": "filter", "path": "owner_team"}
{"type": "filter", "path": "incident_id"}
```
