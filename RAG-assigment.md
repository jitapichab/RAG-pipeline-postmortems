# Build a Company-Specific RAG Pipeline

**Catalyst 3**

## Reference Material

Before starting, review the RAG Cookbook for examples of different RAG patterns. You are encouraged to use it as inspiration for architecture and tradeoffs, not as a strict template.

## Overview

In this assignment, you will design and implement a Retrieval-Augmented Generation pipeline using real documentation from your company.

**Goal:** Practice turning unstructured company knowledge into a reliable retrieval system and evaluate how well your retrieval pipeline performs.

> This is an applied systems assignment focused on design decisions, retrieval behavior, and evaluation — not UI or polish.

## Due Date

**Thursday, January 29th at 11:59 PM CT**

---

## Dataset Requirements

Use company-specific documents, such as:

- Internal wiki or SOPs
- API or developer documentation
- Product docs
- Engineering blogs or knowledge bases

**Target size:** 50 to 100 documents

**Notes:**
- Dataset quality is not graded. Focus on structure and retrieval behavior.
- If documents are sensitive, you do not need to share the raw data with us.

---

## RAG Pipeline Requirements

You must implement a working RAG pipeline that includes:

1. **Document ingestion**
2. **Chunking strategy**
3. **Embedding and indexing** into a vector database
4. **Retrieval**
5. **Generation** using retrieved context

### Vector Database

You may use any vector database. Recommended options:

- MongoDB Atlas Vector Search
- Pinecone

### Framework

You may use any framework, including:

- LangChain
- Vercel AI SDK
- OpenAI Agents SDK
- Anthropic Agents SDK
- A custom or lightweight implementation

> No client or UI is required. The system can run locally, via scripts, or as an API.

---

## RAG Pattern Requirement

Implement **at least one** RAG pattern:

| Pattern | Description |
|---------|-------------|
| Naive RAG | Basic chunking → embedding → vector search |
| Metadata-filtered RAG | Pre-filter by metadata before vector search |
| Hybrid search | BM25 + vector search combined |
| Graph RAG | Knowledge graph + vector search |
| Agentic RAG | Agent dynamically chooses retrieval strategy |

**You must clearly explain which pattern you chose and why it fits your documents and use case.**

---

## Eval Requirement

You must implement **at least one** retrieval-based eval.

### Allowed Evals

- **Precision** — How many retrieved docs are relevant?
- **Recall** — How many relevant docs were retrieved?
- **Groundedness** — Is the response grounded in retrieved context?

> Advanced ranking metrics like MRR or nDCG are not required.

**Important:** The eval must explicitly measure retrieval behavior, not just the final LLM response.

---

## Submission Requirements

### 1. Code

Submit code that shows how your RAG pipeline is constructed, including:

- Ingestion and chunking logic
- Retrieval logic
- Eval logic

> The code does not need to be fully runnable end-to-end, since you may not want to expose internal documents or credentials. Clarity and structure matter more than execution.

### 2. Video Walkthrough

Submit a **3 to 5 minute video** explaining:

1. How your pipeline is structured
2. Your chunking and retrieval decisions
3. The RAG pattern you implemented
4. The eval you chose and why

> A live demo is optional.

---

## Grading Criteria

| Criteria | Weight |
|----------|--------|
| Clear RAG pipeline design | ✓ |
| Correct use of a RAG pattern | ✓ |
| Proper implementation of a retrieval-based eval | ✓ |
| Thoughtful reasoning about tradeoffs | ✓ |
| Code clarity and organization | ✓ |

**Not graded:** Dataset quality, UI, and deployment.

---

*Ready to face the Gauntlet?*

**Gauntlet AI** — Where only the strongest AI-first engineers emerge.
