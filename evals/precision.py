"""
Postmortem RAG - Precision Evaluation
======================================

This module evaluates the RETRIEVAL quality of our RAG system.
It measures Precision@K: What fraction of retrieved documents are actually relevant?

ASSIGNMENT REQUIREMENT:
    "You must implement at least one retrieval-based eval."
    This script satisfies that requirement with Precision@K.

WHY PRECISION MATTERS:
    If we retrieve 5 documents and only 2 are relevant, precision = 2/5 = 40%
    Low precision means the LLM gets noisy context → worse answers.

EVALUATION METHOD:
    We use "LLM-as-Judge" - GPT-4o-mini judges if each retrieved document
    is relevant to the question. This is faster than human labeling and
    correlates well with human judgments.

Author: Your Name
Date: 2026-01-28
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# LANGCHAIN IMPORTS
# -----------------------------------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local modules
from config import OPENAI_API_KEY, JUDGE_MODEL, DEFAULT_TOP_K
from retrieval import retrieve, retrieve_unique, retrieve_with_scores


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_K = DEFAULT_TOP_K

# Whether to use deduplicated retrieval for evaluation
# True = more accurate metrics (one doc per incident)
# False = raw retrieval (may have duplicate incidents)
USE_DEDUPLICATED_RETRIEVAL = True


# =============================================================================
# CONCEPT: PRECISION@K
# =============================================================================
#
# WHAT IS PRECISION?
# ==================
# Precision measures: Of the documents we retrieved, how many were relevant?
#
# Formula:
#     Precision@K = (Number of Relevant Documents in Top K) / K
#
# Example:
#     Query: "database timeout issues"
#     Retrieved 5 documents:
#       1. MongoDB timeout incident ✅ Relevant
#       2. Kafka lag incident ❌ Not relevant  
#       3. Database connection pool exhaustion ✅ Relevant
#       4. Payment API timeout ⚠️ Partially relevant
#       5. Redis cache issue ❌ Not relevant
#     
#     Relevant: 2-3 out of 5
#     Precision@5 = 2/5 = 0.40 (40%)
#
# WHY PRECISION MATTERS FOR RAG:
# ==============================
# - LLM context is limited - irrelevant docs waste tokens
# - Noisy context can confuse the LLM
# - Higher precision = more focused, accurate answers
#
# PRECISION vs RECALL:
# ====================
# - Precision: Did we retrieve good documents? (quality)
# - Recall: Did we find ALL relevant documents? (coverage)
# 
# For RAG, precision is usually more important because:
# - We only need SOME relevant docs, not ALL
# - The LLM can synthesize from a few good sources
#
# WHAT'S A GOOD PRECISION?
# ========================
# - 80%+ : Excellent retrieval
# - 60-80%: Good retrieval
# - 40-60%: Needs improvement
# - <40%: Poor retrieval, noisy context


# =============================================================================
# CONCEPT: LLM-AS-JUDGE
# =============================================================================
#
# WHAT IS LLM-AS-JUDGE?
# =====================
# Instead of manually labeling documents as relevant/not relevant,
# we use an LLM to make this judgment automatically.
#
# HOW IT WORKS:
# 1. Give the LLM the question and a retrieved document
# 2. Ask: "Is this document relevant to answering the question?"
# 3. LLM responds: "RELEVANT" or "NOT_RELEVANT"
# 4. Aggregate judgments to calculate precision
#
# WHY USE LLM-AS-JUDGE:
# - Scalable: Can evaluate hundreds of queries
# - Consistent: Same criteria applied to all documents
# - Fast: Seconds vs hours of human labeling
# - Correlates well with human judgments (~80-90% agreement)
#
# LIMITATIONS:
# - Not perfect - LLM can make mistakes
# - Costs money (API calls)
# - May have biases
#
# ALTERNATIVES:
# - Human labeling (gold standard but slow/expensive)
# - Embedding similarity threshold (fast but less accurate)
# - Exact match (only works for known answers)


# =============================================================================
# JUDGE PROMPT
# =============================================================================

RELEVANCE_JUDGE_PROMPT = """You are a relevance judge for a postmortem/incident knowledge base.

Your task is to determine if the retrieved document chunk is relevant to answering the given question about incidents, outages, or operational issues.

RELEVANCE CRITERIA:
- RELEVANT: The document contains information that would help answer the question.
  This includes: root causes, symptoms, fixes, timelines, affected services, or lessons learned
  that relate to what the question is asking about.

- NOT_RELEVANT: The document does not contain useful information for the question.
  This includes: unrelated incidents, different topics, or content that doesn't address
  what's being asked.

Note: A document can be RELEVANT even if it doesn't fully answer the question,
as long as it provides useful partial information.

---

Question: {question}

Retrieved Document:
{document}

---

Is this document relevant to answering the question?
Respond with ONLY one word: "RELEVANT" or "NOT_RELEVANT"
"""


# =============================================================================
# TEST CASES
# =============================================================================
# These are questions we'll use to evaluate retrieval quality.
# Good test cases:
# - Cover different topics in your corpus
# - Have clear "right answers" (relevant documents)
# - Range from specific to general

TEST_CASES = [
    {
        "id": "database_timeout",
        "question": "What incidents were caused by database timeouts or latency issues?",
        "description": "Database performance incidents",
        "expected_categories": ["database"],  # For validation
    },
    {
        "id": "critical_incidents",
        "question": "What were the most critical production incidents and their root causes?",
        "description": "High-severity incidents",
        "filters": {"severity": "S1 - Critical Severity"},
    },
    {
        "id": "kafka_issues",
        "question": "Were there any incidents related to Kafka, message queues, or event processing?",
        "description": "Messaging/event system issues",
    },
    {
        "id": "deployment_failures",
        "question": "What incidents were caused by deployments or releases gone wrong?",
        "description": "Deployment-related incidents",
        "expected_categories": ["deployment"],
    },
    {
        "id": "payment_service",
        "question": "What issues affected the payment processing services?",
        "description": "Payment system incidents",
    },
    {
        "id": "fixes_and_mitigations",
        "question": "What corrective measures and fixes were implemented after incidents?",
        "description": "Remediation patterns",
    },
    {
        "id": "capacity_issues",
        "question": "Were there incidents caused by capacity limits, scaling, or traffic spikes?",
        "description": "Capacity/scaling incidents",
        "expected_categories": ["capacity"],
    },
    {
        "id": "network_dns",
        "question": "What incidents involved network issues, DNS problems, or connectivity failures?",
        "description": "Network-related incidents",
        "expected_categories": ["network"],
    },
    {
        "id": "mongodb_specific",
        "question": "What MongoDB-related issues have occurred and how were they resolved?",
        "description": "MongoDB-specific incidents",
    },
    {
        "id": "prevention_lessons",
        "question": "What lessons learned and preventive measures came from past incidents?",
        "description": "Operational learnings",
    },
    {
        "id": "aws problems",
        "question": "What kind of problems were related to AWS?",
        "description": "Operational learnings",
    },
]


# =============================================================================
# FUNCTION: create_relevance_judge
# =============================================================================
def create_relevance_judge():
    """
    Create the LLM chain for judging document relevance.
    
    Returns a chain that:
    1. Takes: {"question": "...", "document": "..."}
    2. Returns: "RELEVANT" or "NOT_RELEVANT"
    """
    prompt = ChatPromptTemplate.from_template(RELEVANCE_JUDGE_PROMPT)
    
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,  # Deterministic judgments
        openai_api_key=OPENAI_API_KEY
    )
    
    parser = StrOutputParser()
    
    chain = prompt | llm | parser
    return chain


# =============================================================================
# FUNCTION: judge_relevance
# =============================================================================
def judge_relevance(judge_chain, question: str, document_content: str) -> bool:
    """
    Use LLM to judge if a document is relevant to the question.
    
    Args:
        judge_chain: The LLM chain for judging
        question: The user's question
        document_content: The text content of retrieved document
    
    Returns:
        True if relevant, False otherwise
    """
    response = judge_chain.invoke({
        "question": question,
        "document": document_content
    })
    
    # Parse response - look for RELEVANT but not NOT_RELEVANT
    response_upper = response.upper().strip()
    
    if "NOT_RELEVANT" in response_upper or "NOT RELEVANT" in response_upper:
        return False
    elif "RELEVANT" in response_upper:
        return True
    else:
        # Ambiguous response - default to not relevant
        print(f"    ⚠️ Ambiguous judge response: {response[:50]}")
        return False


# =============================================================================
# FUNCTION: evaluate_precision
# =============================================================================
def evaluate_precision(
    question: str,
    k: int = DEFAULT_K,
    filters: dict = None,
    verbose: bool = False
) -> dict:
    """
    Evaluate Precision@K for a single query.
    
    STEPS:
    1. Retrieve top-K documents for the question
    2. For each document, ask LLM: "Is this relevant?"
    3. Count relevant documents
    4. Calculate precision = relevant / k
    
    Args:
        question: The test question
        k: Number of documents to retrieve
        filters: Optional metadata filters
        verbose: Print detailed output
    
    Returns:
        Dictionary with precision score and details
    """
    # --- Step 1: Retrieve Documents ---
    filter_kwargs = filters or {}
    
    # Use deduplicated retrieval for more accurate metrics
    if USE_DEDUPLICATED_RETRIEVAL:
        documents = retrieve_unique(
            query=question,
            top_k=k,
            merge=False,  # Keep best chunk only for eval
            **filter_kwargs
        )
    else:
        documents = retrieve(
            query=question,
            top_k=k,
            **filter_kwargs
        )
    
    if not documents:
        return {
            "question": question,
            "k": k,
            "retrieved": 0,
            "relevant": 0,
            "precision": 0.0,
            "judgments": [],
            "error": "No documents retrieved"
        }
    
    # --- Step 2: Judge Each Document ---
    judge = create_relevance_judge()
    judgments = []
    relevant_count = 0
    
    for i, doc in enumerate(documents):
        # Judge relevance
        is_relevant = judge_relevance(judge, question, doc.page_content)
        
        if is_relevant:
            relevant_count += 1
        
        judgment = {
            "doc_index": i + 1,
            "relevant": is_relevant,
            "incident_id": doc.metadata.get("incident_id", "Unknown"),
            "severity": doc.metadata.get("severity", "Unknown"),
            "category": doc.metadata.get("root_cause_category", "Unknown"),
        }
        
        if verbose:
            status = "✅" if is_relevant else "❌"
            print(f"    {status} Doc {i+1}: {judgment['incident_id']} ({judgment['category']})")
        
        judgments.append(judgment)
    
    # --- Step 3: Calculate Precision ---
    precision = relevant_count / len(documents)
    
    return {
        "question": question,
        "k": k,
        "retrieved": len(documents),
        "relevant": relevant_count,
        "precision": precision,
        "judgments": judgments,
        "filters": filters
    }


# =============================================================================
# FUNCTION: run_evaluation
# =============================================================================
def run_evaluation(
    test_cases: list = None,
    k: int = DEFAULT_K,
    verbose: bool = True
) -> dict:
    """
    Run precision evaluation on all test cases.
    
    Args:
        test_cases: List of test case dicts (uses default if None)
        k: Number of documents to retrieve
        verbose: Print detailed progress
    
    Returns:
        Summary dict with aggregate metrics and per-query results
    """
    if test_cases is None:
        test_cases = TEST_CASES
    
    print("=" * 70)
    print("📊 PRECISION EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  • K (documents per query): {k}")
    print(f"  • Judge model: {JUDGE_MODEL}")
    print(f"  • Test cases: {len(test_cases)}")
    print(f"  • Timestamp: {datetime.now().isoformat()}")
    print("\n" + "-" * 70)
    
    results = []
    total_relevant = 0
    total_retrieved = 0
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        filters = test_case.get("filters", {})
        
        print(f"\n[{i}/{len(test_cases)}] {test_case['id']}")
        print(f"    Q: {question[:60]}...")
        if filters:
            print(f"    Filters: {filters}")
        
        # Evaluate this query
        result = evaluate_precision(
            question=question,
            k=k,
            filters=filters,
            verbose=verbose
        )
        
        result["test_id"] = test_case["id"]
        result["description"] = test_case.get("description", "")
        
        results.append(result)
        total_relevant += result["relevant"]
        total_retrieved += result["retrieved"]
        
        # Print precision
        precision_pct = result["precision"] * 100
        if precision_pct >= 60:
            status = "✅"
        elif precision_pct >= 40:
            status = "⚠️"
        else:
            status = "❌"
        
        print(f"    {status} Precision@{k}: {precision_pct:.1f}% ({result['relevant']}/{result['retrieved']})")
    
    # --- Aggregate Metrics ---
    avg_precision = sum(r["precision"] for r in results) / len(results) if results else 0
    overall_precision = total_relevant / total_retrieved if total_retrieved > 0 else 0
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "k": k,
            "judge_model": JUDGE_MODEL,
            "num_test_cases": len(test_cases)
        },
        "aggregate_metrics": {
            "average_precision": avg_precision,
            "overall_precision": overall_precision,
            "total_documents_retrieved": total_retrieved,
            "total_relevant_documents": total_relevant,
        },
        "per_query_results": results
    }
    
    # --- Print Summary ---
    print("\n" + "=" * 70)
    print("📈 EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  AGGREGATE METRICS                                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Average Precision@{k}:     {avg_precision:.1%}                              │
│  Overall Precision:        {overall_precision:.1%} ({total_relevant}/{total_retrieved} relevant)         │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print("Per-Query Breakdown:")
    for r in results:
        precision_pct = r["precision"] * 100
        if precision_pct >= 60:
            status = "✅"
        elif precision_pct >= 40:
            status = "⚠️"
        else:
            status = "❌"
        print(f"  {status} {r['test_id']:<25} {precision_pct:>5.1f}%")
    
    # Interpretation
    print("\n" + "-" * 70)
    print("📋 INTERPRETATION:")
    if avg_precision >= 0.6:
        print("  ✅ Good retrieval quality - most retrieved documents are relevant")
    elif avg_precision >= 0.4:
        print("  ⚠️ Moderate retrieval quality - consider improving chunking or embeddings")
    else:
        print("  ❌ Poor retrieval quality - significant noise in retrieved documents")
    
    return summary


# =============================================================================
# FUNCTION: compare_filtered_vs_unfiltered
# =============================================================================
def compare_filtered_vs_unfiltered(k: int = DEFAULT_K) -> dict:
    """
    Compare precision between filtered and unfiltered retrieval.
    
    This demonstrates the VALUE of Metadata-Filtered RAG:
    - Unfiltered: Search all documents
    - Filtered: Pre-filter by metadata, then search
    
    Expected result: Filtered should have higher precision.
    """
    print("=" * 70)
    print("📊 FILTERED vs UNFILTERED COMPARISON")
    print("=" * 70)
    
    # Test cases that have natural filters
    comparison_cases = [
        {
            "question": "What critical incidents affected database systems?",
            "unfiltered": {},
            "filtered": {
                "severity": "S1 - Critical Severity",
                "root_cause_category": "database"
            }
        },
        {
            "question": "What deployment issues occurred recently?",
            "unfiltered": {},
            "filtered": {
                "root_cause_category": "deployment"
            }
        },
        {
            "question": "What capacity or scaling issues have we seen?",
            "unfiltered": {},
            "filtered": {
                "root_cause_category": "capacity"
            }
        },
    ]
    
    results = []
    
    for case in comparison_cases:
        question = case["question"]
        print(f"\n📝 {question[:50]}...")
        
        # Unfiltered
        unfiltered_result = evaluate_precision(
            question=question,
            k=k,
            filters=case["unfiltered"],
            verbose=False
        )
        
        # Filtered
        filtered_result = evaluate_precision(
            question=question,
            k=k,
            filters=case["filtered"],
            verbose=False
        )
        
        improvement = filtered_result["precision"] - unfiltered_result["precision"]
        
        print(f"    Unfiltered: {unfiltered_result['precision']:.1%}")
        print(f"    Filtered:   {filtered_result['precision']:.1%}")
        print(f"    Improvement: {improvement:+.1%}")
        
        results.append({
            "question": question,
            "unfiltered_precision": unfiltered_result["precision"],
            "filtered_precision": filtered_result["precision"],
            "improvement": improvement,
            "filters_used": case["filtered"]
        })
    
    # Summary
    avg_improvement = sum(r["improvement"] for r in results) / len(results)
    
    print("\n" + "-" * 70)
    print(f"📈 Average Precision Improvement with Filters: {avg_improvement:+.1%}")
    
    if avg_improvement > 0:
        print("✅ Metadata filtering improves retrieval precision!")
    else:
        print("⚠️ Filtering didn't help - may need better filter selection")
    
    return {
        "comparison_results": results,
        "average_improvement": avg_improvement
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    """
    Run the precision evaluation.
    
    Usage:
        python evals/precision.py              # Full evaluation
        python evals/precision.py --compare    # Compare filtered vs unfiltered
        python evals/precision.py -k 3         # Evaluate with K=3
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval precision for Postmortem RAG"
    )
    parser.add_argument(
        "-k", type=int, default=DEFAULT_K,
        help=f"Number of documents to retrieve (default: {DEFAULT_K})"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare filtered vs unfiltered retrieval"
    )
    parser.add_argument(
        "--output", type=str,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Less verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    
    # Run evaluation
    if args.compare:
        results = compare_filtered_vs_unfiltered(k=args.k)
    else:
        results = run_evaluation(k=args.k, verbose=not args.quiet)
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
