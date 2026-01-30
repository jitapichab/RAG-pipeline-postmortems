"""
Postmortem RAG - Recall Evaluation
===================================

This module evaluates RECALL: Did we retrieve all the relevant documents?

WHAT IS RECALL?
===============
Recall measures coverage - out of all relevant documents in the corpus,
how many did we successfully retrieve?

Formula:
    Recall@K = (Relevant Documents Retrieved) / (Total Relevant Documents in Corpus)

Example:
    Query: "MongoDB issues"
    Corpus has 8 relevant incidents
    We retrieved 5 docs, 4 were relevant
    Recall = 4/8 = 50% (we found half of the relevant docs)

WHY RECALL MATTERS:
===================
- High precision but low recall = missing important information
- Critical for comprehensive answers
- Ensures we don't miss key incidents when searching

RECALL vs PRECISION:
====================
┌─────────────────────────────────────────────────────────────────────────┐
│  High Precision, Low Recall:                                           │
│  "We got 5 docs, all relevant, but missed 15 other relevant docs"      │
│  → Good quality, bad coverage                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Low Precision, High Recall:                                           │
│  "We got 20 docs, only 8 relevant, but found all relevant docs"        │
│  → Noisy results, good coverage                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  High Precision, High Recall:                                          │
│  "We got 10 docs, 8 relevant, found all relevant docs"                 │
│  → Ideal! Good quality AND coverage                                    │
└─────────────────────────────────────────────────────────────────────────┘

HOW TO MEASURE RECALL:
======================
Unlike precision (where we can use LLM-as-judge), recall requires
GROUND TRUTH - we must know which documents ARE relevant for each query.

This script uses pre-defined ground truth labels based on:
- Incident categories (root_cause_category)
- Incident IDs known to be relevant
- Manual inspection of the corpus

Author: Jorge Tapicha
Date: 2026-01-28
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local modules
from config import OPENAI_API_KEY
from retrieval import retrieve_unique


# =============================================================================
# CONFIGURATION
# =============================================================================

# Use K=10 for better recall coverage
DEFAULT_K = 10


# =============================================================================
# GROUND TRUTH LABELS
# =============================================================================
# For each test query, we define which incidents are considered "relevant"
# This is the "gold standard" that recall is measured against.
#
# HOW THESE WERE CREATED:
# - Based on root_cause_category metadata
# - Based on text content inspection
# - Based on incident titles and descriptions
#
# NOTE: This is a simplified ground truth. In production, you would:
# - Have domain experts label relevance
# - Use multiple annotators and measure agreement
# - Create larger, more comprehensive label sets

TEST_CASES_WITH_GROUND_TRUTH = [
    {
        "id": "database_incidents",
        "question": "What incidents were caused by database issues like MongoDB, Aurora, or Postgres?",
        "description": "Database-related incidents",
        # Use metadata filter to find database-category incidents
        "filters": {"root_cause_category": "database"},
        "relevant_incident_ids": [
            "PFU-111", "PFU-112", "PFU-119", "PFU-141", "PFU-148",
            "PFU-170", "PFU-177", "PFU-197", "PFU-294", "PFU-295",
        ],
    },
    {
        "id": "deployment_incidents",
        "question": "What incidents were caused by deployments, releases, or rollouts?",
        "description": "Deployment-related incidents",
        "filters": {"root_cause_category": "deployment"},
        "relevant_incident_ids": [
            "PFU-110", "PFU-122", "PFU-123", "PFU-124", "PFU-135",
            "PFU-149", "PFU-160", "PFU-183", "PFU-237", "PFU-266",
        ],
    },
    {
        "id": "network_timeout_incidents",
        "question": "What incidents involved network issues, DNS, timeouts, or connectivity problems?",
        "description": "Network/connectivity incidents",
        "filters": {"root_cause_category": "network"},
        "relevant_incident_ids": [
            "PFU-128", "PFU-154", "PFU-161", "PFU-172", "PFU-191",
        ],
    },
    {
        "id": "critical_severity_incidents",
        "question": "What were the most critical (S1) production incidents?",
        "description": "S1 Critical incidents",
        # Use severity filter for categorical query
        "filters": {"severity": "S1 - Critical Severity"},
        "relevant_incident_ids": [
            "PFU-109", "PFU-112", "PFU-123", "PFU-170", "PFU-177",
            "PFU-191", "PFU-194", "PFU-197", "PFU-213", "PFU-239",
            "PFU-287", "PFU-290",
        ],
    },
    {
        "id": "payment_service_incidents",
        "question": "What issues affected the payment-rx-orc or payment services?",
        "description": "Payment service incidents - semantic search",
        # No filter - test semantic search for service names
        "relevant_incident_ids": [
            "PFU-170", "PFU-191", "PFU-239", "PFU-287", "PFU-294", "PFU-295",
        ],
    },
    {
        "id": "kafka_messaging_incidents",
        "question": "What incidents involved Kafka, message queues, or event processing?",
        "description": "Kafka/messaging incidents",
        # Small ground truth - good for semantic search
        "relevant_incident_ids": [
            "PFU-123", "PFU-125", "PFU-289",
        ],
    },
    {
        "id": "capacity_scaling_incidents",
        "question": "What incidents were caused by capacity limits, traffic spikes, or scaling issues?",
        "description": "Capacity/scaling incidents",
        "filters": {"root_cause_category": "capacity"},
        "relevant_incident_ids": [
            "PFU-109", "PFU-114", "PFU-150", "PFU-187", "PFU-211", "PFU-247",
        ],
    },
]


# =============================================================================
# FUNCTION: calculate_recall
# =============================================================================
def calculate_recall(
    question: str,
    relevant_ids: list[str],
    k: int = DEFAULT_K,
    filters: dict = None,
    verbose: bool = False
) -> dict:
    """
    Calculate Recall@K for a single query.
    
    STEPS:
    1. Retrieve top-K documents
    2. Extract incident IDs from retrieved docs
    3. Compare to ground truth relevant IDs
    4. Calculate recall = intersection / total_relevant
    
    Args:
        question: The test question
        relevant_ids: Ground truth - list of relevant incident IDs
        k: Number of documents to retrieve
        filters: Optional metadata filters
        verbose: Print detailed output
    
    Returns:
        Dictionary with recall score and details
    """
    filter_kwargs = filters or {}
    
    # --- Step 1: Retrieve Documents ---
    # Use retrieve_unique to get deduplicated results (one doc per incident)
    documents = retrieve_unique(
        query=question,
        top_k=k,
        merge=False,  # Don't need merged content for recall
        **filter_kwargs
    )
    
    if not documents:
        return {
            "question": question,
            "k": k,
            "retrieved_count": 0,
            "relevant_found": 0,
            "total_relevant": len(relevant_ids),
            "recall": 0.0,
            "retrieved_ids": [],
            "relevant_ids": relevant_ids,
            "found_ids": [],
            "missed_ids": relevant_ids,
        }
    
    # --- Step 2: Extract Retrieved Incident IDs ---
    # With retrieve_unique, each doc is already a unique incident
    retrieved_ids = [
        doc.metadata.get("incident_id", "Unknown")
        for doc in documents
    ]
    
    # --- Step 3: Calculate Recall ---
    # Which relevant IDs did we find?
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)
    
    found_ids = relevant_set & retrieved_set  # Intersection
    missed_ids = relevant_set - retrieved_set  # Relevant but not retrieved
    
    recall = len(found_ids) / len(relevant_ids) if relevant_ids else 0.0
    
    if verbose:
        print(f"    📄 Retrieved {len(retrieved_ids)} unique incidents")
        print(f"    ✅ Found {len(found_ids)}/{len(relevant_ids)} relevant")
        print(f"    ❌ Missed: {list(missed_ids)[:5]}{'...' if len(missed_ids) > 5 else ''}")
    
    return {
        "question": question,
        "k": k,
        "retrieved_count": len(retrieved_ids),
        "relevant_found": len(found_ids),
        "total_relevant": len(relevant_ids),
        "recall": recall,
        "retrieved_ids": list(retrieved_ids),
        "relevant_ids": relevant_ids,
        "found_ids": list(found_ids),
        "missed_ids": list(missed_ids),
    }


# =============================================================================
# FUNCTION: run_recall_evaluation
# =============================================================================
def run_recall_evaluation(
    test_cases: list = None,
    k: int = DEFAULT_K,
    verbose: bool = True
) -> dict:
    """
    Run recall evaluation on all test cases.
    
    Args:
        test_cases: List of test cases with ground truth
        k: Number of documents to retrieve
        verbose: Print detailed progress
    
    Returns:
        Summary dict with aggregate metrics
    """
    if test_cases is None:
        test_cases = TEST_CASES_WITH_GROUND_TRUTH
    
    print("=" * 70)
    print("📊 RECALL EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  • K (documents per query): {k}")
    print(f"  • Test cases: {len(test_cases)}")
    print(f"  • Timestamp: {datetime.now().isoformat()}")
    print("\n" + "-" * 70)
    
    results = []
    total_found = 0
    total_relevant = 0
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        relevant_ids = test_case["relevant_incident_ids"]
        filters = test_case.get("filters", {})
        
        print(f"\n[{i}/{len(test_cases)}] {test_case['id']}")
        print(f"    Q: {question[:55]}...")
        print(f"    Ground truth: {len(relevant_ids)} relevant incidents")
        if filters:
            print(f"    Filters: {filters}")
        
        result = calculate_recall(
            question=question,
            relevant_ids=relevant_ids,
            k=k,
            filters=filters,
            verbose=verbose
        )
        
        result["test_id"] = test_case["id"]
        result["description"] = test_case.get("description", "")
        
        results.append(result)
        total_found += result["relevant_found"]
        total_relevant += result["total_relevant"]
        
        # Print recall
        recall_pct = result["recall"] * 100
        if recall_pct >= 50:
            status = "✅"
        elif recall_pct >= 25:
            status = "⚠️"
        else:
            status = "❌"
        
        print(f"    {status} Recall@{k}: {recall_pct:.1f}% ({result['relevant_found']}/{result['total_relevant']})")
    
    # --- Aggregate Metrics ---
    avg_recall = sum(r["recall"] for r in results) / len(results) if results else 0
    overall_recall = total_found / total_relevant if total_relevant > 0 else 0
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "k": k,
            "num_test_cases": len(test_cases)
        },
        "aggregate_metrics": {
            "average_recall": avg_recall,
            "overall_recall": overall_recall,
            "total_relevant_found": total_found,
            "total_relevant_expected": total_relevant,
        },
        "per_query_results": results
    }
    
    # --- Print Summary ---
    print("\n" + "=" * 70)
    print("📈 RECALL SUMMARY")
    print("=" * 70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  AGGREGATE METRICS                                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Average Recall@{k}:       {avg_recall:.1%}                                 │
│  Overall Recall:          {overall_recall:.1%} ({total_found}/{total_relevant} relevant found)      │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print("Per-Query Breakdown:")
    for r in results:
        recall_pct = r["recall"] * 100
        if recall_pct >= 50:
            status = "✅"
        elif recall_pct >= 25:
            status = "⚠️"
        else:
            status = "❌"
        print(f"  {status} {r['test_id']:<30} {recall_pct:>5.1f}% ({r['relevant_found']}/{r['total_relevant']})")
    
    # Interpretation
    print("\n" + "-" * 70)
    print("📋 INTERPRETATION:")
    print(f"  Note: Recall@{k} measures how many relevant docs we find in top {k} results.")
    print(f"  With K={k} and many relevant docs, recall will naturally be lower.")
    
    if avg_recall >= 0.4:
        print(f"  ✅ Good coverage - finding significant portion of relevant docs")
    elif avg_recall >= 0.2:
        print(f"  ⚠️ Moderate coverage - consider increasing K for comprehensive results")
    else:
        print(f"  ❌ Low coverage - many relevant docs not being retrieved")
        print(f"  💡 Consider: larger K, better embeddings, or query expansion")
    
    return summary


# =============================================================================
# FUNCTION: analyze_recall_vs_k
# =============================================================================
def analyze_recall_vs_k(k_values: list = None) -> dict:
    """
    Analyze how recall changes with different K values.
    
    This helps understand the trade-off:
    - Higher K = Higher recall (find more relevant docs)
    - Higher K = Lower precision (more noise)
    - Higher K = Higher cost (more context for LLM)
    """
    if k_values is None:
        k_values = [3, 5, 10, 15, 20]
    
    print("=" * 70)
    print("📊 RECALL vs K ANALYSIS")
    print("=" * 70)
    print(f"\nAnalyzing recall for K values: {k_values}")
    print("-" * 70)
    
    results_by_k = {}
    
    for k in k_values:
        print(f"\n📍 K = {k}")
        summary = run_recall_evaluation(k=k, verbose=False)
        results_by_k[k] = {
            "average_recall": summary["aggregate_metrics"]["average_recall"],
            "overall_recall": summary["aggregate_metrics"]["overall_recall"],
        }
        print(f"   Average Recall: {results_by_k[k]['average_recall']:.1%}")
    
    # Summary chart
    print("\n" + "=" * 70)
    print("📈 RECALL CURVE")
    print("=" * 70)
    print(f"\n{'K':<5} {'Avg Recall':<15} {'Bar'}")
    print("-" * 50)
    
    for k in k_values:
        recall = results_by_k[k]["average_recall"]
        bar_length = int(recall * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"{k:<5} {recall:.1%}          {bar}")
    
    return results_by_k


# =============================================================================
# MAIN
# =============================================================================
def main():
    """
    Run the recall evaluation.
    
    Usage:
        python evals/recall.py              # Full evaluation
        python evals/recall.py -k 10        # With K=10
        python evals/recall.py --analyze    # Analyze recall vs K
        python evals/recall.py --output results.json
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate recall for Postmortem RAG"
    )
    parser.add_argument(
        "-k", type=int, default=DEFAULT_K,
        help=f"Number of documents to retrieve (default: {DEFAULT_K})"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Analyze recall across different K values"
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
    if args.analyze:
        results = analyze_recall_vs_k()
    else:
        results = run_recall_evaluation(k=args.k, verbose=not args.quiet)
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    print("\n✅ Recall evaluation complete!")


if __name__ == "__main__":
    main()
