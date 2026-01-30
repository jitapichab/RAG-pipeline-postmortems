"""
Postmortem RAG - Groundedness Evaluation
=========================================

This module evaluates whether the LLM's answers are GROUNDED in the retrieved context.
Groundedness measures: Is the answer supported by the provided documents?

WHAT IS GROUNDEDNESS?
=====================
When we give an LLM context and ask a question, the LLM should:
- Base its answer on the provided context
- NOT make up information (hallucinate)
- Acknowledge when information is missing

Groundedness Score:
- 1.0 (100%): Every claim in the answer is supported by the context
- 0.5 (50%): Some claims are supported, some are not
- 0.0 (0%): Answer is completely unsupported/hallucinated

WHY GROUNDEDNESS MATTERS:
=========================
- RAG is designed to reduce hallucination
- If the LLM ignores context and makes things up, RAG fails
- Groundedness measures if RAG is working as intended

EVALUATION METHOD:
==================
We use LLM-as-Judge to evaluate groundedness:
1. Generate an answer using RAG
2. Ask a judge LLM: "Is this answer supported by the context?"
3. The judge checks each claim against the source documents

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

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local modules
from config import OPENAI_API_KEY, JUDGE_MODEL, DEFAULT_TOP_K
from retrieval import retrieve_unique, format_context
from generation import generate_answer


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_K = DEFAULT_TOP_K


# =============================================================================
# CONCEPT: GROUNDEDNESS
# =============================================================================
#
# GROUNDEDNESS vs PRECISION vs RELEVANCE:
# ========================================
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  METRIC        │  MEASURES                    │  EVALUATES             │
# ├─────────────────────────────────────────────────────────────────────────┤
# │  Precision     │  Retrieval quality           │  Retrieved documents   │
# │                │  "Are retrieved docs useful?"│                        │
# ├─────────────────────────────────────────────────────────────────────────┤
# │  Groundedness  │  Answer faithfulness         │  Generated answer      │
# │                │  "Is answer from context?"   │                        │
# ├─────────────────────────────────────────────────────────────────────────┤
# │  Relevance     │  Answer usefulness           │  Generated answer      │
# │                │  "Does answer address Q?"    │                        │
# └─────────────────────────────────────────────────────────────────────────┘
#
# EXAMPLES:
# =========
#
# HIGH GROUNDEDNESS (Good):
#   Context: "The incident was caused by missing MongoDB indexes."
#   Question: "What caused the incident?"
#   Answer: "The incident was caused by missing MongoDB indexes."
#   → Groundedness: 1.0 (answer directly from context)
#
# LOW GROUNDEDNESS (Hallucination):
#   Context: "The incident was caused by missing MongoDB indexes."
#   Question: "What caused the incident?"
#   Answer: "The incident was caused by a DDoS attack on the servers."
#   → Groundedness: 0.0 (made up, not in context)
#
# PARTIAL GROUNDEDNESS:
#   Context: "The incident was caused by missing MongoDB indexes."
#   Question: "What caused the incident and how long did it last?"
#   Answer: "The incident was caused by missing MongoDB indexes. It lasted 2 hours."
#   → Groundedness: 0.5 (first part supported, "2 hours" made up)


# =============================================================================
# GROUNDEDNESS JUDGE PROMPT
# =============================================================================

GROUNDEDNESS_JUDGE_PROMPT = """You are an expert evaluator assessing whether an AI assistant's answer is grounded in the provided context.

TASK:
Evaluate if the answer is fully supported by the source documents (context).
An answer is grounded if every factual claim can be traced back to the context.

EVALUATION CRITERIA:
- FULLY_GROUNDED: All claims in the answer are directly supported by the context
- PARTIALLY_GROUNDED: Some claims are supported, but some are not or are inferred
- NOT_GROUNDED: The answer contains significant information not found in context

IMPORTANT:
- Generic statements like "based on the postmortems..." are acceptable
- Reasonable inferences from context are acceptable
- Made-up specifics (dates, numbers, names not in context) are NOT grounded
- "I don't have information" when context lacks info is GROUNDED (honest)

---

CONTEXT (Source Documents):
{context}

---

QUESTION: {question}

---

ANSWER TO EVALUATE:
{answer}

---

Evaluate the groundedness of this answer.

First, briefly analyze which parts are/aren't supported by the context.
Then provide your final verdict.

Format your response EXACTLY as:
ANALYSIS: [Your brief analysis]
VERDICT: [FULLY_GROUNDED, PARTIALLY_GROUNDED, or NOT_GROUNDED]
SCORE: [1.0, 0.5, or 0.0]
"""


# =============================================================================
# TEST CASES FOR GROUNDEDNESS
# =============================================================================
# These questions should have answers in the postmortem corpus

TEST_CASES = [
    {
        "id": "root_cause_specific",
        "question": "What was the root cause of the MongoDB latency incident?",
        "description": "Specific incident root cause",
    },
    {
        "id": "prevention_measures",
        "question": "What preventive measures were implemented after database incidents?",
        "description": "Remediation actions",
    },
    {
        "id": "service_impact",
        "question": "Which services were affected by critical incidents?",
        "description": "Service impact analysis",
    },
    {
        "id": "timeline_question",
        "question": "How quickly were database incidents typically resolved?",
        "description": "Incident resolution time",
    },
    {
        "id": "pattern_question",
        "question": "What patterns do you see in deployment-related incidents?",
        "description": "Pattern analysis",
    },
    {
        "id": "unanswerable",
        "question": "What was the exact revenue impact of incidents in Q3 2025?",
        "description": "Question likely NOT answerable from context (tests honesty)",
    },
]


# =============================================================================
# FUNCTION: create_groundedness_judge
# =============================================================================
def create_groundedness_judge():
    """
    Create the LLM chain for judging groundedness.
    
    Returns a chain that evaluates if an answer is supported by context.
    """
    prompt = ChatPromptTemplate.from_template(GROUNDEDNESS_JUDGE_PROMPT)
    
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    
    parser = StrOutputParser()
    
    return prompt | llm | parser


# =============================================================================
# FUNCTION: parse_groundedness_verdict
# =============================================================================
def parse_groundedness_verdict(response: str) -> dict:
    """
    Parse the judge's response to extract verdict and score.
    
    Expected format:
        ANALYSIS: ...
        VERDICT: FULLY_GROUNDED
        SCORE: 1.0
    """
    result = {
        "analysis": "",
        "verdict": "UNKNOWN",
        "score": 0.0,
        "raw_response": response
    }
    
    lines = response.strip().split("\n")
    
    for line in lines:
        line_upper = line.upper().strip()
        
        if line_upper.startswith("ANALYSIS:"):
            result["analysis"] = line.split(":", 1)[1].strip() if ":" in line else ""
        
        elif line_upper.startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().upper() if ":" in line else ""
            if "FULLY" in verdict:
                result["verdict"] = "FULLY_GROUNDED"
                result["score"] = 1.0
            elif "PARTIAL" in verdict:
                result["verdict"] = "PARTIALLY_GROUNDED"
                result["score"] = 0.5
            elif "NOT" in verdict:
                result["verdict"] = "NOT_GROUNDED"
                result["score"] = 0.0
        
        elif line_upper.startswith("SCORE:"):
            try:
                score_str = line.split(":", 1)[1].strip()
                result["score"] = float(score_str)
            except (ValueError, IndexError):
                pass
    
    return result


# =============================================================================
# FUNCTION: evaluate_groundedness
# =============================================================================
def evaluate_groundedness(
    question: str,
    k: int = DEFAULT_K,
    filters: dict = None,
    verbose: bool = False
) -> dict:
    """
    Evaluate groundedness for a single question.
    
    STEPS:
    1. Retrieve relevant documents
    2. Generate an answer using RAG
    3. Ask judge: "Is this answer grounded in the context?"
    4. Return groundedness score
    
    Args:
        question: The test question
        k: Number of documents to retrieve
        filters: Optional metadata filters
        verbose: Print detailed output
    
    Returns:
        Dictionary with groundedness score and details
    """
    filter_kwargs = filters or {}
    
    if verbose:
        print(f"\n    📝 Question: {question[:50]}...")
    
    # --- Step 1: Retrieve documents ---
    documents = retrieve(
        query=question,
        top_k=k,
        **filter_kwargs
    )
    
    if not documents:
        return {
            "question": question,
            "error": "No documents retrieved",
            "groundedness_score": 0.0,
            "verdict": "NO_CONTEXT"
        }
    
    # Format context
    context = format_context(documents)
    
    if verbose:
        print(f"    📄 Retrieved {len(documents)} documents")
    
    # --- Step 2: Generate answer ---
    result = generate_answer(
        question=question,
        top_k=k,
        **filter_kwargs,
        verbose=False
    )
    
    answer = result["answer"]
    
    if verbose:
        print(f"    🤖 Answer: {answer[:100]}...")
    
    # --- Step 3: Judge groundedness ---
    judge = create_groundedness_judge()
    
    judge_response = judge.invoke({
        "context": context,
        "question": question,
        "answer": answer
    })
    
    verdict = parse_groundedness_verdict(judge_response)
    
    if verbose:
        score_pct = verdict["score"] * 100
        if verdict["score"] >= 0.8:
            status = "✅"
        elif verdict["score"] >= 0.5:
            status = "⚠️"
        else:
            status = "❌"
        print(f"    {status} Groundedness: {score_pct:.0f}% ({verdict['verdict']})")
    
    return {
        "question": question,
        "answer": answer,
        "context_docs": len(documents),
        "groundedness_score": verdict["score"],
        "verdict": verdict["verdict"],
        "analysis": verdict["analysis"],
        "sources": [doc.metadata.get("incident_id", "Unknown") for doc in documents]
    }


# =============================================================================
# FUNCTION: run_groundedness_evaluation
# =============================================================================
def run_groundedness_evaluation(
    test_cases: list = None,
    k: int = DEFAULT_K,
    verbose: bool = True
) -> dict:
    """
    Run groundedness evaluation on all test cases.
    
    Args:
        test_cases: List of test case dicts
        k: Number of documents to retrieve
        verbose: Print detailed progress
    
    Returns:
        Summary dict with aggregate metrics
    """
    if test_cases is None:
        test_cases = TEST_CASES
    
    print("=" * 70)
    print("📊 GROUNDEDNESS EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  • K (documents per query): {k}")
    print(f"  • Judge model: {JUDGE_MODEL}")
    print(f"  • Test cases: {len(test_cases)}")
    print(f"  • Timestamp: {datetime.now().isoformat()}")
    print("\n" + "-" * 70)
    
    results = []
    total_score = 0.0
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        filters = test_case.get("filters", {})
        
        print(f"\n[{i}/{len(test_cases)}] {test_case['id']}")
        
        result = evaluate_groundedness(
            question=question,
            k=k,
            filters=filters,
            verbose=verbose
        )
        
        result["test_id"] = test_case["id"]
        result["description"] = test_case.get("description", "")
        
        results.append(result)
        total_score += result["groundedness_score"]
    
    # --- Aggregate Metrics ---
    avg_groundedness = total_score / len(results) if results else 0
    
    fully_grounded = sum(1 for r in results if r["verdict"] == "FULLY_GROUNDED")
    partially_grounded = sum(1 for r in results if r["verdict"] == "PARTIALLY_GROUNDED")
    not_grounded = sum(1 for r in results if r["verdict"] == "NOT_GROUNDED")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "k": k,
            "judge_model": JUDGE_MODEL,
            "num_test_cases": len(test_cases)
        },
        "aggregate_metrics": {
            "average_groundedness": avg_groundedness,
            "fully_grounded_count": fully_grounded,
            "partially_grounded_count": partially_grounded,
            "not_grounded_count": not_grounded,
        },
        "per_query_results": results
    }
    
    # --- Print Summary ---
    print("\n" + "=" * 70)
    print("📈 GROUNDEDNESS SUMMARY")
    print("=" * 70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  AGGREGATE METRICS                                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Average Groundedness:     {avg_groundedness:.1%}                            │
│                                                                     │
│  Fully Grounded:           {fully_grounded}/{len(results)} answers                       │
│  Partially Grounded:       {partially_grounded}/{len(results)} answers                       │
│  Not Grounded:             {not_grounded}/{len(results)} answers                       │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print("Per-Query Breakdown:")
    for r in results:
        score_pct = r["groundedness_score"] * 100
        if score_pct >= 80:
            status = "✅"
        elif score_pct >= 50:
            status = "⚠️"
        else:
            status = "❌"
        print(f"  {status} {r['test_id']:<25} {score_pct:>5.0f}% ({r['verdict']})")
    
    # Interpretation
    print("\n" + "-" * 70)
    print("📋 INTERPRETATION:")
    if avg_groundedness >= 0.8:
        print("  ✅ Excellent groundedness - LLM answers are well-supported by context")
        print("  ✅ RAG is effectively reducing hallucination")
    elif avg_groundedness >= 0.5:
        print("  ⚠️ Moderate groundedness - some answers contain unsupported claims")
        print("  ⚠️ Consider improving retrieval or adjusting the prompt")
    else:
        print("  ❌ Poor groundedness - LLM is not properly using the context")
        print("  ❌ Check if retrieval is returning relevant documents")
    
    return summary


# =============================================================================
# MAIN
# =============================================================================
def main():
    """
    Run the groundedness evaluation.
    
    Usage:
        python evals/groundedness.py              # Full evaluation
        python evals/groundedness.py -k 3         # With K=3
        python evals/groundedness.py --output results.json
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate groundedness for Postmortem RAG"
    )
    parser.add_argument(
        "-k", type=int, default=DEFAULT_K,
        help=f"Number of documents to retrieve (default: {DEFAULT_K})"
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
    results = run_groundedness_evaluation(k=args.k, verbose=not args.quiet)
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    print("\n✅ Groundedness evaluation complete!")


if __name__ == "__main__":
    main()
