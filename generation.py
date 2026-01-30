"""
Postmortem RAG - Generation Module
====================================

This module combines retrieval with LLM generation to answer questions.
This is the complete RAG pipeline: Retrieve → Augment → Generate.

ARCHITECTURE:
    Question → Retrieve Relevant Docs → Build Prompt → LLM → Answer

KEY CONCEPT - Retrieval-Augmented Generation:
    Instead of asking the LLM to answer from its training data (which may be
    outdated or hallucinated), we:
    1. RETRIEVE relevant documents from our knowledge base
    2. AUGMENT the prompt with this retrieved context
    3. GENERATE an answer based on the provided context
    
    This grounds the LLM's response in YOUR data.

Author: Jorge Tapicha
Date: 2026-01-28
"""

# =============================================================================
# IMPORTS - Detailed Explanations
# =============================================================================

# -----------------------------------------------------------------------------
# STANDARD LIBRARY
# -----------------------------------------------------------------------------
import os
from typing import Optional

# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
from dotenv import load_dotenv
# Purpose: Load .env file into environment
# Why: Keep API keys out of source code

# -----------------------------------------------------------------------------
# LANGCHAIN-OPENAI: LLM Integration
# -----------------------------------------------------------------------------
from langchain_openai import ChatOpenAI
# 
# PURPOSE: 
#   Wrapper around OpenAI's Chat API (GPT-4, GPT-4o-mini, etc.)
#   Provides a consistent interface for LangChain chains.
#
# HOW IT WORKS:
#   1. Takes your prompt
#   2. Sends it to OpenAI's API
#   3. Returns the response in LangChain format
#
# KEY PARAMETERS:
#   - model: Which model to use ("gpt-4o-mini", "gpt-4o", "gpt-4")
#   - temperature: Randomness (0=deterministic, 1=creative)
#   - max_tokens: Maximum response length
#   - openai_api_key: Your API key
#
# EXAMPLE:
#   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#   response = llm.invoke("What is RAG?")
#   # Returns: AIMessage(content="RAG stands for...")
#
# COST (as of 2024):
#   - gpt-4o-mini: ~$0.15 per 1M input tokens (cheapest)
#   - gpt-4o: ~$5 per 1M input tokens (best quality)
#   - gpt-4: ~$30 per 1M input tokens (legacy)
#
# Install: pip install langchain-openai

# -----------------------------------------------------------------------------
# LANGCHAIN-CORE: Prompts
# -----------------------------------------------------------------------------
from langchain_core.prompts import ChatPromptTemplate
#
# PURPOSE:
#   Create reusable prompt templates with placeholders.
#   Think of it like f-strings but for LLM prompts.
#
# WHY USE TEMPLATES:
#   - Consistent prompt structure
#   - Easy to swap variables (context, question)
#   - Supports system/human message roles
#   - Can be chained with other components
#
# HOW IT WORKS:
#   template = ChatPromptTemplate.from_messages([
#       ("system", "You are a helpful assistant."),
#       ("human", "Answer this: {question}")
#   ])
#   
#   # Later, fill in the variables:
#   prompt = template.invoke({"question": "What is RAG?"})
#   # Returns: [SystemMessage(...), HumanMessage(...)]
#
# MESSAGE TYPES:
#   - "system": Sets the AI's behavior/personality
#   - "human": The user's input
#   - "ai": Previous AI responses (for conversation history)
#
# TEMPLATE VARIABLES:
#   Use {variable_name} as placeholders:
#   "Based on {context}, answer {question}"
#   → invoke({"context": "...", "question": "..."})
#
# Install: pip install langchain-core (usually auto-installed)

# -----------------------------------------------------------------------------
# LANGCHAIN-CORE: Output Parsers
# -----------------------------------------------------------------------------
from langchain_core.output_parsers import StrOutputParser
#
# PURPOSE:
#   Extract and format the LLM's response.
#   Converts LangChain's AIMessage to simple Python types.
#
# WHY NEEDED:
#   - ChatOpenAI returns AIMessage objects, not strings
#   - We usually just want the text content
#   - Parsers handle this conversion cleanly
#
# HOW IT WORKS:
#   parser = StrOutputParser()
#   
#   # Without parser:
#   response = llm.invoke("Hello")
#   # Returns: AIMessage(content="Hello!", ...)
#   
#   # With parser:
#   chain = llm | parser
#   response = chain.invoke("Hello")
#   # Returns: "Hello!"
#
# OTHER PARSERS AVAILABLE:
#   - JsonOutputParser: Parse JSON responses
#   - PydanticOutputParser: Parse into Pydantic models
#   - CommaSeparatedListOutputParser: Parse lists
#
# Install: pip install langchain-core

# -----------------------------------------------------------------------------
# LANGCHAIN EXPRESSION LANGUAGE (LCEL) - The | Operator
# -----------------------------------------------------------------------------
#
# PURPOSE:
#   Chain components together using the pipe operator.
#   Makes building pipelines intuitive and readable.
#
# HOW IT WORKS:
#   chain = prompt | llm | parser
#   
#   This means:
#   1. prompt receives input dict, outputs formatted messages
#   2. llm receives messages, outputs AIMessage
#   3. parser receives AIMessage, outputs string
#
# EXAMPLE:
#   # Define chain
#   chain = (
#       ChatPromptTemplate.from_template("Translate to French: {text}")
#       | ChatOpenAI(model="gpt-4o-mini")
#       | StrOutputParser()
#   )
#   
#   # Use chain
#   result = chain.invoke({"text": "Hello world"})
#   # Returns: "Bonjour le monde"
#
# BENEFITS:
#   - Readable: Left-to-right data flow
#   - Composable: Easy to add/remove steps
#   - Streaming: Supports streaming responses
#   - Async: Supports async/await

# -----------------------------------------------------------------------------
# LOCAL IMPORTS
# -----------------------------------------------------------------------------
from retrieval import retrieve_unique, format_context, get_collection_stats
# Our retrieval module from Step 2
# - retrieve(): Raw vector search (may have duplicate incidents)
# - retrieve_unique(): Deduplicated retrieval (one doc per incident) - RECOMMENDED
# - format_context(): Format docs for LLM prompt
# - get_collection_stats(): Debug helper

# -----------------------------------------------------------------------------
# LOAD ENVIRONMENT
# -----------------------------------------------------------------------------
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM Configuration
MODEL_NAME = "gpt-4o-mini"
# Why gpt-4o-mini:
# - Fast and cheap (~$0.15 per 1M input tokens)
# - Good enough for most RAG tasks
# - For production, consider gpt-4o for complex reasoning

TEMPERATURE = 0.0
# Why 0.0 temperature:
# - Deterministic outputs (same input = same output)
# - For RAG, we want factual answers, not creative ones
# - Higher temperature = more randomness/creativity

TOP_K = 5
# Number of documents to retrieve for context
# Trade-off:
# - More docs = more context, but higher cost and may dilute relevance
# - Fewer docs = cheaper, but might miss relevant info


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# System prompt - Sets the behavior of the assistant
SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) assistant that helps engineers understand and learn from past incidents and postmortems.

Your role is to:
1. Answer questions based ONLY on the provided postmortem context
2. Be precise and technical in your responses
3. Cite specific incidents when referencing information
4. If the context doesn't contain enough information, say so clearly

Guidelines:
- Focus on root causes, fixes, and lessons learned
- Use SRE terminology (RCA, blast radius, MTTR, etc.) when appropriate
- Be concise but thorough
- Never make up information not in the context"""

# RAG prompt template
RAG_PROMPT_TEMPLATE = """Based on the following postmortem documentation, answer the question.

{filter_context}

=== POSTMORTEM CONTEXT ===
{context}
=== END CONTEXT ===

Question: {question}

Instructions:
- Answer based ONLY on the information in the context above
- If the context doesn't contain relevant information, say "I don't have enough information in the provided postmortems to answer this question."
- Cite incident IDs (e.g., PFU-123) when referencing specific incidents
- Be specific about root causes, fixes, and preventive measures

Answer:"""


# =============================================================================
# FUNCTION: create_rag_chain
# =============================================================================
def create_rag_chain():
    """
    Create the LangChain RAG chain.
    
    WHAT IS A CHAIN:
    ================
    A chain is a sequence of operations connected with the | operator.
    Data flows from left to right through each component.
    
    VISUAL REPRESENTATION:
    ======================
    
        Input Dict                  Messages              AIMessage           String
    {"context": "...",    →    [SystemMessage,    →    AIMessage      →    "The answer
     "question": "..."}         HumanMessage]          (content=...)        is..."
           │                         │                      │                  │
           ▼                         ▼                      ▼                  ▼
    ┌──────────────┐        ┌──────────────┐        ┌──────────┐       ┌────────────┐
    │   PROMPT     │   |    │     LLM      │   |    │  PARSER  │   =   │   CHAIN    │
    │  TEMPLATE    │───────▶│  (ChatOpenAI)│───────▶│(StrOutput│       │            │
    │              │        │              │        │  Parser) │       │            │
    └──────────────┘        └──────────────┘        └──────────┘       └────────────┘
    
    COMPONENTS:
    ===========
    1. ChatPromptTemplate: 
       - Takes: {"context": "...", "question": "...", ...}
       - Returns: [SystemMessage("You are..."), HumanMessage("Based on...")]
       
    2. ChatOpenAI:
       - Takes: List of messages
       - Sends to OpenAI API
       - Returns: AIMessage(content="The answer is...", ...)
       
    3. StrOutputParser:
       - Takes: AIMessage object
       - Returns: Just the string content ("The answer is...")
    
    WHY THIS PATTERN:
    =================
    - Separation of concerns: Each component does one thing
    - Reusable: Same chain for different inputs
    - Testable: Can test each component separately
    - Composable: Easy to add logging, caching, etc.
    """
    
    # -------------------------------------------------------------------------
    # Step 1: Create the Prompt Template
    # -------------------------------------------------------------------------
    # from_messages() creates a chat-style prompt with roles
    prompt = ChatPromptTemplate.from_messages([
        # System message: Sets the AI's personality and behavior
        # This is like giving the AI a job description
        ("system", SYSTEM_PROMPT),
        
        # Human message: The actual question with context
        # Variables in {curly_braces} will be filled in at runtime
        ("human", RAG_PROMPT_TEMPLATE)
    ])
    # At this point, prompt is a template waiting for variables
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize the LLM
    # -------------------------------------------------------------------------
    llm = ChatOpenAI(
        model=MODEL_NAME,         # "gpt-4o-mini" - fast and cheap
        temperature=TEMPERATURE,  # 0.0 - deterministic (same input = same output)
        openai_api_key=OPENAI_API_KEY
        # Other useful parameters:
        # max_tokens=1000,        # Limit response length
        # request_timeout=30,     # Timeout in seconds
        # max_retries=2,          # Retry on failure
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Create the Output Parser
    # -------------------------------------------------------------------------
    # StrOutputParser extracts just the text content from AIMessage
    output_parser = StrOutputParser()
    
    # -------------------------------------------------------------------------
    # Step 4: Build the Chain with LCEL (LangChain Expression Language)
    # -------------------------------------------------------------------------
    # The | operator connects components in sequence
    # Data flows: prompt → llm → parser
    chain = prompt | llm | output_parser
    
    # This is equivalent to:
    # def chain(inputs):
    #     messages = prompt.invoke(inputs)
    #     ai_message = llm.invoke(messages)
    #     text = output_parser.invoke(ai_message)
    #     return text
    
    return chain


# =============================================================================
# FUNCTION: build_filter_context
# =============================================================================
def build_filter_context(
    severity: Optional[str | list[str]] = None,
    root_cause_category: Optional[str | list[str]] = None,
    services_affected: Optional[str | list[str]] = None,
    owner_team: Optional[str] = None,
) -> str:
    """
    Build a human-readable description of applied filters.
    
    This is included in the prompt so the LLM knows the scope of the search.
    
    Example output:
    "You are answering based on a filtered subset: severity='S1', category='database'"
    """
    parts = []
    
    if severity:
        if isinstance(severity, list):
            parts.append(f"severity in [{', '.join(severity)}]")
        else:
            parts.append(f"severity='{severity}'")
    
    if root_cause_category:
        if isinstance(root_cause_category, list):
            parts.append(f"category in [{', '.join(root_cause_category)}]")
        else:
            parts.append(f"category='{root_cause_category}'")
    
    if services_affected:
        if isinstance(services_affected, list):
            parts.append(f"services in [{', '.join(services_affected)}]")
        else:
            parts.append(f"service='{services_affected}'")
    
    if owner_team:
        parts.append(f"team='{owner_team}'")
    
    if parts:
        return f"Note: Results are filtered by: {', '.join(parts)}"
    else:
        return "Note: Searching across all postmortems (no filters applied)."


# =============================================================================
# FUNCTION: generate_answer
# =============================================================================
def generate_answer(
    question: str,
    top_k: int = TOP_K,
    severity: Optional[str | list[str]] = None,
    root_cause_category: Optional[str | list[str]] = None,
    services_affected: Optional[str | list[str]] = None,
    owner_team: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Generate an answer using the full RAG pipeline.
    
    THIS IS THE MAIN FUNCTION!
    
    STEPS:
    1. Retrieve relevant documents (with optional filters)
    2. Format documents as context
    3. Build the prompt with question + context
    4. Send to LLM
    5. Return answer with sources
    
    Args:
        question: The user's question
        top_k: Number of documents to retrieve
        severity: Filter by severity level(s)
        root_cause_category: Filter by root cause category
        services_affected: Filter by affected service(s)
        owner_team: Filter by owning team
        verbose: Print debug information
    
    Returns:
        Dictionary with:
        - answer: The generated answer
        - sources: List of source documents used
        - filters_applied: Description of filters used
    
    Example:
        result = generate_answer(
            question="What caused the MongoDB latency incident?",
            root_cause_category="database"
        )
        print(result["answer"])
    """
    
    if verbose:
        print(f"🔍 Question: {question}")
    
    # --- Step 1: Retrieve relevant documents ---
    # Using retrieve_unique to get one document per incident (deduplicated)
    # merge=True combines all chunks from the same incident for complete context
    documents = retrieve_unique(
        query=question,
        top_k=top_k,
        merge=True,  # Merge chunks from same incident
        severity=severity,
        root_cause_category=root_cause_category,
        services_affected=services_affected,
        owner_team=owner_team,
    )
    
    if not documents:
        return {
            "answer": "I couldn't find any relevant postmortems matching your query and filters.",
            "sources": [],
            "filters_applied": build_filter_context(
                severity, root_cause_category, services_affected, owner_team
            )
        }
    
    if verbose:
        print(f"📄 Retrieved {len(documents)} unique incidents")
    
    # --- Step 2: Format context ---
    context = format_context(documents)
    filter_context = build_filter_context(
        severity, root_cause_category, services_affected, owner_team
    )
    
    # --- Step 3: Generate answer ---
    if verbose:
        print(f"🤖 Generating answer with {MODEL_NAME}...")
    
    chain = create_rag_chain()
    
    answer = chain.invoke({
        "context": context,
        "question": question,
        "filter_context": filter_context
    })
    
    # --- Step 4: Prepare response ---
    sources = [
        {
            "incident_id": doc.metadata.get("incident_id", "Unknown"),
            "severity": doc.metadata.get("severity", "Unknown"),
            "category": doc.metadata.get("root_cause_category", "Unknown"),
            "services": doc.metadata.get("services_affected", []),
            "source_file": doc.metadata.get("source_file", "Unknown"),
        }
        for doc in documents
    ]
    
    return {
        "answer": answer,
        "sources": sources,
        "filters_applied": filter_context
    }


# =============================================================================
# FUNCTION: interactive_mode
# =============================================================================
def interactive_mode():
    """
    Run an interactive Q&A session.
    
    Supports commands for setting filters:
    - severity:S1        → Filter by S1 severity
    - category:database  → Filter by database category
    - clear              → Clear all filters
    - quit               → Exit
    """
    print("\n" + "=" * 60)
    print("🤖 Postmortem RAG - Interactive Q&A")
    print("=" * 60)
    print("""
Commands:
  severity:<value>    Set severity filter (e.g., severity:S1 - Critical Severity)
  category:<value>    Set category filter (e.g., category:database)
  service:<value>     Set service filter (e.g., service:payment-rx-orc)
  team:<value>        Set team filter (e.g., team:core)
  clear               Clear all filters
  stats               Show collection statistics
  quit                Exit

Just type a question to search and get an answer.
""")
    print("-" * 60)
    
    # Active filters
    filters = {
        "severity": None,
        "root_cause_category": None,
        "services_affected": None,
        "owner_team": None
    }
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # --- Handle commands ---
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if user_input.lower() == "clear":
            filters = {k: None for k in filters}
            print("✅ Filters cleared")
            continue
        
        if user_input.lower() == "stats":
            stats = get_collection_stats()
            print(f"\n📊 Collection Statistics:")
            print(f"   Total chunks: {stats['total_documents']}")
            print(f"\n   Severities: {stats['severities']}")
            print(f"\n   Categories: {stats['categories']}")
            print(f"\n   Top Services: {dict(list(stats['services'].items())[:5])}")
            continue
        
        if user_input.startswith("severity:"):
            filters["severity"] = user_input.split(":", 1)[1].strip()
            print(f"✅ Severity filter set: {filters['severity']}")
            continue
        
        if user_input.startswith("category:"):
            filters["root_cause_category"] = user_input.split(":", 1)[1].strip()
            print(f"✅ Category filter set: {filters['root_cause_category']}")
            continue
        
        if user_input.startswith("service:"):
            filters["services_affected"] = user_input.split(":", 1)[1].strip()
            print(f"✅ Service filter set: {filters['services_affected']}")
            continue
        
        if user_input.startswith("team:"):
            filters["owner_team"] = user_input.split(":", 1)[1].strip()
            print(f"✅ Team filter set: {filters['owner_team']}")
            continue
        
        # --- It's a question ---
        active_filters = {k: v for k, v in filters.items() if v is not None}
        
        if active_filters:
            print(f"🔍 Searching with filters: {active_filters}")
        
        print("⏳ Thinking...\n")
        
        try:
            result = generate_answer(
                question=user_input,
                **filters,
                verbose=False
            )
            
            print("=" * 60)
            print("🤖 Assistant:")
            print("=" * 60)
            print(result["answer"])
            
            print(f"\n📚 Sources ({len(result['sources'])} documents):")
            for src in result["sources"]:
                print(f"   • {src['incident_id']} | {src['severity']} | {src['category']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """
    Demonstrate the generation pipeline with example queries.
    """
    print("=" * 60)
    print("🚀 Postmortem RAG - Generation Pipeline")
    print("=" * 60)
    
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    
    # Example queries
    example_queries = [
        {
            "question": "What are the most common root causes of database-related incidents?",
            "root_cause_category": "database",
            "description": "Database incidents analysis"
        },
        {
            "question": "What preventive measures were implemented after critical incidents?",
            "severity": "S1 - Critical Severity",
            "description": "Critical incident learnings"
        },
        {
            "question": "How were timeout issues typically resolved?",
            "description": "Timeout resolution patterns"
        },
    ]
    
    print("\n📋 Example Queries:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q['description']}")
    print(f"  {len(example_queries) + 1}. Interactive mode")
    
    print("\n" + "-" * 60)
    choice = input("Select (1-4) or type your own question: ").strip()
    
    if choice == str(len(example_queries) + 1) or choice.lower() == "i":
        interactive_mode()
        return
    
    # Handle selection or custom question
    if choice.isdigit() and 1 <= int(choice) <= len(example_queries):
        q = example_queries[int(choice) - 1]
        question = q["question"]
        filters = {k: v for k, v in q.items() if k not in ["question", "description"]}
    else:
        question = choice if choice else example_queries[0]["question"]
        filters = {}
    
    print(f"\n📝 Question: {question}")
    if filters:
        print(f"📁 Filters: {filters}")
    
    print("\n⏳ Retrieving and generating...\n")
    
    result = generate_answer(question, verbose=True, **filters)
    
    print("\n" + "=" * 60)
    print("🤖 Answer:")
    print("=" * 60)
    print(result["answer"])
    
    print(f"\n📚 Sources:")
    for src in result["sources"]:
        services_str = ", ".join(src["services"][:2]) if src["services"] else "N/A"
        print(f"   • {src['incident_id']}")
        print(f"     Severity: {src['severity']} | Category: {src['category']}")
        print(f"     Services: {services_str}")
    
    print(f"\n{result['filters_applied']}")


if __name__ == "__main__":
    main()
