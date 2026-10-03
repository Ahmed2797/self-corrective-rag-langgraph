"""
Self-Corrective RAG Node Pipeline
==================================
This module contains the nodes and conditional router logic used within 
the LangGraph RAG application state workflow.
"""
import sys
import json
import uuid
import asyncio
from typing import List, Literal

from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from src.chat_model import get_llm
from src.constants import (
    CACHE_THRESHOLD,
    CACHE_VERSION,
    MAX_REWRITE_TRIES,
    INDEX_NAME_CACHE_MEMORY,
    SIMILARITY_THRESHOLD_FIASS,
    SIMILARITY_THRESHOLD_PINECONE,
)
from src.evalution.evaluation_engine import evaluate_answer
from src.exception import CustomException
from src.logger import logging
import src.mcp as mcp
from src.state import *
from src.vector import create_embedding, create_pincone_database

# Initialize primary language model
llm = get_llm()


# =====================================================================
# SECTION 1: QUERY OPTIMIZATION & INITIAL ROUTING NODES
# =====================================================================

# --- Node: Query Optimization ---

query_optimizer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query optimization module for a RAG retrieval system.\n"
            "Your task is to optimize the user's question for document retrieval.\n\n"
            "Goals:\n"
            "1. Remove unnecessary conversational words.\n"
            "2. Preserve the original meaning exactly.\n"
            "3. Extract important keywords and named entities.\n"
            "4. Create a concise retrieval-friendly query.\n\n"
            "Rules:\n"
            "- Do NOT answer the question.\n"
            "- Do NOT add information that is not present in the question.",
        ),
        ("human", "Question:\n{question}"),
    ]
)
optimizer_llm = llm.with_structured_output(OptimizedQueryOutput)


def optimizer_retrieval_node(state: State) -> State:
    """Optimize the user question for document retrieval."""
    try:
        question = state.get("question", "").strip()
        if not question:
            return {"optimized_query": "", "keywords": [], "entities": []}

        result: OptimizedQueryOutput = optimizer_llm.invoke(
            query_optimizer_prompt.format_messages(question=question)
        )
        optimized_query = result.optimized_query.strip() or question

        print("========== QUERY OPTIMIZER NODE ==========")
        logging.info("========== QUERY OPTIMIZER ==========")
        logging.info(f"Original: {question}")
        logging.info(f"Optimized: {optimized_query}")
        logging.info(f"Keywords: {result.keywords}")
        logging.info(f"Entities: {result.entities}")

        return {
            "optimized_query": optimized_query,
            "keywords": result.keywords,
            "entities": result.entities,
        }
    except Exception as e:
        logging.error(f"Query optimization failed: {e}")
        return {
            "optimized_query": state.get("question", ""),
            "keywords": [],
            "entities": [],
        }


# --- Node: Routing & Retrieval Decision ---

decide_route_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query classification module.\n"
            "Classify the user question into one of three routes:\n\n"
            "- 'rag': Questions requiring uploaded documents, files, or internal private knowledge bases.\n"
            "- 'tool': Questions requiring real-time, current, web-based, or time-sensitive data (e.g., today, weather, news, current price).\n"
            "- 'direct': General conceptual or foundational questions answerable via broad parametric knowledge.",
        ),
        ("human", "Question: {question}"),
    ]
)
router_llm = llm.with_structured_output(QueryRoute)

decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether retrieval is needed.\n"
            "Return JSON with key: should_retrieve (boolean).\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts from company documents.\n"
            "- should_retrieve=False for general explanations/definitions.\n"
            "- If unsure, choose True.",
        ),
        ("human", "Question: {question}"),
    ]
)
should_retrieve_llm = llm.with_structured_output(RetrieveDecision)


def decide_route(state: State) -> dict:
    """Classify input query into rag, tool, or direct routes."""
    try:
        logging.info("Starting query routing.")
        print("========== Starting query routing NODE ==========")
        question = state["question"]

        decision: QueryRoute = router_llm.invoke(
            decide_route_prompt.format_messages(question=question)
        )
        logging.info(f"Question: {question} | Selected route: {decision.route}")

        return {"route": decision.route}
    except Exception as e:
        logging.error(f"Error in query routing: {str(e)}")
        raise CustomException(e, sys)


def decide_retrieval(state: State) -> dict:
    """Decide whether the user question requires internal document retrieval."""
    try:
        logging.info("Starting retrieval decision.")
        question = state["question"]

        decision: RetrieveDecision = should_retrieve_llm.invoke(
            decide_retrieval_prompt.format_messages(question=question)
        )
        logging.info(f"Retrieval decision: {decision.should_retrieve}")

        return {"need_retrieval": decision.should_retrieve}
    except Exception as e:
        logging.error(f"Error in decide_retrieval: {str(e)}")
        raise CustomException(e, sys)


# def route_after_decide(state: State) -> Literal["web_search_tools", "retrieve"]:
#     """Conditional Edge: Route based on retrieval decision."""
    # return "retrieve" if state["need_retrieval"] else "web_search_tools"

def route_after_decide(state: State):
    """
    Route the question based on the query classification.

    Returns:
        str:
            - "retrieve" for internal RAG retrieval.
            - "web_search_tools" for external tools/web.
            - "generate_direct" for direct LLM generation.
    """
    route = state.get("route")

    logging.info(f"Routing decision: {route}")

    if route == "rag":
        return "retrieve"

    if route == "tool":
        return "web_search_tools"

    if route == "direct":
        return "generate_direct"

    raise ValueError(f"Unknown route: {route}")


# =====================================================================
# SECTION 2: DIRECT GENERATION & WEB SEARCH (TOOLS)
# =====================================================================

direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant.\n\n"
            "Answer the user's question directly.\n"
            "You have access to a web search tool called Tavily.\n\n"
            "Use Tavily when:\n"
            "- The question requires current or up-to-date information.\n"
            "- The user asks about recent events, prices, news, weather, or changing regulations.\n"
            "- You are not confident in internal static knowledge.\n\n"
            "Do NOT use Tavily for simple general knowledge, basic math, or basic programming concepts.",
        ),
        ("human", "Question: {question}"),
    ]
)


direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant.\n\n"
            "Answer the user's question accurately and directly.\n\n"

            "You have access to a web search tool called Tavily.\n\n"

            "Use Tavily when:\n"
            "- The question requires current or up-to-date information.\n"
            "- The user asks about the latest or recent news, events or developments,.\n"
            "- The information may have changed since your knowledge was learned.\n"
            "- You are not confident that your internal knowledge is sufficient.\n\n"

            "Do NOT use Tavily for:\n"
            "- Simple general knowledge questions.\n"
            "- Basic definitions or explanations.\n"
            "- Basic mathematics.\n"
            "- Basic programming concepts.\n"
            "- Questions that can be answered reliably without external information.\n\n"

            "IMPORTANT:\n"
            "- If Tavily is needed, call the Tavily search tool first.\n"
            "- After receiving the Tavily results, use those results to construct the final answer.\n"
            "- Do not invent current information.\n"
            "- If web search results are insufficient, clearly state that the available information is insufficient.\n"
        ),
        (
            "human",
            "Question: {question}"
        ),
    ]
)


@traceable(name="Get MCP Tool LLM")
def get_tool_llm():
    """Initialize MCP web search tools and bind them to the LLM."""
    logging.info("========== Initializing MCP Web Search Tool ==========")
    asyncio.run(mcp.initialize_mcp())
    return llm.bind_tools([mcp.search_tool])


def generate_with_web_tools(state: State) -> dict:
    """Generate an answer using external search tools when required."""
    try:
        print("========== GENERATE WITH TOOLS ==========")
        logging.info("========== GENERATE WITH TOOLS ==========")

        tool_llm = get_tool_llm()

        messages = direct_generation_prompt.format_messages(
            question=state["question"]
        )

        logging.info("Calling tool-enabled LLM...")
        response = tool_llm.invoke(messages)

        logging.info(f"Initial LLM response: {response}")
        logging.info(f"Tool calls: {response.tool_calls}")

        if not response.tool_calls:
            logging.info("No tool call detected.")
            return {
                "answer": response.content
            }

        messages.append(response)

        for tool_call in response.tool_calls:
            logging.info(
                f"Executing tool: {tool_call['name']} "
                f"with args: {tool_call['args']}"
            )

            if tool_call["name"] == "tavily_search":
                result = asyncio.run(mcp.search_tool.ainvoke(tool_call["args"]))
                logging.info(f"Tavily result: {result}")
                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                    )
                )

        logging.info("Calling LLM for final answer...")
        final_response = tool_llm.invoke(messages)

        logging.info(
            f"Final answer generated: {final_response.content}"
        )

        return {
            "answer": final_response.content
        }

    except Exception as e:
        logging.error(
            f"web_search_tools failed: {str(e)}"
        )
        raise CustomException(e, sys)


def generate_direct(state: State) -> dict:
    """Generate direct LLM response without tools or retrieval."""
    try:
        print("Generating direct answer.")
        logging.info("Generating direct answer.")
        question = state["question"]
        out = llm.invoke(direct_generation_prompt.format_messages(question=question))
        logging.info("Direct answer generated successfully.")

        return {"answer": out.content}
    except Exception as e:
        logging.error(f"Error in generate_direct: {str(e)}")
        raise CustomException(e, sys)


# =====================================================================
# SECTION 3: RAG RETRIEVAL & CONTEXT GENERATION
# =====================================================================

is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance at a TOPIC level.\n"
            "Return JSON matching the schema.\n\n"
            "A document is relevant if it discusses the same entity or topic area as the question.\n"
            "It does NOT need to contain the exact answer.\n"
            "When unsure, return is_relevant=true.",
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ]
)
relevance_llm = llm.with_structured_output(RelevanceDecision)

rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a business RAG chatbot.\n\n"
            "You will receive a CONTEXT block from internal company documents.\n"
            "Task: Answer the question based on the context.\n"
            "Do NOT mention that you received context in your output answer.\n"
            "If the question is not supported by the provided context, clearly state that the answer is based only on the uploaded project documents.",
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ]
)


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").lower().replace("\n", " ").split())


def _extract_keywords(question: str) -> List[str]:
    text = _normalize_text(question)
    tokens = [token for token in text.replace("?", " ").replace(".", " ").split() if len(token) > 2]
    seen = set()
    keyword_list = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            keyword_list.append(token)
    return keyword_list


def _compute_keyword_overlap(doc_text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    normalized = _normalize_text(doc_text)
    score = 0.0
    for keyword in keywords:
        if keyword in normalized:
            score += 1.0
    return min(1.0, score / max(len(keywords), 1))


def _extract_best_snippet(doc_text: str, keywords: List[str], max_chars: int = 220) -> str:
    normalized_text = _normalize_text(doc_text)
    if not keywords:
        return normalized_text[:max_chars]
    for keyword in keywords:
        idx = normalized_text.find(keyword)
        if idx != -1:
            start = max(0, idx - 80)
            end = min(len(normalized_text), idx + 180)
            snippet = normalized_text[start:end].strip()
            if snippet:
                return snippet[:max_chars]
    return normalized_text[:max_chars]


def rerank_documents(docs_with_scores, query: str):
    """Blend semantic vector similarity and lexical coverage for stronger retrieval ranking."""
    keywords = _extract_keywords(query)
    ranked = []

    for doc, score in docs_with_scores:
        semantic_part = 1.0 if score is None else max(0.0, 1.0 - min(abs(float(score)), 1.0))
        lexical_part = _compute_keyword_overlap(doc.page_content, keywords)
        hybrid_score = (0.7 * semantic_part) + (0.3 * lexical_part)
        ranked.append({
            "doc": doc,
            "score": float(score) if score is not None else 0.0,
            "hybrid_score": hybrid_score,
            "lexical_part": lexical_part,
        })

    ranked.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return ranked


def check_retrieval_score(state: State) -> State:
    """Keep close FAISS matches, with a keyword fallback for relevant chunks."""
    try:
        docs = state.get("docs", [])
        scores = state.get("retrieval_scores", [])
        relevant_docs = []

        for doc, score in zip(docs, scores):
            if score <= SIMILARITY_THRESHOLD_FIASS:
                relevant_docs.append(doc)

        # A strict vector cutoff can discard useful chunks for short or
        # document-specific questions. Keep the best lexical match among the
        # already retrieved top-k candidates when that happens.
        if not relevant_docs and docs:
            keywords = _extract_keywords(state.get("question", ""))
            lexical_matches = [
                (doc, _compute_keyword_overlap(doc.page_content, keywords))
                for doc in docs
            ]
            best_doc, best_overlap = max(lexical_matches, key=lambda item: item[1])
            if best_overlap > 0:
                relevant_docs = [best_doc]
                logging.info(
                    "Vector threshold rejected all results; retaining the best "
                    "lexical match (overlap=%.3f).",
                    best_overlap,
                )

        logging.info(f"Relevant documents passing score threshold: {len(relevant_docs)}/{len(docs)}")
        return {"relevant_docs": relevant_docs}
    except Exception as e:
        logging.error(f"Retrieval score check failed: {str(e)}")
        raise CustomException(e, sys)


def is_relevant(state: State) -> dict:
    """Filter retrieved documents based on topic-level LLM grading."""
    try:
        logging.info("Starting document relevance evaluation.")
        docs = state.get("docs", [])
        relevant_docs = []

        for doc in docs:
            decision: RelevanceDecision = relevance_llm.invoke(
                is_relevant_prompt.format_messages(
                    question=state["question"], document=doc.page_content
                )
            )
            if decision.is_relevant:
                relevant_docs.append(doc)

        logging.info(f"Relevant documents topic-checked: {len(relevant_docs)}/{len(docs)}")
        return {"relevant_docs": relevant_docs}
    except Exception as e:
        logging.error(f"Error in is_relevant: {str(e)}")
        raise CustomException(e, sys)


def route_after_relevance(state: State) -> Literal["generate_from_context", "no_answer_found"]:
    """Conditional Edge: Route to context generation or failure node."""
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_answer_found"


def generate_from_context(state: State) -> dict:
    """Generate answer using retrieved relevant document chunks."""
    try:
        logging.info("Generating answer from retrieved context.")
        relevant_docs = state.get("relevant_docs", []) or []
        context = "\n\n---\n\n".join(
            [d.page_content for d in relevant_docs]
        ).strip()

        sources = []
        citations = []
        keywords = _extract_keywords(state.get("question", ""))

        for doc in relevant_docs:
            source_name = doc.metadata.get("source") or doc.metadata.get("file_name") or doc.metadata.get("name")
            if source_name:
                sources.append(str(source_name))

            snippet = _extract_best_snippet(doc.page_content, keywords)
            citations.append({
                "source": str(source_name or "Document"),
                "snippet": snippet,
            })

        if not context:
            logging.warning("No relevant context available.")
            return {
                "answer": "I can only answer based on the uploaded project documents. No relevant information was found for this question.",
                "context": "",
                "sources": [],
                "citations": [],
                "confidence_score": 0.0,
            }

        out = llm.invoke(
            rag_generation_prompt.format_messages(
                question=state["question"], context=context
            )
        )
        logging.info("RAG answer generated successfully.")

        lexical_scores = [_compute_keyword_overlap(doc.page_content, keywords) for doc in relevant_docs]
        avg_overlap = sum(lexical_scores) / max(len(lexical_scores), 1)
        confidence_score = round(max(0.4, min(0.99, 0.45 + (avg_overlap * 0.55))), 3)

        return {
            "answer": out.content,
            "context": context,
            "sources": sources,
            "citations": citations,
            "confidence_score": confidence_score,
        }
    except Exception as e:
        logging.error(f"Error in generate_from_context: {str(e)}")
        raise CustomException(e, sys)


def no_answer_found(state: State) -> dict:
    """Fallback node when context is missing or irrelevant."""
    return {
        "answer": "I can only answer based on the uploaded project documents. No relevant information was found for this question.",
        "context": "",
        "sources": [],
        "citations": [],
        "confidence_score": 0.0,
    }


# =====================================================================
# SECTION 4: GROUNDEDNESS, REVISION & REWRITE CRITICS
# =====================================================================

issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict RAG evidence verifier.
Your ONLY task is to determine whether the ANSWER is supported by the provided CONTEXT.

Return JSON matching schema:
- issup: "fully_supported" | "partially_supported" | "no_support"
- evidence: List of short quote strings directly from context.

Strict Rules:
- Use ONLY the provided CONTEXT.
- Do NOT use outside knowledge or infer missing details.
- Evidence should contain up to 3 short direct quotes from CONTEXT.

Question:
{question}

Answer:
{answer}

Context:
{context}""",
        )
    ]
)
issup_llm = llm.with_structured_output(IsSUPDecision)

revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT reviser.\n\n"
            "Format (quote-only answer):\n"
            "- <direct quote from the CONTEXT>\n"
            "- <direct quote from the CONTEXT>\n\n"
            "Rules:\n"
            "- Use ONLY the CONTEXT.\n"
            "- Do NOT add any new words besides bullet dashes and direct quotes.\n"
            "- Do NOT explain anything or say 'context not provided'.",
        ),
        (
            "human",
            "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}",
        ),
    ]
)

isuse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an evaluator that judges the USEFULNESS of an AI answer for the user's question.

Your ONLY task is to determine whether the answer actually addresses what was asked.

Return:
- isuse: "useful" or "not_useful"
- use_reason: one short sentence explaining the rating.

Question:
{question}

Answer:
{answer}""",
        )
    ]
)
isuse_llm = llm.with_structured_output(IsUSEDecision)

rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs.\n\n"
            "Rules:\n"
            "- Keep it short (6–16 words).\n"
            "- Preserve key entities.\n"
            "- Add high-signal domain keywords.\n"
            "- Do NOT answer the question.",
        ),
        (
            "human",
            "QUESTION:\n{question}\n\n"
            "Previous retrieval query:\n{retrieval_query}\n\n"
            "Answer (if any):\n{answer}",
        ),
    ]
)
rewrite_llm = llm.with_structured_output(RewriteDecision)


def is_sup(state: State) -> State:
    """Verify if the generated answer is grounded in retrieved context."""
    logging.info("========== IsSUP Node ==========")
    relevant_docs = state.get("relevant_docs", [])

    context_chunks = [
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in relevant_docs
    ]
    context = "\n\n".join(context_chunks)

    decision: IsSUPDecision = issup_llm.invoke(
        issup_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=context,
        )
    )

    logging.info(f"IsSUP Decision: {decision.issup}")
    for evidence in decision.evidence:
        logging.info(f" - Evidence: {evidence}")

    return {
        "issup": decision.issup,
        "evidence": decision.evidence,
    }


def revise_answer(state: State) -> dict:
    """Revise generated answer using strict context quotes."""
    out = llm.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {
        "answer": out.content,
        "retries": state.get("retries", 0) + 1,
    }


def accept_answer(state: State) -> dict:
    """Accept the generated answer without edits."""
    return {}



def route_after_issup(
    state: State,
) -> Literal["accept_answer", "revise_answer", "no_answer_found"]:
    """Conditional Edge: Route based on groundedness check (IsSUP)."""
    logging.info("========== ROUTE AFTER ISSUP ==========")
    issup = state.get("issup")
    retries = state.get("retries", 0)

    logging.info(f"IsSUP: {issup} | Retries: {retries}")

    if issup == "fully_supported":
        logging.info("➡️ ACCEPT ANSWER")
        return "accept_answer"

    if retries >= MAX_REWRITE_TRIES:
        logging.info("➡️ MAX RETRIES REACHED → NO ANSWER")
        return "no_answer_found"

    logging.info("➡️ REVISE ANSWER")
    return "revise_answer"


def is_use(state: State) -> State:
    """Judge answer usefulness against the original user query."""
    logging.info("========== IsUSE Node ==========")
    question = state.get("question", "")
    answer = state.get("answer", "")

    decision: IsUSEDecision = isuse_llm.invoke(
        isuse_prompt.format_messages(question=question, answer=answer)
    )

    logging.info(f"IsUSE Rating: {decision.isuse}")
    logging.info(f"Reason: {decision.use_reason}")

    return {
        "isuse": decision.isuse,
        "use_reason": decision.use_reason,
    }


def rewrite_question(state: State) -> dict:
    """Rewrite retrieval query to improve vector search precision."""
    decision: RewriteDecision = rewrite_llm.invoke(
        rewrite_for_retrieval_prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        )
    )

    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "context": "",
    }


def route_after_isuse(
    state: State,
) -> Literal["evaluate_answer", "rewrite_question", "no_answer_found"]:
    """Conditional Edge: Route based on usefulness evaluation (IsUSE)."""
    isuse = state.get("isuse")
    rewrite_tries = state.get("rewrite_tries", MAX_REWRITE_TRIES)

    logging.info("========== ROUTE AFTER ISUSE ==========")
    logging.info(f"IsUSE: {isuse} | Rewrite tries: {rewrite_tries}")

    if isuse == "useful":
        logging.info("➡️ EVALUATE ANSWER")
        return "evaluate_answer"

    if rewrite_tries >= MAX_REWRITE_TRIES:
        logging.info("➡️ MAX REWRITES REACHED → NO ANSWER FOUND")
        return "no_answer_found"

    logging.info("➡️ REWRITE QUESTION")
    return "rewrite_question"


# =====================================================================
# SECTION 5: FINAL EVALUATION & SEMANTIC CACHING
# =====================================================================
@traceable(name="Evaluate Answer Engine")
def evaluate_answer_node(state: State) -> State:
    """Evaluate final answer quality using automated evaluation engine."""
    try:
        logging.info("========== EVALUATE ANSWER ==========")
        relevant_docs = state.get("relevant_docs", [])

        context_chunks = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "content": doc.page_content,
            }
            for doc in relevant_docs
        ]

        evaluation = evaluate_answer(
            question=state["question"],
            answer=state.get("answer", ""),
            context_chunks=context_chunks,
        )

        logging.info("========== EVALUATION METRICS ==========")
        logging.info(f"Hallucination Score: {evaluation.get('hallucination', {}).get('score')}")
        logging.info(f"Groundedness Score: {evaluation.get('groundedness', {}).get('score')}")
        logging.info(f"Overall Quality: {evaluation.get('overall_quality')}")
        logging.info(f"Reliable: {evaluation.get('is_reliable')}")

        return {"evaluation": evaluation}

    except Exception as e:
        logging.error(f"Error during answer evaluation: {str(e)}")
        raise CustomException(e, sys)

@traceable(name="Search Semantic Cache Index")
def search_semantic_cache(question: str):
    """Search Pinecone cache index for semantically similar query answers."""
    try:
        logging.info("Searching semantic cache.")
        query_embedding = create_embedding(question)

        cache_index = create_pincone_database(index_name=INDEX_NAME_CACHE_MEMORY)
        result = cache_index.query(
            vector=query_embedding, top_k=1, include_metadata=True
        )

        logging.info(f"Semantic cache search result: {result}")

        if not result.matches:
            logging.info("Semantic cache miss: no matching records.")
            return None

        match = result.matches[0]
        similarity = match.score

        if similarity < CACHE_THRESHOLD:
            logging.info(f"Semantic cache miss: similarity {similarity:.4f} < {CACHE_THRESHOLD}.")
            return None

        metadata: CacheMetadata = match.metadata

        if metadata.get("cache_version") != CACHE_VERSION:
            logging.info("Semantic cache miss: cache version mismatch.")
            return None

        logging.info("Semantic cache hit.")
        return {
            "answer": metadata.get("answer", ""),
            "question": metadata.get("question", ""),
            "evaluation": json.loads(metadata.get("evaluation", "{}")),
            "similarity": similarity,
        }

    except Exception as e:
        logging.error(f"Error searching semantic cache: {str(e)}")
        raise CustomException(e, sys)

@traceable(name="Save Semantic Cache Index")
def save_semantic_cache(
    question: str,
    answer: str,
    evaluation: EvaluationResult | None = None,
):
    """Upsert new question-answer pair into semantic memory cache."""
    try:
        logging.info("Saving answer to semantic cache.")
        embedding = create_embedding(question)

        cache_index = create_pincone_database(index_name=INDEX_NAME_CACHE_MEMORY)
        metadata: CacheMetadata = {
            "question": question,
            "answer": answer,
            "evaluation": json.dumps(evaluation or {}),
            "cache_version": CACHE_VERSION,
        }

        cache_index.upsert(
            vectors=[
                {
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": metadata,
                }
            ]
        )
        logging.info("Answer saved to semantic cache successfully.")

    except Exception as e:
        logging.error(f"Error saving semantic cache: {str(e)}")
        raise CustomException(e, sys)


def check_semantic_cache_node(state: State) -> dict:
    """Node: Check if similar query answer exists in semantic cache."""
    try:
        question = state["question"]
        logging.info("=" * 50)
        logging.info(f"SEMANTIC CACHE CHECK: {question}")
        logging.info("=" * 50)

        cached = search_semantic_cache(question)

        if cached is None:
            return {"cache_hit": False, "cache_similarity": 0.0}

        logging.info(
            f"Semantic Cache HIT | Similarity={cached['similarity']:.3f}"
        )

        return {
            "cache_hit": True,
            "cache_similarity": cached["similarity"],
            "answer": cached["answer"],
            "evaluation": cached["evaluation"],
        }

    except Exception as e:
        logging.error(f"Error checking semantic cache: {str(e)}")
        raise CustomException(e, sys)

# -----------------------------
# 11) Memory Cache Decision Node
# -----------------------------

def save_semantic_cache_node(state: State) -> State:
    """
    Save the final answer and evaluation to semantic cache.
    """
    logging.info("\n========== Save Cache Node ==========")
    save_semantic_cache(
        question=state["question"],
        answer=state.get("answer", ""),
        evaluation=state.get("evaluation"),
    )

    return {"cache_saved": True}


def route_after_cache(state: State):

    if state.get("cache_hit"):

        return "cached_answer"

    return "continue_rag"


def web_search_node(state: State):

    query = (
        state.get("retrieval_query")
        or state["question"]
    )

    result = asyncio.run(mcp.tavily_mcp_search(query))

    return {
        "web_results": result
    }
