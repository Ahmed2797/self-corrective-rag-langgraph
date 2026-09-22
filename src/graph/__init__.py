from typing import Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from src.state import State
from src.logger import logging
from src.exception import CustomException
from src.retrive.retrieval_pipeline import (
    optimizer_retrieval_node,
    check_semantic_cache_node,
    check_retrieval_score,
    decide_retrieval,decide_route,
    generate_direct,
    generate_with_tools,
    # is_relevant,
    generate_from_context,
    no_answer_found,
    is_sup,
    revise_answer,
    is_use,
    rewrite_question,
    evaluate_answer_node,
    save_semantic_cache_node,
    route_after_cache,
    route_after_decide,
    route_after_relevance,
    route_after_issup,
    route_after_isuse,
)


def retrieve(state: State, retriever) -> State:
    """
    Retrieve documents using similarity search and store their scores.

    Args:
        state (State): Current LangGraph state.
        retriever: Configured FAISS or Pinecone vector store.

    Returns:
        State: Updated state containing retrieved documents and similarity scores.

    Raises:
        CustomException: If document retrieval fails.
    """
    try:
        logging.info("Starting document retrieval.")

        query = state.get("retrieval_query") or state["question"]

        logging.info(f"Retrieval query: {query}")

        docs_with_scores = retriever.similarity_search_with_score(
            query,
            k=5
        )

        docs = []
        retrieval_scores = []

        for doc, score in docs_with_scores:
            # if score <= SIMILARITY_THRESHOLD:
            docs.append(doc)
            retrieval_scores.append(float(score))

        logging.info(
            f"Retrieved {len(docs)} documents, "
            f"{len(retrieval_scores)} passed similarity threshold."
        )

        return {
            "docs": docs,
            "retrieval_scores": retrieval_scores,
        }

    except Exception as e:
        logging.error(
            f"Error during document retrieval: {str(e)}"
        )
        raise CustomException(e)


def create_graph(retriever, checkpointer: Optional[BaseCheckpointSaver] = None):
    """
    Create and compile the LangGraph RAG workflow with multi-turn memory.

    Args:
        retriever: Configured vector store retriever (FAISS, Pinecone, etc.).
        checkpointer: LangGraph checkpointer for state persistence/memory. 
                      Defaults to MemorySaver if None is provided.

    Returns:
        CompiledStateGraph: Production-ready compiled LangGraph application.

    Raises:
        CustomException: If graph creation or compilation fails.
    """
    try:
        logging.info("Starting LangGraph creation.")

        g = StateGraph(State)

        # Default to in-memory checkpointer if none provided
        if checkpointer is None:
            checkpointer = MemorySaver()

        logging.info("Adding LangGraph nodes.")

        # ------------------------------------------
        # Cache Nodes
        # ------------------------------------------
        g.add_node("check_semantic_cache", check_semantic_cache_node)
        g.add_node("save_semantic_cache", save_semantic_cache_node)

        # ------------------------------------------
        # Retrieval & Routing Nodes
        # ------------------------------------------
        # g.add_node("decide_retrieval", decide_retrieval)
        g.add_node("decide_retrieval", decide_route)
        g.add_node("query_optimizer", optimizer_retrieval_node)
        g.add_node("retrieve", lambda state: retrieve(state, retriever))
        g.add_node("check_retrieval_score", check_retrieval_score)

        # ------------------------------------------
        # Generation Nodes
        # ------------------------------------------
        g.add_node("generate_direct", generate_direct)
        g.add_node("generate_with_tools", generate_with_tools)
        g.add_node("generate_from_context", generate_from_context)

        # ------------------------------------------
        # Answer Verification & Self-Correction
        # ------------------------------------------
        g.add_node("is_sup", is_sup)
        g.add_node("revise_answer", revise_answer)
        g.add_node("is_use", is_use)
        g.add_node("rewrite_question", rewrite_question)

        # ------------------------------------------
        # Evaluation & Fallback
        # ------------------------------------------
        g.add_node("evaluate_answer", evaluate_answer_node)
        g.add_node("no_answer_found", no_answer_found)

        logging.info("LangGraph nodes added successfully.")

        # ==================================================
        # EDGES & ROUTING
        # ==================================================
        logging.info("Adding LangGraph edges.")

        # 1. Entry Point -> Semantic Cache Check
        g.add_edge(START, "check_semantic_cache")

        # 2. Semantic Cache Routing
        g.add_conditional_edges(
            "check_semantic_cache",
            route_after_cache,
            {
                "cached_answer": END,
                "continue_rag": "decide_retrieval",
            },
        )

        # 3. Retrieval Decision Routing
        g.add_conditional_edges(
            "decide_retrieval",
            route_after_decide,
            {
                "generate_direct":"generate_direct",
                "generate_with_tools": "generate_with_tools",
                "retrieve": "retrieve",
            },
        )

        # 4. Tool Generation -> Evaluation
        g.add_edge("generate_direct", "save_semantic_cache")
        g.add_edge("generate_with_tools", "save_semantic_cache")

        # 5. Retrieval -> Score Check & Relevance Filtering
        g.add_edge("retrieve", "check_retrieval_score")

        g.add_conditional_edges(
            "check_retrieval_score",
            route_after_relevance,
            {
                "generate_from_context": "generate_from_context",
                "query_optimizer": "query_optimizer",
                "no_answer_found": "no_answer_found",
            },
        )

        # 6. Query Optimizer Loop back to Retrieval
        g.add_edge("query_optimizer", "retrieve")

        # 7. Context Generation -> Support Check (Hallucination Verification)
        g.add_edge("generate_from_context", "is_sup")

        # 8. Support Routing (IsSUP -> Usefulness / Revision / Fallback)
        g.add_conditional_edges(
            "is_sup",
            route_after_issup,
            {
                "accept_answer": "is_use",
                "revise_answer": "revise_answer",
                "no_answer_found": "no_answer_found",
            },
        )

        # 9. Revision Loop back to Support Check
        g.add_edge("revise_answer", "is_sup")

        # 10. Usefulness Routing (IsUSE -> Evaluation / Rewrite / Fallback)
        g.add_conditional_edges(
            "is_use",
            route_after_isuse,
            {
                "evaluate_answer": "evaluate_answer",
                "rewrite_question": "rewrite_question",
                "no_answer_found": "no_answer_found",
            },
        )

        # 11. Rewrite Question Loop back to Retrieval
        g.add_edge("rewrite_question", "retrieve")

        # 12. Save Cache -> END
        g.add_edge("evaluate_answer", "save_semantic_cache")
        g.add_edge("save_semantic_cache", END)

        # 13. Fallback -> END
        g.add_edge("no_answer_found", END)

        logging.info("LangGraph edges added successfully.")

        # ==================================================
        # COMPILE WITH MEMORY
        # ==================================================
        app = g.compile(checkpointer=checkpointer)

        logging.info("LangGraph compiled successfully with checkpointer persistence.")

        return app

    except Exception as e:
        logging.error(f"Error while creating LangGraph: {str(e)}")
        raise CustomException(e)

# app = create_graph(retrieve)
# app