from langgraph.graph import StateGraph, START, END
from src.state import State
from src.retrive.retrieval_pipeline import (
    check_semantic_cache_node,
    decide_retrieval,
    generate_direct,
    is_relevant,
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
from src.exception import CustomException
from src.logger import logging


def retrieve(state: State, retriever):
    """
    Retrieve documents using the configured retriever.

    Args:
        state (State): Current LangGraph state.
        retriever: Configured FAISS or Pinecone retriever.

    Returns:
        dict: Retrieved documents.

    Raises:
        CustomException: If document retrieval fails.
    """
    try:
        logging.info("Starting document retrieval.")

        q = state.get("retrieval_query") or state["question"]

        logging.info(f"Retrieval query: {q}")

        docs = retriever.invoke(q)

        logging.info(f"Retrieved {len(docs)} documents.")

        return {"docs": docs}

    except Exception as e:
        logging.error(f"Error during document retrieval: {str(e)}")
        raise CustomException(e)


def create_graph(retriever):
    """
    Create and compile the LangGraph RAG workflow.

    Args:
        retriever: Configured FAISS or Pinecone retriever.

    Returns:
        CompiledStateGraph: Compiled LangGraph application.

    Raises:
        CustomException: If graph creation or compilation fails.
    """
    try:
        logging.info("Starting LangGraph creation.")

        g = StateGraph(State)

        logging.info("Adding LangGraph nodes.")

        g.add_node("check_semantic_cache", check_semantic_cache_node)
        g.add_node("decide_retrieval", decide_retrieval)
        g.add_node("generate_direct", generate_direct)
        g.add_node("retrieve", lambda state: retrieve(state, retriever))
        g.add_node("is_relevant", is_relevant)
        g.add_node("generate_from_context", generate_from_context)
        g.add_node("no_answer_found", no_answer_found)
        g.add_node("is_sup", is_sup)
        g.add_node("revise_answer", revise_answer)
        g.add_node("is_use", is_use)
        g.add_node("rewrite_question", rewrite_question)
        g.add_node("evaluate_answer", evaluate_answer_node)
        g.add_node("save_semantic_cache", save_semantic_cache_node)

        logging.info("LangGraph nodes added successfully.")

        logging.info("Adding LangGraph edges.")

        g.add_edge(START, "check_semantic_cache")

        g.add_conditional_edges(
            "check_semantic_cache",
            route_after_cache,
            {"cached_answer": END, "continue_rag": "decide_retrieval"},
        )

        g.add_conditional_edges(
            "decide_retrieval",
            route_after_decide,
            {"generate_direct": "generate_direct", "retrieve": "retrieve"},
        )

        g.add_edge("generate_direct", END)

        g.add_edge("retrieve", "is_relevant")

        g.add_conditional_edges(
            "is_relevant",
            route_after_relevance,
            {"generate_from_context": "generate_from_context", "no_answer_found": "no_answer_found"},
        )

        g.add_edge("no_answer_found", END)

        g.add_edge("generate_from_context", "is_sup")

        g.add_conditional_edges(
            "is_sup",
            route_after_issup,
            {"accept_answer": "is_use", "revise_answer": "revise_answer"},
        )

        g.add_edge("revise_answer", "is_sup")

        g.add_conditional_edges(
            "is_use",
            route_after_isuse,
            {"evaluate_answer": "evaluate_answer", "rewrite_question": "rewrite_question", "no_answer_found": "no_answer_found"},
        )

        g.add_edge("rewrite_question", "retrieve")

        g.add_edge("evaluate_answer", "save_semantic_cache")

        g.add_edge("save_semantic_cache", END)

        logging.info("LangGraph edges added successfully.")

        app = g.compile()

        logging.info("LangGraph compiled successfully.")

        return app

    except Exception as e:
        logging.error(f"Error while creating LangGraph: {str(e)}")
        raise CustomException(e)

# app = graph(retriever)