import logging
import json
from datetime import datetime, timezone
from src.state import State


class ProductionLogger:
    """
    Production logger for monitoring the RAG pipeline.
    """

    def __init__(self, log_file: str = "logs/rag_production.log"):
        self.logger = logging.getLogger("rag_production")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            formatter = logging.Formatter("%(message)s")

            file_handler = logging.FileHandler(
                log_file,
                encoding="utf-8"
            )

            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_query(self, state: State, metrics: dict):
        """
        Log a completed RAG query and its performance metrics.

        Args:
            state (State): Final LangGraph state.
            metrics (dict): Runtime metrics.
        """

        evaluation = state.get("evaluation", {})

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),

            # Query
            "question": state.get("question", ""),
            "optimized_query": state.get("optimized_query", ""),

            # Retrieval
            "need_retrieval": state.get("need_retrieval"),
            "docs_retrieved": len(state.get("docs", [])),
            "relevant_docs": len(state.get("relevant_docs", [])),

            # Answer
            "answer_length": len(state.get("answer", "")),
            "is_cached": state.get("cache_hit", False),

            # RAG verification
            "issup": state.get("issup"),
            "isuse": state.get("isuse"),

            # Evaluation
            "hallucination_score": (
                evaluation
                .get("hallucination", {})
                .get("hallucination_score")
            ),

            "groundedness_score": (
                evaluation
                .get("groundedness", {})
                .get("groundedness_score")
            ),

            "sycophancy_score": (
                evaluation
                .get("sycophancy", {})
                .get("sycophancy_score")
            ),

            "overall_quality": evaluation.get(
                "overall_quality"
            ),

            "is_reliable": evaluation.get(
                "is_reliable"
            ),

            # Performance
            "latency_ms": metrics.get("latency_ms"),
            "tokens_used": metrics.get("tokens_used"),
            "query_cost": metrics.get("query_cost"),
            "Completion tokens": metrics.get("Completion tokens"),
            "Prompt tokens": metrics.get("Prompt tokens"),
        }

        self.logger.info(
            json.dumps(
                log_entry,
                ensure_ascii=False
            )
        )